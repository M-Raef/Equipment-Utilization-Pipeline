"""
motion.py — MOG2 Background Subtraction Motion Analyser
---------------------------------------------------------
Replaces Farneback optical flow entirely.

WHY MOG2 OVER OPTICAL FLOW FOR THIS PROBLEM:
  Farneback optical flow computes a motion vector for EVERY pixel by comparing
  texture gradients between frames. When a yellow machine sits on yellow dirt,
  the gradients are similar and flow vectors become unreliable (false INACTIVE).

  MOG2 (Mixture of Gaussians v2) instead builds a statistical background model
  by learning what each pixel looks like when nothing is moving. Any pixel that
  deviates significantly from its learned background is marked as "foreground"
  (white in the mask). This is much more robust to colour similarity because
  it compares CHANGE over time, not texture similarity between neighbours.

ZONE LAYOUT (top → bottom, 3 equal horizontal strips):
  Zone 0  top 33%   arm / boom / bucket / top of load bed
  Zone 1  mid 33%   cab / upper body
  Zone 2  bot 33%   tracks / wheels / lower frame

ACTIVE DECISION:
  A zone is active if the fraction of white (foreground) pixels > MOV_THRESHOLD.
  Default 5% — meaning at least 1 in 20 pixels in the zone must be "moving".
  Kept deliberately low because construction equipment is large but moves slowly.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from enum import Enum
from collections import deque


# ── Tunable constants ────────────────────────────────────────────────────────
MOV_THRESHOLD = 0.05   # fraction of white pixels to call a zone active (5%)
NUM_ZONES     = 3      # vertical zones inside each bounding box
SMOOTH_N      = 5      # frames to smooth zone-active signal (reduces flicker)


class MotionSource(str, Enum):
    NONE        = "none"
    ARM_ONLY    = "arm_only"
    TRACKS_ONLY = "tracks_only"
    ARM_BODY    = "arm_body"
    FULL        = "full"


@dataclass
class ZoneMotion:
    """MOG2-based motion result for one zone."""
    fg_fraction : float   # fraction of foreground (white) pixels  0.0–1.0
    active      : bool    # True if fg_fraction > MOV_THRESHOLD


@dataclass
class FrameMotion:
    """All-zone result for one tracked object in one frame."""
    zones         : list[ZoneMotion]   # [zone0(top), zone1(mid), zone2(bot)]
    is_active     : bool
    motion_source : MotionSource
    overall_fg    : float              # mean fg_fraction across all zones


# ── MOG2 Engine ──────────────────────────────────────────────────────────────

class MOG2Engine:
    """
    Wraps cv2.createBackgroundSubtractorMOG2.

    One shared instance per video — MOG2 learns the background
    from the whole scene and gets better over time.

    Parameters
    ----------
    history      : how many past frames feed the background model
                   (longer = more stable, slower to adapt to lighting changes)
    var_threshold: Mahalanobis distance threshold for foreground decision
                   higher = less sensitive (fewer false positives on static scenes)
    detect_shadows: mark shadows grey (127) instead of white — we convert to
                   binary so set False to save CPU
    """

    def __init__(self,
                 history       : int  = 200,
                 var_threshold : float = 40.0,
                 detect_shadows: bool  = False):
        self._mog2 = cv2.createBackgroundSubtractorMOG2(
            history       = history,
            varThreshold  = var_threshold,
            detectShadows = detect_shadows,
        )
        # Small morphological kernel to remove salt-and-pepper noise
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def apply(self, frame_gray: np.ndarray) -> np.ndarray:
        """
        Feed the current frame to MOG2 and return a clean binary mask.
        White pixels (255) = foreground / moving.
        Black pixels (0)   = background / static.
        """
        mask = self._mog2.apply(frame_gray)          # raw MOG2 output
        # Remove tiny noise blobs (isolated pixels from compression artefacts)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self._kernel)
        # Fill small holes inside moving objects (connected regions)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)
        return mask   # dtype uint8, values 0 or 255


# ── Per-track motion analyser ─────────────────────────────────────────────────

class EquipmentMotionAnalyser:
    """
    Applies a shared MOG2 mask to a specific bounding box and
    measures the foreground pixel fraction in each vertical zone.

    Parameters
    ----------
    motion_threshold : fg_fraction above which a zone is "active"
    smoothing_frames : frames to average the zone-active boolean over
    num_zones        : number of vertical strips (default 3)
    """

    def __init__(self,
                 motion_threshold: float = MOV_THRESHOLD,
                 smoothing_frames: int   = SMOOTH_N,
                 num_zones       : int   = NUM_ZONES):
        self.threshold    = motion_threshold
        self.smooth_n     = smoothing_frames
        self.num_zones    = num_zones

        # Per-zone fg_fraction history for smoothing
        self._fg_history: list[deque] = [
            deque(maxlen=smoothing_frames) for _ in range(num_zones)
        ]

    def reset(self):
        for h in self._fg_history:
            h.clear()

    def analyse(self, mog2_mask: np.ndarray, bbox: list) -> FrameMotion:
        """
        Parameters
        ----------
        mog2_mask : binary mask (H_frame, W_frame) from MOG2Engine.apply()
        bbox      : [x1, y1, x2, y2] in the SAME coordinate space as mog2_mask
                    (caller must scale if mask was computed on a resized frame)

        Returns
        -------
        FrameMotion with per-zone activity breakdown
        """
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1 = max(0, x1);  y1 = max(0, y1)
        x2 = min(mog2_mask.shape[1] - 1, x2)
        y2 = min(mog2_mask.shape[0] - 1, y2)

        if x2 <= x1 or y2 <= y1:
            return self._inactive_result()

        roi  = mog2_mask[y1:y2, x1:x2]   # crop mask to bounding box
        roi_h = roi.shape[0]

        zones: list[ZoneMotion] = []
        zone_h = roi_h / self.num_zones

        for z in range(self.num_zones):
            zy1 = int(z * zone_h)
            zy2 = int((z + 1) * zone_h)
            zone_roi = roi[zy1:zy2, :]

            if zone_roi.size == 0:
                zones.append(ZoneMotion(0.0, False))
                continue

            # Fraction of pixels that are foreground (255)
            fg_frac = float(np.count_nonzero(zone_roi)) / float(zone_roi.size)

            # Smooth over recent frames
            self._fg_history[z].append(fg_frac)
            smooth_fg = float(np.mean(self._fg_history[z]))

            zones.append(ZoneMotion(
                fg_fraction = smooth_fg,
                active      = smooth_fg > self.threshold,
            ))

        return self._classify(zones)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _inactive_result(self) -> FrameMotion:
        zones = [ZoneMotion(0.0, False) for _ in range(self.num_zones)]
        return FrameMotion(zones, False, MotionSource.NONE, 0.0)

    def _classify(self, zones: list[ZoneMotion]) -> FrameMotion:
        arm    = zones[0]
        cab    = zones[1]
        tracks = zones[2] if len(zones) > 2 else ZoneMotion(0.0, False)

        is_active = any(z.active for z in zones)

        if not is_active:
            source = MotionSource.NONE
        elif arm.active and not tracks.active:
            source = MotionSource.ARM_ONLY
        elif tracks.active and not arm.active:
            source = MotionSource.TRACKS_ONLY
        elif arm.active and cab.active and not tracks.active:
            source = MotionSource.ARM_BODY
        else:
            source = MotionSource.FULL

        overall = float(np.mean([z.fg_fraction for z in zones]))
        return FrameMotion(zones=zones, is_active=is_active,
                           motion_source=source, overall_fg=overall)