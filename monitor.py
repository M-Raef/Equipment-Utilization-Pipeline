"""
monitor.py — Equipment State Manager (BoT-SORT + MOG2)
-------------------------------------------------------
Tracking  : YOLOv8 built-in BoT-SORT (no sort.py / reid.py needed)
Motion    : MOG2 background subtraction (no optical flow)
Grace     : machine stays ACTIVE for 30s after last real motion
"""

import time
import cv2
import numpy as np
from dataclasses import dataclass, field
from enum import Enum

from motion   import MOG2Engine, EquipmentMotionAnalyser
from activity import ActivityClassifier, Activity

# MOG2 runs on a half-resolution frame to save CPU
MOG2_SCALE    = 0.5
GRACE_SECONDS = 60.0


class UtilState(str, Enum):
    ACTIVE   = "ACTIVE"
    INACTIVE = "INACTIVE"


# ── Per-machine record ────────────────────────────────────────────────────────

@dataclass
class EquipmentRecord:
    original_id  : int
    class_name   : str
    state        : UtilState = UtilState.INACTIVE
    activity     : Activity  = Activity.WAITING
    motion_source: str       = "none"

    total_active_sec  : float = 0.0
    total_idle_sec    : float = 0.0
    total_tracked_sec : float = 0.0
    _last_motion_time : float = field(default_factory=time.time, repr=False)
    _ever_active      : bool  = field(default=False, repr=False)

    @property
    def utilization_pct(self) -> float:
        if self.total_tracked_sec < 0.01:
            return 0.0
        return round(100 * self.total_active_sec / self.total_tracked_sec, 1)

    def tick(self, dt: float, raw_is_active: bool) -> bool:
        now = time.time()
        
        # If MOG2 detects actual motion, reset the timer
        if raw_is_active:
            self._last_motion_time = now
            self._ever_active = True
            
        # Check if we are still within the 60-second grace window
        in_grace  = self._ever_active and (now - self._last_motion_time) < GRACE_SECONDS
        
        # The machine is effectively ACTIVE if it is moving OR in the grace period
        effective = raw_is_active or in_grace
        
        self.total_tracked_sec += dt
        if effective:
            self.total_active_sec += dt
        else:
            self.total_idle_sec   += dt
            
        return effective

    def summary(self) -> dict:
        return {
            "equipment_id"      : f"EQ-{self.original_id:03d}",
            "class"             : self.class_name,
            "state"             : self.state.value,
            "activity"          : self.activity.value,
            "motion_source"     : self.motion_source,
            "total_active_sec"  : round(self.total_active_sec,  1),
            "total_idle_sec"    : round(self.total_idle_sec,    1),
            "total_tracked_sec" : round(self.total_tracked_sec, 1),
            "utilization_pct"   : self.utilization_pct,
        }


# ── Monitor ───────────────────────────────────────────────────────────────────

class EquipmentMonitor:
    """
    Call update() once per frame with the YOLO track results and the raw frame.
    BoT-SORT IDs come directly from YOLO — no custom tracker needed.
    """

    def __init__(self,
                 motion_threshold: float = 0.05,
                 device          : str   = "cpu"):   # device kept for API compat

        self.mog2_engine = MOG2Engine(
            history        = 200,
            var_threshold  = 40.0,
            detect_shadows = False,
        )

        self.motion_threshold = motion_threshold

        # Per-track objects (keyed by BoT-SORT track ID)
        self._motion_analysers    : dict[int, EquipmentMotionAnalyser] = {}
        self._activity_classifiers: dict[int, ActivityClassifier]      = {}
        self._records             : dict[int, EquipmentRecord]         = {}

        self._frame_idx = 0

    # ── Main update ───────────────────────────────────────────────────────────

    def update(self,
               frame_bgr  : np.ndarray,
               track_results,            # raw output of model.track()
               fps        : float = 30.0) -> list[dict]:
        """
        Parameters
        ----------
        frame_bgr     : current BGR frame
        track_results : list returned by model.track() — each element is a
                        ultralytics Results object with .boxes containing
                        .id (track id), .xyxy, .cls, .conf
        fps           : video fps for dt calculation

        Returns
        -------
        list of result dicts, one per confirmed track
        """
        dt = 1.0 / max(fps, 1.0)
        self._frame_idx += 1

        # ── MOG2 mask on downscaled grey frame ────────────────────────────
        small     = cv2.resize(frame_bgr, (0, 0), fx=MOG2_SCALE, fy=MOG2_SCALE)
        gray      = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        mog2_mask = self.mog2_engine.apply(gray)   # binary mask, same size as `small`

        # ── Parse YOLO track results ──────────────────────────────────────
        results = []
        for r in track_results:
            if r.boxes is None or r.boxes.id is None:
                continue   # no confirmed tracks in this result

            boxes   = r.boxes.xyxy.cpu().numpy()      # (N,4)
            ids     = r.boxes.id.cpu().numpy().astype(int)   # (N,)
            classes = r.boxes.cls.cpu().numpy().astype(int)  # (N,)
            names   = r.names                         # dict {int: str}

            for bbox, tid, cid in zip(boxes, ids, classes):
                x1, y1, x2, y2 = bbox.tolist()
                cls_name = names[cid]

                # Create per-track objects on first appearance
                if tid not in self._motion_analysers:
                    self._motion_analysers[tid] = EquipmentMotionAnalyser(
                        motion_threshold=self.motion_threshold,
                        smoothing_frames=5,
                    )
                    self._activity_classifiers[tid] = ActivityClassifier(
                        class_name=cls_name, smooth_n=5,
                    )

                analyser = self._motion_analysers[tid]
                act_clf  = self._activity_classifiers[tid]

                # Scale bbox to match the downscaled MOG2 mask
                scaled = [v * MOG2_SCALE for v in [x1, y1, x2, y2]]
                frame_motion = analyser.analyse(mog2_mask, scaled)
                raw_active   = frame_motion.is_active
                motion_src   = frame_motion.motion_source.value
                activity     = act_clf.classify(frame_motion)

                # Ensure record exists
                if tid not in self._records:
                    self._records[tid] = EquipmentRecord(
                        original_id=tid, class_name=cls_name
                    )
                rec = self._records[tid]

                effective_active = rec.tick(dt, raw_active)
                rec.state         = UtilState.ACTIVE if effective_active else UtilState.INACTIVE
                rec.activity      = activity if effective_active else Activity.WAITING
                rec.motion_source = motion_src
                rec.class_name    = cls_name

                results.append({
                    "frame_id"    : self._frame_idx,
                    "track_id"    : tid,
                    "equipment_id": f"EQ-{tid:03d}",
                    "class"       : cls_name,
                    "bbox"        : [x1, y1, x2, y2],
                    **rec.summary(),
                })

        return results

    # ── Draw ─────────────────────────────────────────────────────────────────

    def draw(self, frame_bgr: np.ndarray, results: list[dict]) -> np.ndarray:
        vis  = frame_bgr.copy()
        FONT = cv2.FONT_HERSHEY_SIMPLEX
        AA   = cv2.LINE_AA

        ACTIVITY_COLORS = {
            "WAITING"      : (160, 160, 160),
            "MOVING"       : (0,   200, 255),
            "LOADING_DIRT" : (0,   210,  80),
            "BEING_LOADED" : (0,   180, 255),
        }

        for r in results:
            x1, y1, x2, y2 = [int(v) for v in r["bbox"]]
            x1 = max(0, x1);  y1 = max(0, y1)
            x2 = min(vis.shape[1]-1, x2)
            y2 = min(vis.shape[0]-1, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            is_active = r["state"] == "ACTIVE"
            box_color = (0, 210, 80) if is_active else (30, 80, 220)
            act_color = ACTIVITY_COLORS.get(r["activity"], (200, 200, 200))

            # Bounding box
            cv2.rectangle(vis, (x1, y1), (x2, y2), box_color, 2)

            # Semi-transparent dark panel at top of box
            panel_y2 = min(y1 + 60, y2)
            roi = vis[y1:panel_y2, x1:x2]
            if roi.size > 0:
                vis[y1:panel_y2, x1:x2] = cv2.addWeighted(
                    roi, 0.35, np.zeros_like(roi), 0.65, 0
                )

            # Line 1 — ID  CLASS  STATE
            cv2.putText(vis,
                        f"{r['equipment_id']}  {r['class'].upper()}  {r['state']}",
                        (x1+5, y1+16), FONT, 0.48, box_color, 1, AA)

            # Line 2 — Activity (colour-coded)
            cv2.putText(vis,
                        f"Activity: {r['activity']}",
                        (x1+5, y1+33), FONT, 0.44, act_color, 1, AA)

            # Line 3 — Utilisation stats
            cv2.putText(vis,
                        f"Util:{r['utilization_pct']}%  "
                        f"A:{r['total_active_sec']}s  "
                        f"I:{r['total_idle_sec']}s",
                        (x1+5, y1+50), FONT, 0.40, (200, 200, 200), 1, AA)

        return vis

    def get_all_summaries(self) -> list[dict]:
        return [rec.summary() for rec in self._records.values()]