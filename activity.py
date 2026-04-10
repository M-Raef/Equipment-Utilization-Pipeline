"""
activity.py — Interaction-Based Activity Classifier (Mining Optimized)
----------------------------------------------------------------------
Optimized for high-capacity excavators and haulers.
Categories based on the 4-phase loading cycle seen in mining operations.
"""

from enum import Enum
from collections import deque

class Activity(str, Enum):
    WAITING          = "WAITING"
    MOVING           = "MOVING"
    DIGGING          = "DIGGING"
    SWINGING_LOADING = "SWINGING/LOADING"
    DUMPING          = "DUMPING"
    BEING_LOADED     = "BEING_LOADED"

def _is_loader(cls: str) -> bool:
    k = cls.lower()
    return "excavat" in k or "wheel" in k or "loader" in k

def _is_hauler(cls: str) -> bool:
    k = cls.lower()
    return "dump" in k or "truck" in k or "haul" in k

# ── Thresholds (MOG2 fg_fraction, 0.0–1.0) ─────────────────────────────────
TRACKS_FG_THR       = 0.06   # Zone 2 (lower) 
ARM_FG_THR          = 0.04   # Zone 0 (top) - lowered for slow arm movements
CAB_FG_THR          = 0.04   # Zone 1 (mid) - rotation indicator
DUMP_FG_THR         = 0.12   # High intensity motion splash in Zone 0
BEING_LOADED_FG_THR = 0.07   # Dirt falling into the truck bed

# ── Loader Logic (Excavators / Wheel Loaders) ──────────────────────────────

def _classify_loader(motion) -> Activity:
    """
    Priority: Tracks (Driving) > Cab (Rotation) > Arm (Work)
    """
    if not motion.is_active:
        return Activity.WAITING

    zones  = motion.zones
    arm    = zones[0] if len(zones) > 0 else None
    cab    = zones[1] if len(zones) > 1 else None
    tracks = zones[2] if len(zones) > 2 else None

    tracks_fg = tracks.fg_fraction if tracks else 0.0
    cab_fg    = cab.fg_fraction    if cab    else 0.0
    arm_fg    = arm.fg_fraction    if arm    else 0.0

    # 1. Machine is driving to a new spot
    if tracks_fg > TRACKS_FG_THR:
        return Activity.MOVING

    # 2. Machine is rotating (Cab motion is the primary indicator of Swinging)
    if cab_fg > CAB_FG_THR:
        return Activity.SWINGING_LOADING

    # 3. Machine is dumping (Intense pixel change in Zone 0 when bucket opens)
    if arm_fg > DUMP_FG_THR:
        return Activity.DUMPING

    # 4. Machine is reaching or pulling (Arm motion while stationary)
    if arm_fg > ARM_FG_THR:
        return Activity.DIGGING

    return Activity.WAITING

# ── Hauler Logic (Dump Trucks) ──────────────────────────────────────────────

def _classify_hauler(motion) -> Activity:
    """
    Stationary trucks detect dirt falling as foreground motion in the top zone.
    """
    if not motion.is_active:
        return Activity.WAITING

    zones  = motion.zones
    top    = zones[0] if len(zones) > 0 else None
    wheels = zones[2] if len(zones) > 2 else None

    wheels_fg = wheels.fg_fraction if wheels else 0.0
    top_fg    = top.fg_fraction    if top    else 0.0

    if wheels_fg > TRACKS_FG_THR:
        return Activity.MOVING

    # Parked + High motion at the top of the bed = Receiving material
    if wheels_fg <= TRACKS_FG_THR and top_fg > BEING_LOADED_FG_THR:
        return Activity.BEING_LOADED

    return Activity.WAITING

# ── Unified Classifier ──────────────────────────────────────────────────────

class ActivityClassifier:
    def __init__(self, class_name: str = "excavators", smooth_n: int = 7):
        self.class_name = class_name
        self._buf = deque(maxlen=smooth_n)
        self._fn  = (_classify_loader if _is_loader(class_name) 
                     else _classify_hauler if _is_hauler(class_name) 
                     else _classify_loader)

    def classify(self, motion) -> Activity:
        self._buf.append(self._fn(motion))
        return self._majority()

    def _majority(self) -> Activity:
        if not self._buf:
            return Activity.WAITING
        counts: dict = {}
        for a in self._buf:
            counts[a] = counts.get(a, 0) + 1
        return max(counts, key=counts.get)