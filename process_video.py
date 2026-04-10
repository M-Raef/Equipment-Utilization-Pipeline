"""
process_video.py — Offline batch processor (BoT-SORT + MOG2)
--------------------------------------------------------------
Edit VIDEO_PATH, MODEL_PATH, OUTPUT_PATH below, then run:
    python process_video.py
"""

import time
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from monitor import EquipmentMonitor

# ── ✏️  EDIT THESE ───────────────────────────────────────────────────────────
VIDEO_PATH  = r"C:\Users\raefa\OneDrive\Desktop\Raef\RAEF\equipment-utilization-pipeline\Codes\testingVideo-3_HD.mp4"
MODEL_PATH  = r"C:\Users\raefa\OneDrive\Desktop\Raef\RAEF\equipment-utilization-pipeline\Codes\equipment-utilization-pipeline-weights.pt"
OUTPUT_PATH = r"C:\Users\raefa\OneDrive\Desktop\Raef\RAEF\equipment-utilization-pipeline\Codes\outputs\testingVideo-3_HD.mp4"   # where the result is saved

# ─────────────────────────────────────────────────────────────────────────────

CONF          = 0.45
IOU_NMS       = 0.40
IMGSZ         = 640
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
PROCESS_EVERY = 2     # run model.track() every N frames


def setup_cuda(device: str) -> str:
    if device == "cuda":
        if not torch.cuda.is_available():
            print("[WARN] CUDA not found — using CPU")
            return "cpu"
        torch.cuda.set_device(0)
        _ = torch.zeros(1, device="cuda")
        torch.cuda.synchronize()
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
    return device


def main():
    device = setup_cuda(DEVICE)

    print(f"[INFO] Loading model : {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    # Warm-up inference
    model.predict(
        source=np.zeros((IMGSZ, IMGSZ, 3), dtype=np.uint8),
        device=device, imgsz=IMGSZ,
        half=("cuda" in device), verbose=False,
    )

    print(f"[INFO] Opening video : {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {VIDEO_PATH}")

    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[INFO] {width}x{height} @ {fps:.1f} fps | "
          f"{total_frames} frames | device={device}")
    print(f"[INFO] Output → {OUTPUT_PATH}\n")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

    monitor = EquipmentMonitor(motion_threshold=0.05, device=device)

    frame_idx    = 0
    last_results = []
    t_start      = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # ── model.track() every PROCESS_EVERY frames ─────────────────────
        # ⬇️  THIS IS THE LINE TO CHANGE to use your custom BoT-SORT config:
        #     Add  tracker="custom_botsort.yaml"  to model.track()
        if frame_idx % PROCESS_EVERY == 0:
            track_out = model.track(
                source      = frame,
                conf        = CONF,
                iou         = IOU_NMS,
                imgsz       = IMGSZ,
                device      = device,
                half        = ("cuda" in device),
                persist     = True,      # REQUIRED: keeps track IDs consistent frame-to-frame
                verbose     = False,
                tracker     = "custom_botsort.yaml",  # ← custom buffer + GMC config
                agnostic_nms= True,
            )
            last_results = monitor.update(frame, track_out, fps=fps)

        # ── Annotate every frame ──────────────────────────────────────────
        annotated = monitor.draw(frame, last_results)

        elapsed = time.perf_counter() - t_start
        pfps    = frame_idx / max(elapsed, 1e-6)
        eta     = (total_frames - frame_idx) / max(pfps, 1e-6)
        cv2.putText(annotated,
                    f"Processing {frame_idx}/{total_frames}  "
                    f"({pfps:.1f} fps)  ETA {eta:.0f}s",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 255, 255), 2, cv2.LINE_AA)

        writer.write(annotated)

        if frame_idx % 30 == 0 or frame_idx == total_frames:
            pct = frame_idx / total_frames * 100
            bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
            print(f"\r[{bar}] {pct:5.1f}%  {frame_idx}/{total_frames}"
                  f"  {pfps:.1f} fps  ETA {eta:.0f}s   ",
                  end="", flush=True)

    cap.release()
    writer.release()

    total_time = time.perf_counter() - t_start
    print(f"\n\n[DONE] {frame_idx} frames in {total_time:.1f}s → {OUTPUT_PATH}")

    summaries = monitor.get_all_summaries()
    print("\n" + "═"*76)
    print(f"  {'ID':<8} {'Class':<14} {'State':<10} {'Activity':<16} "
          f"{'Active':>8} {'Idle':>8} {'Util%':>7}")
    print("  " + "─"*76)
    for s in summaries:
        print(f"  {s['equipment_id']:<8} {s['class']:<14} {s['state']:<10} "
              f"{s['activity']:<16} "
              f"{s['total_active_sec']:>7.1f}s "
              f"{s['total_idle_sec']:>7.1f}s "
              f"{s['utilization_pct']:>6.1f}%")
    print("═"*76 + "\n")


if __name__ == "__main__":
    main()