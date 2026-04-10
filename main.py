"""
main.py — Live Real-Time Pipeline (RTX 4060 Optimized)
------------------------------------------------------------------
Key changes:
  1. Processes EVERY frame to prevent tracker physics desync (No double boxes).
  2. Perfectly formats the Kafka JSON payload per assessment requirements.
  3. Includes cv2.WINDOW_NORMAL to prevent the zoomed-in screen bug.
"""

import argparse
import json
import time
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from monitor import EquipmentMonitor

def parse_args():
    p = argparse.ArgumentParser()
    # Point this to your CLEAN, unprocessed video or camera (0)
    p.add_argument("--source",      type=str,   default="0")
    p.add_argument("--model",       type=str,   default="yolo26n.pt") 
    p.add_argument("--conf",        type=float, default=0.45) 
    p.add_argument("--iou",         type=float, default=0.40) 
    p.add_argument("--imgsz",       type=int,   default=640)
    p.add_argument("--device",      type=str,   default="cuda")
    p.add_argument("--motion-thr",  type=float, default=0.05) 
    p.add_argument("--no-show",     action="store_true")
    return p.parse_args()


def setup_device(requested: str) -> str:
    if requested == "cuda":
        if not torch.cuda.is_available():
            print("[WARN] CUDA not available — falling back to CPU")
            return "cpu"
        torch.cuda.set_device(0)
        _ = torch.zeros(1, device="cuda")
        torch.cuda.synchronize()
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
    return requested


def warmup_model(model: YOLO, device: str, imgsz: int):
    dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    model.predict(
        source=dummy, device=device, imgsz=imgsz,
        half=("cuda" in device), verbose=False,
    )


class KafkaPublisher:
    def __init__(self, topic="equipment_events", bootstrap="localhost:9092"):
        self._producer = None
        self._topic = topic
        try:
            from kafka import KafkaProducer
            self._producer = KafkaProducer(
                bootstrap_servers=[bootstrap],
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                linger_ms=50,
            )
            print(f"[Kafka] Connected → topic='{topic}'")
        except Exception as e:
            print(f"[Kafka] Not available ({e})")

    def send(self, payload):
        if self._producer:
            self._producer.send(self._topic, payload)

    def flush(self):
        if self._producer:
            self._producer.flush()


def main():
    args   = parse_args()
    device = setup_device(args.device)

    print(f"[INFO] Loading YOLO: {args.model}")
    model = YOLO(args.model)
    warmup_model(model, device, args.imgsz)

    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {args.source}")

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"[INFO] {width}x{height} @ {fps:.1f} fps | device={device} | Real-Time Mode")

    # --- FIX FOR THE "ZOOMED IN" WINDOW ---
    if not args.no_show:
        window_name = "Eagle Vision - Real-Time Dashboard"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720) 

    monitor = EquipmentMonitor(motion_threshold=args.motion_thr, device=device)
    kafka = KafkaPublisher()

    frame_idx  = 0
    t_start    = time.perf_counter()
    disp_count = 0

    print("\n[INFO] Running Pipeline. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_idx += 1

        # We process EVERY frame for real-time accuracy and smooth tracker physics
        track_out = model.track(
            source      = frame,
            conf        = args.conf,
            iou         = args.iou,
            imgsz       = args.imgsz,
            device      = device,
            half        = ("cuda" in device),
            persist     = True,
            verbose     = False,
            tracker     = "custom_botsort.yaml", 
        )
        
        # Update motion and activity states
        last_results = monitor.update(frame, track_out, fps=fps)

        # # ── KAFKA STREAMING [cite: 26-42] ──
        # for r in last_results:
        #     total_seconds = frame_idx / fps
        #     hours, remainder = divmod(total_seconds, 3600)
        #     minutes, seconds = divmod(remainder, 60)
        #     timestamp = f"{int(hours):02d}:{int(minutes):02d}:{seconds:06.3f}"

        #     kafka.send({
        #         "frame_id"       : r["frame_id"],
        #         "equipment_id"   : r["equipment_id"],
        #         "equipment_class": r["class"],
        #         "timestamp"      : timestamp,
        #         "utilization"    : {},
        #         "current_state"  : r["state"],
        #         "current_activity": r["activity"],
        #         "motion_source"  : r["motion_source"],
        #         "time_analytics" : {
        #             "total_tracked_seconds": r["total_tracked_sec"],
        #             "total_active_seconds" : r["total_active_sec"],
        #             "total_idle_seconds"   : r["total_idle_sec"],
        #             "utilization_percent"  : r["utilization_pct"],
        #         },
        #     })

        # ── KAFKA STREAMING ──
        for r in last_results:
            total_seconds = frame_idx / fps
            hours, remainder = divmod(total_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            timestamp = f"{int(hours):02d}:{int(minutes):02d}:{seconds:06.3f}"

            # Extract just the number from "EQ-001" to make it "EX-001"
            eq_num = int(r["equipment_id"].split("-")[1])
            
            kafka.send({
                "frame_id": frame_idx,
                "equipment_id": f"EX-{eq_num:03d}", # Matches the requested EX-001 format
                "equipment_class": r["class"],
                "timestamp": timestamp,
                "utilization": {
                    "current_state": r["state"],
                    "current_activity": r["activity"],
                    "motion_source": r["motion_source"],
                },
                "time_analytics": {
                    "total_tracked_seconds": round(r["total_tracked_sec"], 1),
                    "total_active_seconds": round(r["total_active_sec"], 1),
                    "total_idle_seconds": round(r["total_idle_sec"], 1),
                    "utilization_percent": r["utilization_pct"]
                }
            })

        # Draw the clean annotations on a fresh copy of the frame
        annotated = monitor.draw(frame.copy(), last_results)
        
        #new
        # In main.py
        # Lower the quality to 50% for the dashboard to make it MUCH faster
        cv2.imwrite("latest_frame.jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 50])

        # Draw ONLY ONE text overlay for live performance status
        disp_count += 1
        elapsed = time.perf_counter() - t_start
        current_fps = disp_count / max(elapsed, 1e-6)
        
        status_text = f"Live Processing | Frame: {frame_idx} | Pipeline FPS: {current_fps:.1f}"
        cv2.putText(annotated, status_text,
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (255, 255, 255), 2, cv2.LINE_AA)

        if not args.no_show:
            cv2.imshow(window_name, annotated)
            # 1ms delay is enough for real-time playback
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    kafka.flush()
    cap.release()
    cv2.destroyAllWindows()
    print("\n[INFO] Real-Time Processing Complete!")

if __name__ == "__main__":
    main()


#python main.py --source "C:\Users\raefa\OneDrive\Desktop\Raef\RAEF\equipment-utilization-pipeline\Codes\testingVideo-3_HD.mp4" --model "C:\Users\raefa\OneDrive\Desktop\Raef\RAEF\equipment-utilization-pipeline\Codes\equipment-utilization-pipeline-weights.pt"