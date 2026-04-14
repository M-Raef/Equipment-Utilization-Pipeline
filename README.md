# 🚜Real-Time Equipment Utilization Pipeline

A real-time, microservices-based computer vision pipeline designed to track and analyze heavy construction equipment utilization. This system processes video feeds to detect machinery, track utilization states (Active/Inactive), classify specific work activities, and stream live analytics to an Apache Kafka broker, persisting data to a PostgreSQL database with a live Streamlit dashboard.

---

## 🏗️ Architecture & Design Decisions (Technical Write-up)

To meet the requirements of real-time processing and articulated motion detection, several specific engineering trade-offs and design decisions were made:

### 1. Solving Articulated Equipment Motion
**The Challenge:** Standard object detection struggles when a machine's tracks are stationary but its arm is actively digging. If the overall bounding box doesn't move across the screen, standard optical flow often flags the machine as incorrectly "Idle."

**The Solution:** I implemented **Region-Based Motion Analysis** using a Mixture of Gaussians (MOG2) background subtractor. Each bounding box is dynamically split into three vertical zones:
* **Zone 0 (Top):** Arm / Bucket / Load Bed
* **Zone 1 (Middle):** Cab / Upper Structure
* **Zone 2 (Bottom):** Tracks / Wheels

By independently monitoring the `fg_fraction` (foreground pixel density) in these zones, the pipeline accurately detects `ACTIVE` states even if only the excavator's arm is moving.

### 2. Activity Classification Mechanics
Rather than relying on fragile pixel-direction heuristics, activity classification utilizes an **Interaction-Based Logic** model mapped to the 3-zone architecture:
* **DIGGING:** High foreground motion in Zone 0 (Arm) while Zone 1 (Cab) remains relatively still.
* **SWINGING/LOADING:** Simultaneous motion in Zone 0 and Zone 1 (indicates the upper structure is rotating).
* **DUMPING:** A high-intensity "pixel splash" (>0.12 density) in Zone 0 as the bucket opens and releases material.
* **BEING_LOADED (Haulers):** Zone 2 (Wheels) is stationary, but Zone 0 (Load Bed) registers high motion from falling dirt.

### 3. Model Selection: YOLO26
I selected **YOLO26** for this prototype due to its native **NMS-Free End-to-End architecture**. By eliminating the Non-Maximum Suppression post-processing step, the model achieves significantly lower latency. This ensures true real-time performance on edge devices and provides up to 43% faster inference as a CPU fallback.

### 4. Tracking, Occlusion, and Re-ID
When machines hide behind dirt piles, standard trackers assign a new ID upon reappearance, resetting utilization timers. To solve this, tracking is handled by YOLO's built-in **BoT-SORT**. I implemented a custom configuration (`custom_botsort.yaml`) that increases the `track_buffer` to 150 frames. This provides a 5-second tracking "memory," ensuring IDs and utilization timers remain consistent through heavy occlusion.

---

## 📁 System Components & Execution Modes

The system is designed as a distributed microservice architecture.

### 1. `main.py` (Live AI Core)
This is the core CV microservice script. It runs the YOLO26 model and the MOG2 motion engine in true real-time, processing every frame as it arrives and streaming structured JSON payloads to the Kafka broker.

### 2. `database_sink.py` (Data Persistence)
A backend consumer service that listens to the `equipment_events` Kafka topic and permanently writes the utilization data to a PostgreSQL/TimescaleDB table.

### 3. `dashboard.py` (Analytics UI)
A Streamlit frontend that queries the PostgreSQL database to display a live video feed, real-time machine status, and a utilization percentage dashboard.

### 4. `process_video.py` (Offline Batch Processor)
A utility script designed for high-resolution video rendering. It disables the Kafka streaming and live UI to focus 100% of the GPU compute power on rendering a fully annotated output `.mp4` file.

---

## 🚀 Installation & Setup

### 1. Clone & Environment Setup
```bash
git clone [https://github.com/M-Raef/Equipment-Utilization-Pipeline.git](https://github.com/M-Raef/Equipment-Utilization-Pipeline.git)
cd Equipment-Utilization-Pipeline
python -m venv env

# Activate environment (Windows)
.\env\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt

# For NVIDIA GPU Acceleration (Highly Recommended):
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu126](https://download.pytorch.org/whl/cu126)
```

### 3. Start the Infrastructure (Docker)
Ensure Docker Desktop is running, then spin up the Kafka and PostgreSQL containers:
```bash
docker-compose up -d
```
---

## ⚙️ Running the Full System
To run the live pipeline, you will need to open three separate terminals (ensure your virtual environment is activated in all of them).

- Terminal 1: Start the Database Sink
```bash
python database_sink.py
```

- Terminal 2: Launch the Dashboard
```bash
streamlit run dashboard.py
```

- Terminal 3: Start the AI Engine
```bash
python main.py --source "testingVideo-3_HD.mp4" --model "equipment-utilization-pipeline-weights.pt"
```
---

## 📺 Visual Demonstration

Due to GitHub's file size limitations, the full high-definition annotated demo video is hosted on Google Drive. This video showcases real-time activity classification, 3-zone motion detection, and utilization tracking in action.

👉 [**Watch the Output Demo Video (Annotated)**](https://drive.google.com/file/d/1eXEF9nGOt2Q5GihNemebLWQ_XSEv9QQ7/view?usp=sharing)

---

## 📥 Dataset & Source Video

If you wish to run the pipeline yourself using the same source material seen in the demo, you can download the raw FHD video here:

👉 [**Download Source Video (Raw HD)**](https://drive.google.com/file/d/1f8q34Xa4GByvYM-q6Ku0x_zHWprsGe-K/view?usp=sharing)

Place the downloaded file in the root directory and use the `--source` flag as detailed in the Usage section.
