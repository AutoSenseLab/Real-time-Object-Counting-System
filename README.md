# Real-time Object Counting System

ระบบนับ object แบบ real-time โดยใช้ YOLOv8 + OpenCV
รองรับหลาย input และ track แต่ละตัวด้วย unique ID

---

## ทำไมถึงทำ project นี้

งานด้าน Industrial AI หลายอย่างต้องการระบบนับที่แม่นยำและ real-time
ไม่ว่าจะเป็นนับชิ้นงานบนสายพาน นับคนเข้าออกพื้นที่ หรือนับรถในลานจอด

project นี้สร้างขึ้นเพื่อแสดงให้เห็นว่าสามารถออกแบบและ implement
ระบบแบบนี้ได้จริง ตั้งแต่ detection ไปจนถึง tracking และ counting logic

---

## Demo

![Demo](Demo/test_output.gif)

ทดสอบกับ video จำลอง (pedestrian tracking)
เนื่องจาก data จริงจากสถานที่จริงไม่สามารถเปิดเผยได้

---

## Features

- detect และนับ object แบบ real-time ด้วย YOLOv8
- track แต่ละตัวด้วย unique ID — ไม่นับซ้ำเมื่อ object เคลื่อนที่
- แสดง count บนหน้าจอแบบ overlay
- รองรับ webcam, video file และ RTSP / IP camera
- export ผลลัพธ์เป็น CSV log

---

## Installation

```bash
git clone https://github.com/AutoSenseLab/Real-time-Object-Counting-System.git
cd Real-time-Object-Counting-System

pip install -r requirements.txt
```

---

## วิธีใช้

```bash
# Webcam
python main.py --source 0

# Video file
python main.py --source video.mp4

# IP Camera
python main.py --source rtsp://192.168.1.1/stream

# Export log
python main.py --source video.mp4 --export
```

---

## Tech Stack

```
YOLOv8 (Ultralytics)  —  object detection
OpenCV                —  video processing
Python                —  core logic
```

---

## Project Structure

```
Real-time-Object-Counting-System/
├── main.py          ← entry point
├── tracker.py       ← tracking logic
├── counter.py       ← counting logic
├── utils.py         ← helper functions
├── requirements.txt
└── README.md
```

---

## Use Cases

- นับชิ้นงานบนสายพานการผลิต
- นับคนเข้าออกพื้นที่
- นับรถในลานจอด
- นับสินค้าในโกดัง

---

**AutoSenseLab** · [GitHub](https://github.com/AutoSenseLab) · [Email](mailto:ditsayabodin12@gmail.com)
