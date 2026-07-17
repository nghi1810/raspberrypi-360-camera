# 🚗 Raspberry Pi 360° Camera & ADAS

![Raspberry Pi](https://img.shields.io/badge/Raspberry%20Pi-5%20(8GB)-C51A4A?logo=raspberry-pi&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8.svg)
![Hardware](https://img.shields.io/badge/Hardware-ADAS-success.svg)

A custom Advanced Driver Assistance System (ADAS) utilizing four fisheye cameras, ultrasound sensors, and computer vision to create a 360-degree bird's-eye view and a front/rear collision warning system. 

<div align="center">
  <img width="803" alt="System Overview" src="https://github.com/user-attachments/assets/114b3154-0019-4b1b-9aae-053e1df17feb" />
</div>

---

## 📖 Project Overview & Prerequisites

Building a 360-degree camera system for a vehicle is a complex multidisciplinary project involving hardware assembly, PCB design, 3D printing, Arduino programming, and Python computer vision pipelines. 

**Before starting:** It is highly recommended to spend your first few weeks researching ADAS architectures, 360-degree camera stitching techniques, and Python-based computer vision optimization. 

---

## 🛠️ Hardware Requirements & Budget

The estimated cost for this project is around **3–4 million VND**. 

* **Compute Unit:** Raspberry Pi 5 (8GB version). *(Note: For higher performance, an automotive display with an integrated CPU or an NVIDIA Jetson Nano is highly recommended).*
* **Cameras:** 4 × USB Fisheye Cameras. 
* **Sensors:** Ultrasound sensors for proximity warnings.

> **Sourcing Note:** Finding true USB fisheye cameras can be difficult, as most modules on the market are standard lenses. Try searching on Amazon or regional platforms like Shopee (in Southeast Asia) for the correct hardware.

<div align="center">
  <img width="403" alt="Hardware Setup 1" src="https://github.com/user-attachments/assets/f163ede2-df87-4609-9175-7cc72636adeb" />
  <img width="225" alt="Hardware Setup 2" src="https://github.com/user-attachments/assets/ae260fc0-355d-42e3-8a13-381707f1e1a3" />
</div>

---

## 📐 Calibration & Undistortion

Accurate camera calibration is the most critical step for seamless image stitching. You must be patient and record all intrinsic and extrinsic parameters meticulously.

* **Vehicle-Specific:** You **cannot** reuse the $K$ (intrinsic) and $D$ (distortion) matrices across different setups. Each vehicle has unique dimensions (length, width, height) and camera mounting angles.
* **Process:** Follow the provided processing scripts step-by-step to calculate the correct distortion parameters for your specific vehicle.

<div align="center">
  <img width="521" alt="Calibration Process" src="https://github.com/user-attachments/assets/3bde1f0e-820d-4923-96c5-faa121eb5c96" />
</div>

---

## ⚡ Performance Optimization

Developing this system involved navigating cross-platform challenges across macOS, Windows, and Raspberry Pi OS (Linux). 

To overcome hardware limitations and reduce latency:
* **Concurrency:** The codebase heavily utilizes `multiprocessing` and `multithreading` to separate image capture, processing, and display pipelines.
* **Current Performance:** The system currently runs at **12–20 FPS** on the Raspberry Pi 5. While this sounds low, it represents the absolute ceiling of hardware and code optimization for four simultaneous USB camera streams on this specific board.

<div align="center">
  <img width="779" alt="Bird's Eye View Output" src="https://github.com/user-attachments/assets/3f459001-b296-4ef6-a396-dbe12a76d4fc" />
</div>

---

## 💡 Development Guidelines & Best Practices

If you are working in a team or attempting to replicate this build, keep these principles in mind:

1. **System Architecture:** Plan a clear architecture from day one (camera placement, wire routing, data flow) before writing code.
2. **Modular Testing:** Test every sensor, camera, and script independently before integrating them into the master system.
3. **Meticulous Calibration:** Spend extra time ensuring your camera calibration is perfect; this dictates the quality of your distance estimation and stitching.
4. **Task Delegation:** Divide the workload. One person should not handle PCB design, 3D printing, Arduino code, and Python pipelines simultaneously.
5. **Documentation & Backups:** Keep frequent backups of your code/configs and document every bug and solution—it will save you weeks of debugging later.
6. **Real-World Testing:** Allocate ample time for field testing. Lighting, glare, and vibration in a real vehicle are vastly different from a controlled lab environment.
