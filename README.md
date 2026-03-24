# raspberrypi-360-camera
Driver assistance system using ultrasound sensors and computer vision
First, you need to read a lot of documentation about ADAS, 360-degree camera systems, and how to use Python through YouTube or other resources. Spend the first few weeks researching and preparing the necessary equipment to build a 360-degree camera system for a car along with a front and rear collision warning system.

This project will cost around 3–4 million VND, including a Raspberry Pi 5 (8GB version) and four fisheye cameras. Note that the cameras are the hardest part to purchase, as USB fisheye cameras are quite rare on the market—most available options are standard fisheye modules. You can try finding them on Amazon or Shopee within Southeast Asia.

At the beginning, you must carefully read the processing files that I have sent you earlier. You need to be patient and work step by step, recording all parameters to ensure accurate undistortion that matches your specific vehicle. Each vehicle has different dimensions (length, width, height), so you cannot reuse the same K and D parameters across different setups.

I encountered many issues when coding across macOS, Windows, and Raspberry Pi OS (Linux on Raspberry Pi 5). You should use multiprocessing and multithreading to reduce latency and achieve the highest possible FPS for the system. Even though the system only runs at around 12–20 FPS, it demonstrates that we have optimized both the code and hardware as much as possible.

You may consider using a car display with an integrated CPU or a Jetson Nano to achieve better performance. It is very important to record almost all necessary parameters during development. If you are working in a team, you should divide the tasks, since this project involves many components such as PCB design, hardware assembly, 3D printing, Arduino programming, Python programming, and more.

In addition, you should:

Plan a clear system architecture from the beginning (camera placement, wiring, data flow).
Test each module independently before integrating the whole system.
Calibrate cameras carefully to ensure seamless stitching and accurate distance estimation.
Optimize image processing pipelines to reduce computational load.
Keep backups of your code and configurations regularly.
Document every issue you encounter and how you solve it, as this will save a lot of time later.
Finally, allocate enough time for testing in real-world conditions, since performance can vary significantly outside of controlled environments.
