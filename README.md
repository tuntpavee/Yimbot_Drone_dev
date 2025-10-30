# Yimbot Drone Dev

This repository contains the ROS 2 (Humble) vision processing workspaces for the Yimbot Drone project. It is designed to run in a split-machine setup:

1.  **Publisher (Raspberry Pi 4):** A separate RPi (running `Rpi_Ard_ser` workspace) captures video from a webcam and publishes it to ROS 2 topics.
2.  **Subscriber (VM/PC):** This workspace (`yolo_dt`) subscribes to the video stream and performs advanced vision tasks like object tracking (YOLOv8) or depth estimation (MiDaS).

---

## System Overview

* **Machine A (Publisher):** Raspberry Pi 4
    * **OS:** Ubuntu 22.04 (Server or Desktop)
    * **ROS:** ROS 2 Humble
    * **Workspace:** `~/Rpi_Ard_ser`
* **Machine B (Subscriber/Processor):** PC/Mac with Ubuntu 22.04 VM
    * **OS:** Ubuntu 22.04 (in Parallels VM)
    * **ROS:** ROS 2 Humble
    * **Workspace:** `~/yolo_dt` (this repository)
* **Network:** Both machines must be on the same network and must set `export ROS_DOMAIN_ID=30` to communicate.

---

## Required Libraries & Dependencies

### 1. Machine A (Raspberry Pi Publisher)

This setup assumes you are using the `Rpi_Ard_ser` workspace, which is **not** included in this repo.

* **System Packages:**
    ```bash
    sudo apt update
    sudo apt install ros-humble-desktop v4l-utils python3-pip git
    ```
* **ROS 2 Packages:**
    ```bash
    sudo apt install ros-humble-cv-bridge
    ```
* **User Permissions:**
    ```bash
    sudo usermod -aG video $USER
    # You MUST reboot after this
    ```

### 2. Machine B (Ubuntu VM Processor)

This is for the `yolo_dt` workspace (this repository).

* **System Packages:**
    ```bash
    sudo apt update
    sudo apt install ros-humble-desktop python3-pip git
    ```
* **ROS 2 Packages:**
    ```bash
    # For viewing compressed images in RQT
    sudo apt install ros-humble-image-transport-plugins
    # For cv_bridge and system-level OpenCV
    sudo apt install ros-humble-cv-bridge python3-opencv 
    ```
* **Python (pip) Packages:**
    ```bash
    # IMPORTANT: Fix NumPy version conflicts for ROS 2 Humble
    pip install "numpy<2" --force-reinstall
    
    # Install vision libraries
    pip install ultralytics
    pip install torch torchvision timm
    ```
* **Hardware (for RealSense):**
    * A **powered** USB 3.0 hub is required for stable RealSense operation.

---

## Installation (Machine B - Ubuntu VM)

1.  **Clone this Repository:**
    ```bash
    cd ~
    git clone [https://github.com/tuntpavee/Yimbot_Drone_dev.git](https://github.com/tuntpavee/Yimbot_Drone_dev.git) yolo_dt
    cd yolo_dt
    ```

2.  **Initialize Submodules (for `px4_msgs`):**
    ```bash
    git submodule update --init --recursive
    ```

3.  **Install Dependencies:**
    ```bash
    # Source ROS 2 to get the rosdep command
    source /opt/ros/humble/setup.bash
    
    # Install dependencies, skipping pip-installed packages
    rosdep install -i --from-path src --rosdistro humble -y --skip-keys "opencv-python ultralytics"
    ```

4.  **Build the Workspace:**
    ```bash
    colcon build
    ```

---

## How to Run

### Step 1: Run Publisher on Pi (Machine A)

1.  SSH into your Raspberry Pi.
2.  Source your ROS 2 environment and workspace:
    ```bash
    source /opt/ros/humble/setup.bash
    source ~/Rpi_Ard_ser/install/setup.bash
    ```
3.  Set the Domain ID:
    ```bash
    export ROS_DOMAIN_ID=30
    ```
4.  Run your camera publisher (using the name from your `setup.py`):
    ```bash
    # This publishes the uncompressed /image_raw and /camera_info
    ros2 run my_mav_pkg camera_publisher 
    ```
    *Note: If you are using the compressed version, run that script instead.*

### Step 2: Run Processor on VM (Machine B)

1.  Open a terminal in your Ubuntu VM.
2.  Source your ROS 2 environment and this workspace:
    ```bash
    source /opt/ros/humble/setup.bash
    source ~/yolo_dt/install/setup.bash
    ```
3.  Set the Domain ID:
    ```bash
    export ROS_DOMAIN_ID=30
    ```
4.  **Run ONE** of the following modules:

    ---
    #### To Run YOLOv8 Interactive Tracking:
    A window will pop up. Drag a box to select an object, and the tracker will follow it.
    ```bash
    ros2 run yolo_subscriber yolo_node
    ```
    * **Note:** This node subscribes to `/image_raw` (uncompressed).

    ---
    #### To Run MiDaS Depth Estimation:
    A window will pop up showing the estimated depth map.
    ```bash
    ros2 run midas_estimator midas_node
    ```
    * **Note:** This node also subscribes to `/image_raw` (uncompressed).

    ---
    #### To Run Foxglove Bridge (for Visualization):
    ```bash
    ros2 run foxglove_bridge foxglove_bridge
    ```
    * Then connect Foxglove Studio on your Mac to `ws://<VM_IP_ADDRESS>:8765`.
