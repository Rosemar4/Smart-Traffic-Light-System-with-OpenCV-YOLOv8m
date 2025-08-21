# Smart-Traffic-Light-System-with-OpenCV-YOLOv8n
This project uses OpenCV and YOLOv8m to build an AI-powered smart traffic light system. It detects vehicles in multiple lanes and assigns the green light to the lane with the highest vehicle count, switching every 60 seconds.
✨ Features

🚗 Real-time vehicle detection using YOLOv8m

🔄 Dynamic traffic light control based on vehicle density

⏱️ Automatic switching after 60 seconds

📷 Works with video feed or live camera

📂 Project Structure

src/detect.py → Vehicle detection and counting

src/traffic_light.py → Traffic light control logic

src/main.py → Main script to run the system

⚙️ Installation
git clone https://github.com/yourusername/smart-traffic-light-opencv-yolov8m.git
cd smart-traffic-light-opencv-yolov8m

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows

# Install dependencies
pip install -r requirements.txt


requirements.txt should contain:

ultralytics
opencv-python
numpy

🚀 Usage

Run the system with:

python src/main.py --video data/test.mp4


For live camera:

python src/main.py --camera 0

🛠️ How It Works

YOLOv8m detects vehicles in each lane.

Counts are compared for all lanes.

Lane with highest count gets the green light.

After 60 seconds, system re-checks and switches.
