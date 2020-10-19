Create catkin workspace,

`mkdir -p ~/catkin_ws/src`

`cd ~/catkin_ws/src`

Clone repo in the src folder.

`git clone --recursive https://github.com/dmatos2012/pointrcnn_detector_ros.git`

# Prepare environment
First, create a virtual environment, as such:

`python3 -m venv path/to/virtual/env/name`

`source path/to/virtual/env/name/bin/activate`

# Requirements
Once you have the venv setup, install python 3 requirements. In the root folder then,

`pip install -r requirements.txt`

Additionally, you need to build extra PointRCNN libraries, specifically point C from their repo.

`cd src/PointRCNN/`

`sh build_and_install.sh` 


Then run the script, which would build it cv bridge to be compatible with python3. 

`cd ~/catkin_ws`

`sh src/pointrcnn_detector_ros/configure_environment.sh path/to/virtual/env/name/bin/python`

`source devel/setup.bash`

Finally launch it:

Remember to add pre-trained model in tools/.. and the kittibag in data/
`roslaunch prcnn_ros kitti_car_det.launch`



