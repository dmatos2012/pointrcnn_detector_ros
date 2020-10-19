catkin init
catkin config -DPYTHON_EXECUTABLE=$1 -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so
catkin config --install
git clone https://github.com/ros-perception/vision_opencv.git src/vision_opencv
cd src/vision_opencv/
git checkout 1.13.0
cd ../../
catkin build cv_bridge
source install/setup.bash --extend
rm -r build
rm -r devel
catkin_make -DPYTHON_EXECUTABLE=$1 -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so
