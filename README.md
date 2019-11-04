# Udacity-3D-Camera-Lidar-Object-Tracking
Object tracking algorithms using camera and lidar sensor data based on the Udacity Nanodegree Program "Become a Sensor Fusion Engineer"

Within this protect object tracking algorithms based on camera sensor data are implemented. 
Keypoint detectors, descriptors, and methods to match them between successive images are considered.
How to detect objects in an image using the YOLO deep-learning framework is considered.
Finally, you will know how to associate regions in a camera image with Lidar points in 3D space.

<img src="images/course_code_structure.png" width="779" height="414" />
<img src="example_pic.png" width="290" height="290" />

In this final project, the missing parts in the schematic have been implemented. It contains the four major tasks: 
1. First, a way to match 3D objects over time by using keypoint correspondences is developed. 
2. Second, computing the TTC based on Lidar measurements is considered. 
3. It will be proceeded to do the same using the camera, which requires to first associate keypoint matches to regions of interest and then to compute the TTC based on those matches. 
4. And lastly, various tests with the framework will be conducted. The goal is to identify the most suitable detector/descriptor combination for TTC estimation and also to search for problems that can lead to faulty measurements by the camera or Lidar sensor.
 

## Dependencies for Running Locally
* cmake >= 2.8
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* Git LFS
  * Weight files are handled using [LFS](https://git-lfs.github.com/)
* OpenCV >= 4.1
  * This must be compiled from source using the `-D OPENCV_ENABLE_NONFREE=ON` cmake flag for testing the SIFT and SURF detectors.
  * The OpenCV 4.1.0 source code can be found [here](https://github.com/opencv/opencv/tree/4.1.0)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory in the top level project directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./3D_object_tracking`.
