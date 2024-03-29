#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "Eigen/Core"
#include <string>

struct CameraConfig{
  std::string img_dir;
  cv::Mat K, D;
  cv::Mat img;
  double width, height;
  double fx,fy,cx,cy;
  double k1,k2,p1,p2,k3,k4,k5,k6;
};
class ImageCalibration{
public:
    ImageCalibration(std::string img0_dir, std::string img1_dir);
    ~ImageCalibration() =  default;
    cv::Mat readImage(std::string img_dir, cv::Mat K, cv::Mat D);
    void readParams();
    void showImage(cv::Mat img, std::vector<cv::KeyPoint> point);
    void calibrate();
private:
  cv::Mat img0_, img1_;
  cv::Mat K0_, K1_;
  cv::Mat D0_, D1_;
};
