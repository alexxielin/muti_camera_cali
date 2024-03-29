#ifndef CALI_H_
#define CALI_H_
#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "Eigen/Core"
#include <string>
#include <nlohmann/json.hpp>
class ImageCalibration
{
public:
  ImageCalibration();
  ~ImageCalibration() = default;
  void readParams();
  void showImage(cv::Mat img, std::vector<cv::KeyPoint> point);
  void calibrate();

  cv::Mat K0_, K1_;
  cv::Mat D0_, D1_;
};
#endif
