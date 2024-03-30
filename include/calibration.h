#ifndef CALI_H_
#define CALI_H_
#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "Eigen/Core"
#include <string>
#include <nlohmann/json.hpp>
#include <unordered_map>
class ImageCalibration
{
public:
  ImageCalibration();
  ~ImageCalibration() = default;
  void readParams();
  void calibrate(std::vector<cv::KeyPoint> &points0, std::vector<cv::KeyPoint> &points1, std::vector<cv::DMatch> &matchs);
  void undistorted(cv::Mat input_img, cv::Mat &output_img, cv::Mat K, cv::Mat D);
  cv::Mat K0_, K1_;
  cv::Mat D0_, D1_;
  std::vector<cv::DMatch> ransac_matchs;
  cv::Mat F, E;
};
#endif
