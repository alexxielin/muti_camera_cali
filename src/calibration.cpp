#include "calibration.h"
#include <opencv2/calib3d.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>

ImageCalibration::ImageCalibration()
{
}
void ImageCalibration::readParams()
{
  // 从json文件中读取相机参数
  // std::ifstream file("/home/xielin/cali/SuperPoint-SuperGlue-TensorRT-main/config/camera_intrinsics.json");
  std::ifstream file("./config/camera_intrinsics.json");
  nlohmann::json j;
  file >> j;
  // 从JSON对象中获取数据
  // camera1
  double c0_fx = j["col1"]["A"]["fx"];
  double c0_fy = j["col1"]["A"]["fy"];
  double c0_cx = j["col1"]["A"]["cx"];
  double c0_cy = j["col1"]["A"]["cy"];
  double c0_k1 = j["col1"]["A"]["ph_k1"];
  double c0_k2 = j["col1"]["A"]["ph_k2"];
  double c0_p1 = j["col1"]["A"]["ph_p1"];
  double c0_p2 = j["col1"]["A"]["ph_p2"];
  double c0_k3 = j["col1"]["A"]["ph_k3"];
  double c0_k4 = j["col1"]["A"]["ph_k4"];
  double c0_k5 = j["col1"]["A"]["ph_k5"];
  double c0_k6 = j["col1"]["A"]["ph_k6"];
  // camera2
  double c1_fx = j["col1"]["B"]["fx"];
  double c1_fy = j["col1"]["B"]["fy"];
  double c1_cx = j["col1"]["B"]["cx"];
  double c1_cy = j["col1"]["B"]["cy"];
  double c1_k1 = j["col1"]["B"]["ph_k1"];
  double c1_k2 = j["col1"]["B"]["ph_k2"];
  double c1_p1 = j["col1"]["B"]["ph_p1"];
  double c1_p2 = j["col1"]["B"]["ph_p2"];
  double c1_k3 = j["col1"]["B"]["ph_k3"];
  double c1_k4 = j["col1"]["B"]["ph_k4"];
  double c1_k5 = j["col1"]["B"]["ph_k5"];
  double c1_k6 = j["col1"]["B"]["ph_k6"];

  // 生成相机内参矩阵
  K0_ = (cv::Mat_<double>(3, 3) << c0_fx, 0, c0_cx, 0, c0_fy, c0_cy, 0, 0, 1);
  K1_ = (cv::Mat_<double>(3, 3) << c1_fx, 0, c1_cx, 0, c1_fy, c1_cy, 0, 0, 1);
  // 生成畸变系数
  D0_ = (cv::Mat_<double>(1, 8) << c0_k1, c0_k2, c0_p1, c0_p2, c0_k3, c0_k4, c0_k5, c0_k6);
  D1_ = (cv::Mat_<double>(1, 8) << c1_k1, c1_k2, c1_p1, c1_p2, c1_k3, c1_k4, c1_k5, c1_k6);
}

void ImageCalibration::showImage(cv::Mat img, std::vector<cv::KeyPoint> point)
{
  for (auto &p : point)
  {
    cv::circle(img, p.pt, 3, cv::Scalar(0, 255, 0), 1);
  }
  cv::Mat img_resized;
  cv::Size size(2300, 1200); // 你想要的新的图像大小
  cv::resize(img, img_resized, size);
  cv::imshow("Image", img_resized);
  cv::waitKey(0);
}

void ImageCalibration::calibrate(std::vector<cv::KeyPoint> &points0, std::vector<cv::KeyPoint> &points1, std::vector<cv::DMatch> &matchs)
{
  std::vector<cv::Point2f> un_point0, un_point1;
  // //像素点去畸变
  // 将 KeyPoint 转换为 Point2f
  std::vector<cv::Point2f> points0_, points1_;
  points0_.reserve(points0.size());
  points1_.reserve(points1.size());

  // 得到两个图像的特征点（像素坐标）
  for (const auto &kp : points0)
  {
    points0_.emplace_back(kp.pt);
  }
  for (const auto &kp : points1)
  {
    points1_.emplace_back(kp.pt);
  }

  // 得到两个图像匹配上的匹配特征点（像素坐标）
  std::vector<cv::Point2f> match_points0_, match_points1_;
  std::vector<std::pair<int, int>> Idx;
  for (const auto &val : matchs)
  {
    int queryidx = val.queryIdx;
    int trainidx = val.trainIdx;
    match_points0_.emplace_back(points0_[queryidx]);
    match_points1_.emplace_back(points1_[trainidx]);
    Idx.emplace_back(queryidx, trainidx);
  }

  std::vector<uchar> inliers_mask(matchs.size());
  F = cv::findFundamentalMat(match_points0_, match_points1_, inliers_mask, cv::FM_RANSAC);
  E = K1_.t() * F * K0_;
  int count_idx = -1;
  for (auto &val : inliers_mask)
  {
    count_idx++;
    if (val)
    {
      std::pair<int, int> temp = Idx[count_idx];
      ransac_matchs.emplace_back(temp.first, temp.second, 0);
    }
  }

  /* std::vector<cv::Point2f> point2f0, point2f1;
  for (size_t i = 0; i < points0.size(); i++)
  {
    point2f0.push_back(points0[i].pt);
    point2f1.push_back(points1[i].pt);
  }

  // 使用cv::findFundamentalMat函数进行RANSAC
  std::vector<uchar> inliers_mask(match.size());
  cv::findFundamentalMat(point2f0, point2f1, inliers_mask, cv::FM_RANSAC);

  // 保存内点匹配
  std::vector<cv::DMatch> inliers;
  for (size_t i = 0; i < inliers_mask.size(); ++i)
  {
    if (inliers_mask[i])
    {
      inliers.push_back(match[i]);
    }
  }

  // 创建一个用于显示匹配结果的图像
  cv::Mat img_matches0, img_matches_resized0;
  cv::drawMatches(img0_, points0, img1_, points1, inliers, img_matches0);

  // 调整图像的大小
  cv::Size size0(2560, 1300); // 你想要的新的图像大小
  cv::resize(img_matches0, img_matches_resized0, size0);

  // 显示调整大小后的匹配结果
  cv::imshow("Matches", img_matches_resized0);
  cv::waitKey(0); */
}
