#include "calibration.h"
#include <opencv2/calib3d.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <nlohmann/json.hpp>

ImageCalibration::ImageCalibration(std::string img0_dir, std::string img1_dir){
  readParams();
  img0_ = readImage(img0_dir, K0_, D0_);
  img1_ = readImage(img1_dir, K1_, D1_);
}
void ImageCalibration::readParams(){
  // 从json文件中读取相机参数
  std::ifstream file("../config/camera_intrinsics.json");
  nlohmann::json j;
  file >> j;
  // 从JSON对象中获取数据
  //camera1
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
  //camera2
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

cv::Mat ImageCalibration::readImage(std::string img_dir, cv::Mat K, cv::Mat D){
  cv::Mat img;
  img = imread(img_dir, cv::IMREAD_COLOR);
  if(img.empty()){
      std::cout << "Could not read the image: " << img_dir << std::endl;
  }
  else{
      std::cout << "Image read successfully" << std::endl;
  }
  cv::Mat map1, map2, img_undistorted;
  cv::initUndistortRectifyMap(K, D, cv::Mat(), K, img.size(), CV_32FC1, map1, map2);
  cv::remap(img, img_undistorted, map1, map2, cv::INTER_LINEAR);
  return img_undistorted;
}

void ImageCalibration::showImage(cv::Mat img, std::vector<cv::KeyPoint> point){
  for(auto& p: point){
    cv::circle(img, p.pt, 3, cv::Scalar(0, 255, 0), 1);
  }
  cv::Mat img_resized;
  cv::Size size(2300, 1200);  // 你想要的新的图像大小
  cv::resize(img, img_resized, size);
  cv::imshow("Image", img_resized);
  cv::waitKey(0);
}

void ImageCalibration::calibrate(){
  // Do calibration
  std::cout << "Calibrating..." << std::endl;
  //提取sift特征点
  std::vector<cv::KeyPoint> points0, points1;
  cv::Mat descriptors1, descriptors2;

  cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create(800, 1.2f, 16);
  cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();
  cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");


  //-- 第一步:检测 Oriented FAST 角点位置
  detector->detect(img0_, points0);
  detector->detect(img1_, points1);
  std::cout << "Number of keypoints (img0): " << points0.size() << '\n';
  std::cout << "Number of keypoints (img0): " << points1.size() << '\n';
  showImage(img0_, points0);
  showImage(img1_, points1);
    //-- 第二步:根据角点位置计算 BRIEF 描述子
  descriptor->compute(img0_, points0, descriptors1);
  descriptor->compute(img1_, points1, descriptors2);

  //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
  std::vector<cv::DMatch> match;
  matcher->match(descriptors1, descriptors2, match);

    //-- 第四步:匹配点对筛选
  double min_dist = 10000, max_dist = 0;

  //找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
  for (int i = 0; i < descriptors1.rows; i++) {
    double dist = match[i].distance;
    if (dist < min_dist) min_dist = dist;
    if (dist > max_dist) max_dist = dist;
  }
  printf("-- Max dist : %f \n", max_dist);
  printf("-- Min dist : %f \n", min_dist);

  //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
  for (int i = 0; i < descriptors1.rows; i++) {
    if (match[i].distance <= std::max(2 * min_dist, 30.0)) {
      match.push_back(match[i]);
    }
  }

  cv::Mat img_matches, img_matches_resized;
  cv::drawMatches(img0_, points0, img1_, points1, match, img_matches);


  cv::Size size(2300, 1200);  // 你想要的新的图像大小
  cv::resize(img_matches, img_matches_resized, size);

  // 显示匹配结果
  cv::imshow("Matches", img_matches_resized);
  cv::waitKey(0);

  // std::vector<cv::KeyPoint> un_point0, un_point1;
  // //像素点去畸变
  // cv::undistortPoints(points0, un_point0, K0_, D0_);
  // cv::undistortPoints(points1, un_point1, K1_, D1_);

  std::vector<cv::Point2f> point2f0, point2f1;
  for (size_t i = 0; i < points0.size(); i++) {
    point2f0.push_back(points0[i].pt);
    point2f1.push_back(points1[i].pt);
  }

  // 使用cv::findFundamentalMat函数进行RANSAC
  std::vector<uchar> inliers_mask(match.size());
  cv::findFundamentalMat(point2f0, point2f1, inliers_mask, cv::FM_RANSAC);

    // 保存内点匹配
  std::vector<cv::DMatch> inliers;
  for (size_t i = 0; i < inliers_mask.size(); ++i) {
    if (inliers_mask[i]) {
      inliers.push_back(match[i]);
    }
  }

    // 创建一个用于显示匹配结果的图像
  cv::Mat img_matches0, img_matches_resized0;
  cv::drawMatches(img0_, points0, img1_, points1, inliers, img_matches0);

  // 调整图像的大小
  cv::Size size0(2560, 1300);  // 你想要的新的图像大小
  cv::resize(img_matches0, img_matches_resized0, size0);

  // 显示调整大小后的匹配结果
  cv::imshow("Matches", img_matches_resized0);
  cv::waitKey(0);
}

/* int main(int argc, char** argv) {
  if(argc != 3){
    std::cout << "Usage: ./calibration <image0_dir> <image0_dir>" << std::endl;
    return -1;
  }
  // 读取图像
  std::string img0_dir = argv[1];
  std::string img1_dir = argv[2];
  ImageCalibration img_calib(img0_dir, img1_dir);
  img_calib.calibrate();
} */
