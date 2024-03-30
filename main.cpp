//
// Created by haoyuefan on 2021/9/22.
//
#include <memory>
#include <chrono>
#include "utils.h"
#include "super_glue.h"
#include "super_point.h"
#include "calibration.h"
int main(int argc, char **argv)
{
    std::string project_path = "/home/xielin/cali/SuperPoint-SuperGlue-TensorRT-main/";
    std::string config_path = project_path + "config/config.yaml";
    std::string model_dir = project_path + "weights/";
    std::string image0_path = project_path + "image/0.jpg";
    std::string image1_path = project_path + "image/1.jpg";
    ImageCalibration calier;
    calier.readParams();

    cv::Mat image0_ = cv::imread(image0_path, cv::IMREAD_GRAYSCALE);
    cv::Mat image1_ = cv::imread(image1_path, cv::IMREAD_GRAYSCALE);
    cv::Mat image0, image1;
    // 对整个图像去畸变(这种去畸变方式会保留更多的像素点但是会丢掉边缘信息)
    // cv::undistort(image0_, image0, calier.K0_, calier.D0_);
    // cv::undistort(image1_, image1, calier.K1_, calier.D1_);

    // 对整个图像去畸变(这种去畸变方式会少掉一部分的像素点但是会保留边缘信息)
    calier.undistorted(image0_, image0, calier.K0_, calier.D0_);
    calier.undistorted(image1_, image1, calier.K1_, calier.D1_);

    Configs configs(config_path, model_dir);
    int width = configs.superglue_config.image_width;
    int height = configs.superglue_config.image_height;

    // cv::imshow("image0", image0);
    // cv::waitKey(0);

    std::cout << "Building inference engine......" << std::endl;
    auto superpoint = std::make_shared<SuperPoint>(configs.superpoint_config);
    // 现在superpoint的序列化引擎支持2000*2000的图像输入
    superpoint->build();

    auto superglue = std::make_shared<SuperGlue>(configs.superglue_config);
    superglue->build();

    Eigen::Matrix<double, 259, Eigen::Dynamic> feature_points0, feature_points1;
    std::vector<cv::DMatch> superglue_matches;

    superpoint->infer(image0, feature_points0);
    superpoint->infer(image1, feature_points1);

    superglue->matching_points(feature_points0, feature_points1, superglue_matches); // 可以观察到superglue的输出是两个特征点vector的匹配上的索引
    cv::Mat match_image;
    std::vector<cv::KeyPoint> keypoints0, keypoints1;
    for (size_t i = 0; i < feature_points0.cols(); ++i)
    {
        double score = feature_points0(0, i);
        double x = feature_points0(1, i);
        double y = feature_points0(2, i);
        keypoints0.emplace_back(x, y, 8, -1, score);
    }
    for (size_t i = 0; i < feature_points1.cols(); ++i)
    {
        double score = feature_points1(0, i);
        double x = feature_points1(1, i);
        double y = feature_points1(2, i);
        keypoints1.emplace_back(x, y, 8, -1, score);
    }
    cv::drawMatches(image0, keypoints0, image1, keypoints1, superglue_matches, match_image);
    // visualize
    cv::imwrite("./result/match_image_without_ransac.jpg", match_image);
    // cv::waitKey(-1);

    // 去除外点
    // 像素点去畸变
    // 通过计算去畸变后的F矩阵来剔除外点
    calier.calibrate(keypoints0, keypoints1, superglue_matches);

    cv::Mat match_image_ransac;
    cv::drawMatches(image0, keypoints0, image1, keypoints1, calier.ransac_matchs, match_image_ransac);
    cv::imwrite("./result/match_image_with_ransac.jpg", match_image_ransac);
    std::cout << calier.E << "\n";
    // 进行奇异值分解（SVD）
    cv::Mat R, t;
    cv::SVD svd(calier.E, cv::SVD::FULL_UV);

    // 恢复旋转矩阵
    R = svd.u * cv::Mat::eye(3, 3, CV_64F) * svd.vt;

    // 恢复平移向量
    t = svd.u.col(2);

    // 输出旋转矩阵和平移向量
    std::cout << "Rotation Matrix:\n"
              << R << std::endl;
    std::cout << "Translation Vector:\n"
              << t << std::endl;
    return 0;
}
