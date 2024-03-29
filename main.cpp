//
// Created by haoyuefan on 2021/9/22.
//

#include <memory>
#include <chrono>
#include "utils.h"
#include "super_glue.h"
#include "super_point.h"

int main(int argc, char **argv)
{
    std::string project_path = "/home/xielin/cali/SuperPoint-SuperGlue-TensorRT-main/";
    std::string config_path = project_path + "config/config.yaml";
    std::string model_dir = project_path + "weights/";
    std::string image0_path = project_path + "image/0.jpg";
    std::string image1_path = project_path + "image/1.jpg";

    cv::Mat image0 = cv::imread(image0_path, cv::IMREAD_GRAYSCALE);
    cv::Mat image1 = cv::imread(image1_path, cv::IMREAD_GRAYSCALE);

    if (image0.empty() || image1.empty())
    {
        std::cerr << "Input image is empty. Please check the image path." << std::endl;
        return 0;
    }

    Configs configs(config_path, model_dir);
    int width = configs.superglue_config.image_width;
    int height = configs.superglue_config.image_height;

    cv::resize(image0, image0, cv::Size(width, height));
    cv::resize(image1, image1, cv::Size(width, height));
    // cv::imshow("image0", image0);
    // cv::waitKey(0);
    std::cout << "First image size: " << image0.cols << "x" << image0.rows << std::endl;
    std::cout << "Second image size: " << image1.cols << "x" << image1.rows << std::endl;

    std::cout << "Building inference engine......" << std::endl;
    auto superpoint = std::make_shared<SuperPoint>(configs.superpoint_config);
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
    /* for (size_t i = 0; i < superglue_matches.size(); i++)
    {
        cv::DMatch match = superglue_matches[i];
        std::cout << "Query idx: " << match.queryIdx << ", Train idx: " << match.trainIdx
                  << ", Distance: " << match.distance << std::endl;
    } */
    cv::drawMatches(image0, keypoints0, image1, keypoints1, superglue_matches, match_image);
    // visualize
    cv::imshow("match_image", match_image);
    cv::waitKey(-1);

    // 去除外点
    // 像素点去畸变
    std::vector<cv::KeyPoint> un_point0, un_point1;
    cv::undistortPoints(keypoints0, un_point0, K0_, D0_);
    cv::undistortPoints(keypoints1, un_point1, K1_, D1_);

    return 0;
}
