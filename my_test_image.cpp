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
    std::string config_path = "/home/xielin/xielin/whole_ws/point_glue_ws/SuperPoint-SuperGlue-TensorRT-main/config/config.yaml";
    std::string model_dir = "/home/xielin/xielin/whole_ws/point_glue_ws/SuperPoint-SuperGlue-TensorRT-main/weights/";
    std::string image0_path = "/home/xielin/xielin/whole_ws/point_glue_ws/SuperPoint-SuperGlue-TensorRT-main/image/image0.png";
    std::string image1_path = "/home/xielin/xielin/whole_ws/point_glue_ws/SuperPoint-SuperGlue-TensorRT-main/image/image1.png";

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
    if (!superpoint->build())
    {
        std::cerr << "Error in SuperPoint building engine. Please check your onnx model path." << std::endl;
        return 0;
    }
    auto superglue = std::make_shared<SuperGlue>(configs.superglue_config);
    if (!superglue->build())
    {
        std::cerr << "Error in SuperGlue building engine. Please check your onnx model path." << std::endl;
        return 0;
    }
    std::cout << "SuperPoint and SuperGlue inference engine build success." << std::endl;

    Eigen::Matrix<double, 259, Eigen::Dynamic> feature_points0, feature_points1;
    std::cout << "9999999  " << feature_points0.cols() << std::endl;
    std::vector<cv::DMatch> superglue_matches;

    std::cout << "---------------------------------------------------------" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    if (!superpoint->infer(image0, feature_points0))
    {
        std::cerr << "Failed when extracting features from first image." << std::endl;
        return 0;
    }
    std::cout << "First image feature_pts scores: " << std::endl; // 根据这个输出结果，说明superpoint的输出并没有按照得分排序，而是按照像素坐标系排序，以v坐标
    for (size_t i = 0; i < 10; i++)
    {
        std::cout << "得分为" << feature_points0(0, i) << " 像素坐标为" << feature_points0(1, i) << " " << feature_points0(2, i) << "\n";
    }
    std::cout << "\n";
    std::cout << "First image feature points number: " << feature_points0.cols() << std::endl;
    if (!superpoint->infer(image1, feature_points1))
    {
        std::cerr << "Failed when extracting features from second image." << std::endl;
        return 0;
    }
    std::cout << "Second image feature_pts scores: " << std::endl;
    for (size_t i = 0; i < 10; i++)
    {
        std::cout << "得分为" << feature_points1(0, i) << " 像素坐标为" << feature_points1(1, i) << " " << feature_points1(2, i) << " ";
    }
    std::cout << "\n";
    std::cout << "Second image feature points number: " << feature_points1.cols() << std::endl;

    superglue->matching_points(feature_points0, feature_points1, superglue_matches); // 可以观察到superglue的输出是两个特征点vector的匹配上的索引
    std::cout << "match feature pair: " << superglue_matches.size() << std::endl;
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
    for (size_t i = 0; i < superglue_matches.size(); i++)
    {
        cv::DMatch match = superglue_matches[i];
        std::cout << "Query idx: " << match.queryIdx << ", Train idx: " << match.trainIdx
                  << ", Distance: " << match.distance << std::endl;
    }
    cv::drawMatches(image0, keypoints0, image1, keypoints1, superglue_matches, match_image);
    cv::imwrite("match_image.png", match_image);
    feature_points0.setZero();
    std::cout << "9999999  " << feature_points0.cols() << std::endl;
    superpoint->infer(image1, feature_points0);
    std::cout << feature_points0.cols() << std::endl;
    // visualize
        cv::imshow("match_image", match_image);
    cv::waitKey(-1);

    return 0;
}
