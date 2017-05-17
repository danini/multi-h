#pragma once

#define _USE_MATH_DEFINES
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
 
class ASiftDetector
{
public:
  ASiftDetector();
 
  void detectAndCompute(const cv::Mat& img, std::vector< cv::KeyPoint >& keypoints, cv::Mat& descriptors);

protected:
  struct ParallelOp : public cv::ParallelLoopBody {
    ParallelOp(const cv::Mat& _img, std::vector<std::vector<cv::KeyPoint>> &kps, std::vector<cv::Mat> &dsps);
    void affineSkew(double tilt, double phi, cv::Mat& img, cv::Mat& mask, cv::Mat& Ai) const;
    void operator()( const cv::Range &r ) const;

    cv::Mat img;
    std::vector<cv::KeyPoint>* keypoints_array;
    cv::Mat* descriptors_array;
  };
  
};