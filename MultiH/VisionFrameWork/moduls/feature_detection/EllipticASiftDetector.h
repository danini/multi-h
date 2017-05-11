#pragma once

#include "EllipticKeyPoint.h"

class EllipticASiftDetector
{
public:
  EllipticASiftDetector();
 
  void detectAndCompute(const cv::Mat& img, std::vector< EllipticKeyPoint >& keypoints, cv::Mat& descriptors, std::string method = "SIFT");

protected:
  struct ParallelOp : public cv::ParallelLoopBody {
    ParallelOp(const cv::Mat& _img, std::vector<std::vector<EllipticKeyPoint>> &kps, std::vector<cv::Mat> &dsps);
    void affineSkew(double tilt, double phi, cv::Mat& img, cv::Mat& mask, cv::Mat& Ai, cv::Mat& A) const;
	void operator()(const cv::Range &r) const;
	void doTracking(cv::Mat const &img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors, std::string method) const;
	void doHessian(cv::Mat const &img, std::vector<cv::KeyPoint>& keypoints, std::vector<cv::Mat>& affines, cv::Mat& descriptors, std::string method) const;

	cv::Mat gray_image;
    cv::Mat img;
    std::vector<EllipticKeyPoint>* keypoints_array;
	cv::Mat* descriptors_array;
	std::string method;
  };

  std::string _method;
};
