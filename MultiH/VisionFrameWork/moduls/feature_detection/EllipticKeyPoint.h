#pragma once

#include <cv.h>

class EllipticKeyPoint : public cv::KeyPoint {
public:
    EllipticKeyPoint();
	EllipticKeyPoint(const EllipticKeyPoint& kp);
	EllipticKeyPoint(const cv::KeyPoint& kp, const cv::Mat_<double> Ai);
	virtual ~EllipticKeyPoint();

    static void convert( const std::vector<cv::KeyPoint>& src, std::vector<EllipticKeyPoint>& dst );
    static void convert( const std::vector<EllipticKeyPoint>& src, std::vector<cv::KeyPoint>& dst );

	cv::Point2d applyAffineHomography(const cv::Mat_<double>& H, const cv::Point2d& pt);
	
    cv::Mat_<double> transformation;
	cv::Mat_<double> ownAffinity;
};
