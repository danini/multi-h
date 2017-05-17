#include "ASiftDetector.h"

#include <iostream>
#include <algorithm>
#include <opencv2\xfeatures2d\nonfree.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#define DEBUG_WINDOW 0
#define USE_SURF 0

ASiftDetector::ASiftDetector()
{
}

ASiftDetector::ParallelOp::ParallelOp(const cv::Mat& _img, std::vector<std::vector<cv::KeyPoint>> &kps, std::vector<cv::Mat> &dsps) {
	img = _img;
	keypoints_array = &kps[0];
	descriptors_array = &dsps[0];
}

void ASiftDetector::ParallelOp::affineSkew(double tilt, double phi, cv::Mat& img, cv::Mat& mask, cv::Mat& Ai) const 
{
	int h = img.rows;
	int w = img.cols;

	mask = cv::Mat(h, w, CV_8UC1, cv::Scalar(255)); 

	cv::Mat A = cv::Mat::eye(2,3, CV_32F);

	if(phi != 0.0)
	{
		phi *= M_PI/180.;
		double s = sin(phi);
		double c = cos(phi);

		A = (cv::Mat_<float>(2,2) << c, -s, s, c);

		cv::Mat corners = (cv::Mat_<float>(4,2) << 0, 0, w, 0, w, h, 0, h);
		cv::Mat tcorners = corners*A.t();
		cv::Mat tcorners_x, tcorners_y;
		tcorners.col(0).copyTo(tcorners_x);
		tcorners.col(1).copyTo(tcorners_y);
		std::vector<cv::Mat> channels;
		channels.push_back(tcorners_x);
		channels.push_back(tcorners_y);
		merge(channels, tcorners);

		cv::Rect rect = cv::boundingRect(tcorners);
		A =  (cv::Mat_<float>(2,3) << c, -s, -rect.x, s, c, -rect.y);

		warpAffine(img, img, A, cv::Size(rect.width, rect.height), cv::INTER_LINEAR, cv::BORDER_REPLICATE);
	}
	if(tilt != 1.0)
	{
		double s = 0.8*sqrt(tilt*tilt-1);
		GaussianBlur(img, img, cv::Size(0,0), s, 0.01);
		resize(img, img, cv::Size(0,0), 1.0/tilt, 1.0, cv::INTER_NEAREST);
		A.row(0) = A.row(0)/tilt;
	}
	if(tilt != 1.0 || phi != 0.0)
	{
		h = img.rows;
		w = img.cols;
		warpAffine(mask, mask, A, cv::Size(w, h), cv::INTER_NEAREST);
	}
	invertAffineTransform(A, Ai);
}

void ASiftDetector::ParallelOp::operator()( const cv::Range &r ) const {
	
	for (register int tl = r.start; tl != r.end; ++tl) {

		std::vector<cv::KeyPoint>& keypoints0 = keypoints_array[tl-1];
		cv::Mat& descriptors0 = descriptors_array[tl-1];
		double t = pow(2, 0.5*tl);

		for(double /*TODO: used to be int, i changed dis(Ivan)*/ phi = 0; phi < 180; phi += 72.0/t)
		{
			std::vector<cv::KeyPoint> kps;
			cv::Mat desc;

			cv::Mat timg, mask, Ai;
			img.copyTo(timg);

			affineSkew(t, phi, timg, mask, Ai);

	#if DEBUG_WINDOW
			cv::Mat img_disp;
			bitwise_and(mask, timg, img_disp);
			namedWindow( "Skew", WINDOW_AUTOSIZE );// Create a window for display.
			imshow( "Skew", img_disp );
			waitKey(0);
	#endif

			{
				cv::Ptr<cv::Feature2D> f2d = cv::xfeatures2d::SIFT::create();
				
				f2d->detect(timg, kps, mask);
				f2d->compute(timg, kps, desc);

				for(unsigned int i = 0; i < kps.size(); i++)
				{
					cv::Point3f kpt(kps[i].pt.x, kps[i].pt.y, 1);
					cv::Mat kpt_t = Ai*cv::Mat(kpt);
					kps[i].pt = *kpt_t.ptr<cv::Point2f>();
				}
				keypoints0.insert(keypoints0.end(), kps.begin(), kps.end());
				descriptors0.push_back(desc);
			}
#if USE_SURF
			{
				SurfFeatureDetector detector(350, 16);
				SurfDescriptorExtractor extractor;

				detector.detect(timg, kps, mask);
				extractor.compute(timg, kps, desc);

				for(unsigned int i = 0; i < kps.size(); i++)
				{
					Point3f kpt(kps[i].pt.x, kps[i].pt.y, 1);
					cv::Mat kpt_t = Ai*cv::Mat(kpt);
					kps[i].pt = *kpt_t.ptr<Point2f>();
				}
				keypoints0.insert(keypoints0.end(), kps.begin(), kps.end());

				cv::Mat cols(desc.rows,desc.cols, desc.type(), cvcv::Scalar(0.));
				cv::hconcat(desc, cols, desc);
				descriptors0.push_back(desc);
			}
#endif
		}
	}
}

void ASiftDetector::detectAndCompute(const cv::Mat& img, std::vector< cv::KeyPoint >& keypoints, cv::Mat& descriptors)
{
	auto keypoints_array = std::vector<std::vector<cv::KeyPoint>>(5);
	auto descriptors_array = std::vector<cv::Mat>(5);

	auto sum = ASiftDetector::ParallelOp(img, keypoints_array, descriptors_array);
	parallel_for_(cv::Range(1, 6), sum); // non-inclusive end: 6 (elements: 1,2,3,4,5)
	
	// Merge!
	keypoints.clear();
	descriptors = cv::Mat(0, 128, CV_32F);

	for(int tl = 0; tl < 5; tl++) {
		keypoints.insert(keypoints.end(), keypoints_array[tl].begin(), keypoints_array[tl].end());
		descriptors.push_back(descriptors_array[tl]);
	}
}