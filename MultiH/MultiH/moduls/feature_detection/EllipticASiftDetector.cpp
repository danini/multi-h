#define _USE_MATH_DEFINES

#include "EllipticASiftDetector.h"
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2\xfeatures2d\nonfree.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cv.h>

#include <cmath>

#include "pyramid.h"
#include "helpers.h"
#include "affine.h"
#include "siftdesc.h"
#include "HessianAffineDetector.h"

EllipticASiftDetector::EllipticASiftDetector()
{
}

EllipticASiftDetector::ParallelOp::ParallelOp(const cv::Mat& _img, std::vector<std::vector<EllipticKeyPoint>> &kps, std::vector<cv::Mat> &dsps) {
	img = _img;
	keypoints_array = &kps[0];
	descriptors_array = &dsps[0];
}

void EllipticASiftDetector::ParallelOp::affineSkew(double tilt, double phi, cv::Mat& img, cv::Mat& mask, cv::Mat& Ai, cv::Mat& A) const 
{
	int h = img.rows;
	int w = img.cols;

	mask = cv::Mat(h, w, CV_8UC1, cv::Scalar(255)); 

	A = cv::Mat::eye(2,3, CV_64F);

	if(phi != 0.0)
	{
		phi *= M_PI / 180.;
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

void EllipticASiftDetector::ParallelOp::operator()( const cv::Range &r ) const {

	for (register int tl = r.start; tl != r.end; ++tl) {

		std::vector<EllipticKeyPoint>& keypoints0 = keypoints_array[tl-1];
		cv::Mat& descriptors0 = descriptors_array[tl-1];
		double t = pow(2, 0.5*tl);

		for(double /*TODO: used to be int, i changed dis(Ivan)*/ phi = 0; phi < 180; phi += 72.0/t)
		{
			std::vector<EllipticKeyPoint> ekps;
			std::vector<cv::KeyPoint> kps;
			std::vector<cv::Mat> affs;
			cv::Mat desc;

			cv::Mat timg, mask, Ai, A;

			if (method == "HESSIAN")
				gray_image.copyTo(timg);
			else
				img.copyTo(timg);

			affineSkew(t, phi, timg, mask, Ai, A);
			
			if (method == "CUSTOM")
			{
				doTracking(timg, kps, desc, "SIFT");

				ekps.resize(kps.size());
				for (unsigned int i = 0; i < kps.size(); i++)
				{
					ekps[i] = EllipticKeyPoint(kps[i], Ai);
				}
				keypoints0.insert(keypoints0.end(), ekps.begin(), ekps.end());
				descriptors0.push_back(desc);

				doTracking(timg, kps, desc, "cv::ORB");

				ekps.resize(kps.size());
				for (unsigned int i = 0; i < kps.size(); i++)
				{
					ekps[i] = EllipticKeyPoint(kps[i], Ai);
				}
				keypoints0.insert(keypoints0.end(), ekps.begin(), ekps.end());
				descriptors0.push_back(desc);
			}
			else if (method == "HESSIAN")
			{
				doHessian(timg, kps, affs, desc, method);

				ekps.resize(kps.size());
				for (unsigned int i = 0; i < kps.size(); i++)
				{
					ekps[i] = EllipticKeyPoint(kps[i], Ai);
					ekps[i].ownAffinity = affs[i];
				}
				keypoints0.insert(keypoints0.end(), ekps.begin(), ekps.end());
				descriptors0.push_back(desc);
			} else
			{
				doTracking(timg, kps, desc, method);

				ekps.resize(kps.size());
				for (unsigned int i = 0; i < kps.size(); i++)
				{
					ekps[i] = EllipticKeyPoint(kps[i], Ai);
				}
				keypoints0.insert(keypoints0.end(), ekps.begin(), ekps.end());
				descriptors0.push_back(desc);
			}
		}
	}
}

void EllipticASiftDetector::ParallelOp::doHessian(cv::Mat const &img, std::vector<cv::KeyPoint>& keypoints, std::vector<cv::Mat>& affines, cv::Mat& descriptors, std::string method) const
{
	HessianAffineParams par;

	// copy params 
	PyramidParams p;
	p.threshold = par.threshold;

	AffineShapeParams ap;
	ap.maxIterations = par.max_iter;
	ap.patchSize = par.patch_size;
	ap.mrSize = par.desc_factor;

	SIFTDescriptorParams sp;
	sp.patchSize = par.patch_size;

	HessianAffineDetector detector(img, p, ap, sp);
	detector.detectPyramidKeypoints(img);

	detector.getKeypoints(keypoints, affines, descriptors);
}

void EllipticASiftDetector::ParallelOp::doTracking(cv::Mat const &img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors, std::string method) const
{
	/*
	"FAST" – FastFeatureDetector
	"STAR" – StarFeatureDetector
	"SIFT" – SIFT (nonfree module)
	"SURF" – SURF (nonfree module)
	"cv::ORB" – cv::ORB
	"cv::BRISK" – cv::BRISK
	"MSER" – MSER
	"GFTT" – GoodFeaturesToTrackDetector
	"HARRIS" – GoodFeaturesToTrackDetector with Harris detector enabled
	"Dense" – DenseFeatureDetector
	"SimpleBlob" – SimpleBlobDetector
	*/
	if (method == "SIFT")
	{
		cv::Ptr<cv::Feature2D> f2d = cv::xfeatures2d::SIFT::create();
		f2d->detect(img, keypoints);
		f2d->compute(img, keypoints, descriptors);
	} else if (method == "SURF")
	{
		cv::Ptr<cv::Feature2D> f2d = cv::xfeatures2d::SURF::create();
		f2d->detect(img, keypoints);
		f2d->compute(img, keypoints, descriptors);
	}
	else if (method == "cv::ORB")
	{
		cv::Ptr<cv::ORB> orb = cv::ORB::create();
		orb->detect(img, keypoints);
		orb->compute(img, keypoints, descriptors);
	}
	else if (method == "cv::AKAZE")
	{
		cv::Ptr<cv::AKAZE> akaze = cv::AKAZE::create();
		akaze->detect(img, keypoints);
		akaze->compute(img, keypoints, descriptors);
	}
	else if (method == "cv::BRISK")
	{
		cv::Ptr<cv::BRISK> brisk = cv::BRISK::create();
		brisk->detect(img, keypoints);
		brisk->compute(img, keypoints, descriptors);
	}
	else if (method == "cv::KAZE")
	{
		cv::Ptr<cv::KAZE> brisk = cv::KAZE::create();
		brisk->detect(img, keypoints);
		brisk->compute(img, keypoints, descriptors);
	}
}

void EllipticASiftDetector::detectAndCompute(const cv::Mat& img, std::vector< EllipticKeyPoint >& keypoints, cv::Mat& descriptors, std::string method)
{
	_method = method;

	auto keypoints_array = std::vector<std::vector<EllipticKeyPoint>>(5);
	auto descriptors_array = std::vector<cv::Mat>(5);

	auto sum = EllipticASiftDetector::ParallelOp(img, keypoints_array, descriptors_array);
	sum.method = _method;

	if (method == "HESSIAN")
	{
		cv::Mat gray_image = cv::Mat::zeros(img.rows, img.cols, CV_32FC1);
		for (int r = 0; r < img.rows; ++r)
			for (int c = 0; c < img.cols; ++c)
				gray_image.at<float>(r, c) = ((float)img.at<cv::Vec3b>(r, c).val[0] + img.at<cv::Vec3b>(r, c).val[1] + img.at<cv::Vec3b>(r, c).val[2]) / 3.0f;

		sum.gray_image = gray_image;
	}

	parallel_for_(cv::Range(1, 6), sum); // non-inclusive end: 6 (elements: 1,2,3,4,5)

	// Merge!
	keypoints.clear();
	descriptors = cv::Mat(0, 128, CV_64F);

	for(auto tl = 0; tl < 5; tl++) {
		keypoints.insert(keypoints.end(), keypoints_array[tl].begin(), keypoints_array[tl].end());
		descriptors.push_back(descriptors_array[tl]);
	}

	if (method == "HESSIAN")
	{
		sum.gray_image.release();

		for (int i = 0; i < keypoints.size(); ++i)
			keypoints[i].transformation = keypoints[i].transformation * keypoints[i].ownAffinity;
	}
} 