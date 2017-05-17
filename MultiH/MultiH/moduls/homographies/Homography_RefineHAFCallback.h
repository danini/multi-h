#pragma once

#include <opencv2\calib3d\calib3d.hpp>
#include "Utilities.hpp"

template<typename T>
inline void RefineHomographyHAF(cv::InputArray _srcPoints, cv::InputArray _dstPoints, cv::InputArray _affines, cv::InputArray _epipole, cv::InputArray _F, cv::OutputArray _H)
{
	cv::Mat pts1 = _srcPoints.getMat();
	cv::Mat pts2 = _dstPoints.getMat();
	cv::Mat affines = _affines.getMat();
	cv::Mat F = _F.getMat();
	cv::Mat H = _H.getMat();
	cv::Mat epipole = _epipole.getMat();

	// Check the types
	CV_Assert((pts1.type() == CV_32F || pts1.type() == CV_64F || pts1.type() == CV_32FC2 || pts1.type() == CV_64FC2) && pts1.type() == pts2.type());
	CV_Assert(affines.type() == CV_32F || affines.type() == CV_64F);
	CV_Assert(F.type() == CV_32F || F.type() == CV_64F);
	CV_Assert(H.type() == CV_32F || H.type() == CV_64F);

	// Check the sizes
	CV_Assert(pts1.rows > 0 && pts1.rows == pts2.rows && pts1.cols == pts2.cols && affines.rows == pts1.rows && affines.cols == 4);
	CV_Assert(pts1.cols == 2 || pts1.cols == 3);

	// Check the fundamental matrix
	CV_Assert(F.cols == 3 && F.rows == 3);

	// Check the homography
	CV_Assert(H.cols == 3 && H.rows == 3);

	// Rescale the homography to make Lambda equals to one
	T lambda = (H.at<T>(0, 0) - epipole.at<T>(0) * H.at<T>(2, 0)) / F.at<T>(1, 0);
	H = H * (1.0 / lambda);

	cv::Mat H3 = cv::Mat_<T>(3, 1);
	H3.at<T>(0) = H.at<T>(2, 0);
	H3.at<T>(1) = H.at<T>(2, 1);
	H3.at<T>(2) = H.at<T>(2, 2);

	Homography_RefineHAFCallback<T> *refiner = new Homography_RefineHAFCallback<T>(pts1, pts2, affines, F, epipole, true);
	cv::Ptr<Homography_RefineHAFCallback<T>> refPtr(refiner);

	cv::Ptr<cv::LMSolverImpl> ptr(new cv::LMSolverImpl(refPtr, 1000));
	ptr->run(H3);

	T h31 = H3.at<T>(0);
	T h32 = H3.at<T>(1);
	T h33 = H3.at<T>(2);

	T h21 = epipole.at<T>(1) * h31 - F.at<T>(0, 0);
	T h22 = epipole.at<T>(1) * h32 - F.at<T>(0, 1);
	T h23 = epipole.at<T>(1) * h33 - F.at<T>(0, 2);
	T h11 = epipole.at<T>(0) * h31 + F.at<T>(1, 0);
	T h12 = epipole.at<T>(0) * h32 + F.at<T>(1, 1);
	T h13 = epipole.at<T>(0) * h33 + F.at<T>(1, 2);

	H = (cv::Mat_<T>(3, 3) << h11, h12, h13,
		h21, h22, h23,
		h31, h32, h33);
}

template<typename T>
class Homography_RefineHAFCallback : public cv::LMSolver::Callback
{
public:
	cv::Mat src, dst;
	cv::Mat affines;
	cv::Mat F, e;
	bool useReferenceAffine;

	Homography_RefineHAFCallback(cv::InputArray _src, cv::InputArray _dst, cv::InputArray _affines, cv::InputArray _F, cv::InputArray _e, bool _useReferenceAffine)
	{
		affines = _affines.getMat();
		src = _src.getMat();
		dst = _dst.getMat();
		F = _F.getMat();
		e = _e.getMat();
		useReferenceAffine = _useReferenceAffine;
	}

	~Homography_RefineHAFCallback() {}

	bool compute(cv::InputArray _param, cv::OutputArray _err, cv::OutputArray _Jac) const;
};

template<typename T>
bool Homography_RefineHAFCallback<T>::compute(cv::InputArray _param, cv::OutputArray _err, cv::OutputArray _Jac) const
{
	int i, count = src.rows;
	int eqNum = 6;
	int varNum = 3;
	cv::Mat param = _param.getMat();
	_err.getMatRef() = cv::Mat_<T>(count * eqNum, 1);
	cv::Mat err = _err.getMat(), J;
	if (_Jac.needed())
	{
		_Jac.getMatRef() = cv::Mat_<T>(count * eqNum, param.rows);
		J = _Jac.getMat();
		CV_Assert(J.isContinuous() && J.cols == 3);
	}

	const cv::Point2d* M = src.ptr<cv::Point2d>();
	const cv::Point2d* m = dst.ptr<cv::Point2d>();
	const T* h = param.ptr<T>();
	T* errptr = err.ptr<T>();
	T* Jptr = J.data ? J.ptr<T>() : 0;

	T ex = e.at<T>(0);
	T ey = e.at<T>(1);

	for (i = 0; i < count; i++)
	{
		T x1 = M[i].x;
		T y1 = M[i].y;
		T x2 = m[i].x;
		T y2 = m[i].y;

		T s = h[0] * x1 + h[1] * y1 + h[2];
		s = fabs(s) > DBL_EPSILON ? 1. / s : 0;

		T h21 = e.at<T>(1) * h[0] - F.at<T>(0, 0);
		T h22 = e.at<T>(1) * h[1] - F.at<T>(0, 1);
		T h23 = e.at<T>(1) * h[2] - F.at<T>(0, 2);
		T h11 = e.at<T>(0) * h[0] + F.at<T>(1, 0);
		T h12 = e.at<T>(0) * h[1] + F.at<T>(1, 1);
		T h13 = e.at<T>(0) * h[2] + F.at<T>(1, 2);

		T a11, a12, a21, a22;
		if (useReferenceAffine)
		{
			a11 = affines.at<T>(i, 0);
			a12 = affines.at<T>(i, 1);
			a21 = affines.at<T>(i, 2);
			a22 = affines.at<T>(i, 3);
		}
		else
		{
			a11 = (h11 - h[0] * x2) * s;
			a12 = (h12 - h[1] * y2) * s;
			a21 = (h21 - h[0] * x2) * s;
			a22 = (h22 - h[1] * y2) * s;
		}

		T xi = (h11 * x1 + h12 * y1 + h13)*s;
		T yi = (h21 * x1 + h22 * y1 + h23)*s;

		errptr[i * eqNum + 0] = -(a11 - s * (h[3] * F.at<T>(1, 0) + ex * h[0] - h[0] * xi)) - s * F.at<T>(1, 0);
		errptr[i * eqNum + 1] = -(a12 - s * (h[3] * F.at<T>(1, 1) + ex * h[1] - h[1] * xi)) - s * F.at<T>(1, 1);
		errptr[i * eqNum + 2] = -(a21 - s * (-h[3] * F.at<T>(0, 0) + ey * h[0] - h[0] * yi)) + s * F.at<T>(0, 0);
		errptr[i * eqNum + 3] = -(a22 - s * (-h[3] * F.at<T>(0, 1) + ey * h[1] - h[1] * yi)) + s * F.at<T>(0, 1);
		errptr[i * eqNum + 4] = x2 - xi + s * (F.at<T>(1, 0) * x1 + F.at<T>(1, 1) * y1 + F.at<T>(1, 2));
		errptr[i * eqNum + 5] = y2 - yi + s * (-F.at<T>(0, 0) * x1 - F.at<T>(0, 1) * y1 - F.at<T>(0, 2));

		if (Jptr)
		{
			Jptr[0] = s * (ex - xi);
			Jptr[1] = Jptr[2] = 0;

			Jptr[3] = 0;
			Jptr[4] = s * (ex - xi);
			Jptr[5] = 0;

			Jptr[6] = s * (ey - yi);
			Jptr[7] = Jptr[8] = 0;

			Jptr[9] = 0;
			Jptr[10] = s * (ey - yi);
			Jptr[11] = 0;

			Jptr[12] = -ex * s * x1;
			Jptr[13] = -ex * s * y1;
			Jptr[14] = -ex * s;

			Jptr[15] = -ey * s * x1;
			Jptr[16] = -ey * s * y1;
			Jptr[17] = -ey * s;
				
			Jptr += eqNum * varNum;
		}
	}

	return true;
}