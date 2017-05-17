#pragma once

#include <opencv2\calib3d\calib3d.hpp>
#include "Utilities.hpp"

template<typename T>
inline void RefineHomography3PT(cv::InputArray _srcPoints, cv::InputArray _dstPoints, cv::InputArray _epipole, cv::InputArray _F, cv::OutputArray _H)
{
	cv::Mat pts1 = _srcPoints.getMat();
	cv::Mat pts2 = _dstPoints.getMat();
	cv::Mat F = _F.getMat();
	cv::Mat H = _H.getMat();
	cv::Mat epipole = _epipole.getMat();

	// Check the types
	CV_Assert((pts1.type() == CV_32F || pts1.type() == CV_64F || pts1.type() == CV_32FC2 || pts1.type() == CV_64FC2) &&
		pts1.type() == pts2.type());
	CV_Assert(F.type() == CV_32F || F.type() == CV_64F);
	CV_Assert(H.type() == CV_32F || H.type() == CV_64F);

	// Check the sizes
	CV_Assert(pts1.rows > 0 && pts1.rows == pts2.rows && pts1.cols == pts2.cols);
	CV_Assert(pts1.cols == 2 || pts1.cols == 3);

	// Check the fundamental matrix
	CV_Assert(F.cols == 3 && F.rows == 3);

	// Check the homography
	CV_Assert(H.cols == 3 && H.rows == 3);
	
	cv::Mat H3 = cv::Mat_<T>(3, 1);
	H3.at<T>(0) = H.at<T>(2, 0);
	H3.at<T>(1) = H.at<T>(2, 1);
	H3.at<T>(2) = H.at<T>(2, 2);

	Homography_Refine3PTCallback<T> *refiner = new Homography_Refine3PTCallback<T>(pts1, pts2, F, epipole);
	cv::Ptr<Homography_Refine3PTCallback<T>> refPtr(refiner);

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

	_H.getMatRef() = H;
}

template<typename T>
class Homography_Refine3PTCallback : public cv::LMSolver::Callback
{
public:
	cv::Mat src, dst;
	cv::Mat F, e;

	Homography_Refine3PTCallback(cv::InputArray _src, cv::InputArray _dst, cv::InputArray _F, cv::InputArray _e)
	{
		src = _src.getMat();
		dst = _dst.getMat();
		F = _F.getMat();
		e = _e.getMat();
	}

	~Homography_Refine3PTCallback() {}

	bool compute(cv::InputArray _param, cv::OutputArray _err, cv::OutputArray _Jac) const;
};

template<typename T>
bool Homography_Refine3PTCallback<T>::compute(cv::InputArray _param, cv::OutputArray _err, cv::OutputArray _Jac) const
{
	int i, count = src.rows;
	int eqNum = 2;
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
			
		T xi = (h11 * x1 + h12 * y1 + h13)*s;
		T yi = (h21 * x1 + h22 * y1 + h23)*s;

		errptr[i * eqNum + 0] = x2 - xi;
		errptr[i * eqNum + 1] = y2 - yi;

		if (Jptr)
		{
			Jptr[0] = ex * s * x1;
			Jptr[1] = ex * s * y1;
			Jptr[2] = ex * s;

			Jptr[3] = ey * s * x1;
			Jptr[4] = ey * s * y1;
			Jptr[5] = ey * s;

			Jptr += eqNum * varNum;
		}
	}

	return true;
}


template<typename T>
inline void NormalizePoints(cv::InputArray points, cv::OutputArray results, cv::OutputArray transform)
{
	cv::Mat pts = points.getMat();

	CV_Assert(pts.type() == CV_32FC2 || pts.type() == CV_32FC3 || pts.type() == CV_64FC2 || pts.type() == CV_64FC3 || pts.type() == CV_32F || pts.type() == CV_64F);
	CV_Assert(pts.cols > 0 || pts.rows > 0);

	results.create(pts.size(), pts.type());
	cv::Mat resultMat = results.getMat();

	int N = pts.cols;
	T avgDist = 0;
	T avgRatio;

	if (pts.type() == CV_32F || pts.type() == CV_64F)
	{
		N = pts.rows;
		cv::Mat massPoint = cv::Mat::zeros(1, pts.cols, pts.type());

		for (unsigned int i = 0; i < N; ++i)
			massPoint += pts.row(i);
		massPoint = (1 / (T)N) * massPoint;

		// Translate the cloud to the origo
		for (unsigned int i = 0; i < N; ++i)
		{
			cv::Mat pt = pts.row(i) - massPoint;
			pt.copyTo(resultMat.row(i));
			avgDist = avgDist + norm(resultMat.row(i));
		}

		avgDist = avgDist / N;
		avgRatio = sqrt(2) / avgDist;

		// Scale the point cloud to sqrt(2)
		for (unsigned int i = 0; i < N; ++i)
			resultMat.row(i) = resultMat.row(i) * avgRatio;

		// Calculate normalization transformation
		cv::Mat _transform = cv::Mat_<T>::zeros(massPoint.cols + 1, massPoint.cols + 1);

		for (unsigned int i = 0; i < massPoint.cols; ++i)
		{
			_transform.at<T>(i, i) = avgRatio;
			_transform.at<T>(i, massPoint.cols) = -massPoint.at<T>(i) * avgRatio;
		}
		_transform.at<T>(massPoint.cols, massPoint.cols) = 1;

		transform.getMatRef() = _transform;

	}
	else if (pts.type() == CV_32FC3 || pts.type() == CV_64FC3) // Normalize 3D point cloud
	{
		cv::Point3_<T> massPoint;

		// Calculate mass point
		massPoint = cv::Point3_<T>(0, 0, 0);
		for (unsigned int i = 0; i < N; ++i)
			massPoint += pts.at<cv::Point3_<T>>(i);
		massPoint = (1 / (T)N) * massPoint;

		// Translate the cloud to the origo
		for (unsigned int i = 0; i < N; ++i)
		{
			resultMat.at<cv::Point3_<T>>(i) = pts.at<cv::Point3_<T>>(i) - massPoint;
			avgDist = avgDist + norm(resultMat.at<cv::Point3_<T>>(i));
		}

		avgDist = avgDist / N;
		avgRatio = sqrt(2) / avgDist;

		// Scale the point cloud to sqrt(2)
		for (unsigned int i = 0; i < N; ++i)
			resultMat.at<cv::Point3_<T>>(i) = resultMat.at<cv::Point3_<T>>(i) * avgRatio;

		// Calculate 3D normalization transformation
		cv::Mat _transform = cv::Mat_<T>::zeros(4, 4);

		_transform.at<T>(0, 0) = avgRatio;
		_transform.at<T>(1, 1) = avgRatio;
		_transform.at<T>(2, 2) = avgRatio;
		_transform.at<T>(3, 3) = 1;

		_transform.at<T>(0, 3) = -massPoint.x * avgRatio;
		_transform.at<T>(1, 3) = -massPoint.y * avgRatio;
		_transform.at<T>(2, 3) = -massPoint.z * avgRatio;

		transform.getMatRef() = _transform;
	}
	else // Normalize 2D point cloud
	{
		cv::Point_<T> massPoint;

		// Calculate mass point
		massPoint = cv::Point_<T>(0, 0);
		for (unsigned int i = 0; i < N; ++i)
			massPoint += pts.at<cv::Point_<T>>(i);
		massPoint = (1 / (T)N) * massPoint;

		// Translate the cloud to the origo
		for (unsigned int i = 0; i < N; ++i)
		{
			resultMat.at<cv::Point_<T>>(i) = pts.at<cv::Point_<T>>(i) - massPoint;
			avgDist = avgDist + norm(resultMat.at<cv::Point_<T>>(i));
		}

		avgDist = avgDist / N;
		avgRatio = sqrt(2) / avgDist;

		// Scale the point cloud to sqrt(2)
		for (unsigned int i = 0; i < N; ++i)
			resultMat.at<cv::Point_<T>>(i) = resultMat.at<cv::Point_<T>>(i) * avgRatio;

		// Calculate 2D normalization transformation
		cv::Mat _transform = cv::Mat_<T>::zeros(3, 3);

		_transform.at<T>(0, 0) = avgRatio;
		_transform.at<T>(1, 1) = avgRatio;
		_transform.at<T>(2, 2) = 1;

		_transform.at<T>(0, 2) = -massPoint.x * avgRatio;
		_transform.at<T>(1, 2) = -massPoint.y * avgRatio;

		transform.getMatRef() = _transform;
	}
}