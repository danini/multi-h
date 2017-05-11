#pragma once

#include <vector>
#include <map>
#include "moduls/alpha_expansion/GCoptimization.h"

#define DEFAULT_THRESHOLD_FUNDAMENTAL_MATRIX	3.0
#define DEFAULT_THRESHOLD_HOMOGRAPHY			2.5
#define DEFAULT_LOCALITY						0.002
#define DEFAULT_LAMBDA							0.5

#define DEFAULT_AFFINE_THRESHOLD				1.0
#define DEFAULT_LINENESS_THRESHOLD				0.005
#define MAX_ITERATION_NUMBER					500
#define CONVERGENCE_THRESHOLD					1e-5

#define USE_CONCURRENCY							1
#define LOG_TO_CONSOLE							0

class MultiH
{
public:
	struct EnergyDataStruct
	{
		const std::vector<cv::Point2d> * const src_points;
		const std::vector<cv::Point2d> * const dst_points;
		const std::vector<cv::Mat> * const homographies;
		const double energy_lambda;
		const double one_per_energy_lambda;
		const double truncated_sqr_threshold;
		const double sqr_homography_threshold;

		EnergyDataStruct(const std::vector<cv::Point2d> * const _p1,
			const std::vector<cv::Point2d> * const _p2, 
			const std::vector<cv::Mat> * const _hs,
			const double _lambda,
			const double _sqr_homography_threshold) :
			src_points(_p1),
			dst_points(_p2),
			homographies(_hs),
			energy_lambda(100 * _lambda),
			one_per_energy_lambda(100.0 / _lambda),
			sqr_homography_threshold(_sqr_homography_threshold),
			truncated_sqr_threshold(_sqr_homography_threshold * 81.0 / 16.0)
		{
		}
	};

	MultiH(double _thr_fund_mat = DEFAULT_THRESHOLD_FUNDAMENTAL_MATRIX, 
		double _thr_hom = DEFAULT_THRESHOLD_HOMOGRAPHY,
		double _locality = DEFAULT_LOCALITY,
		double _lambda = DEFAULT_LAMBDA,
		int _minimum_inlier_number = 0);

	~MultiH();
	void Release();

	bool Process(std::vector<cv::Point2d> _srcPoints, std::vector<cv::Point2d> _dstPoints, std::vector<cv::Mat> _affines);
	bool Process();

	int GetLabel(int idx) { return labeling[idx]; }
	void GetLabels(std::vector<int> &_labeling) { _labeling = labeling; }
	void GetSourcePoints(std::vector<cv::Point2d> &_src_points) { _src_points = src_points; }
	void GetDestinationPoints(std::vector<cv::Point2d> &_dst_points) { _dst_points = src_points; }
	void GetAffinities(std::vector<cv::Mat> &_affinities) { _affinities = affinities; }
	int GetPointNumber() { return static_cast<int>(labeling.size()); }
	int GetClusterNumber() { return static_cast<int>(cluster_homographies.size()); }
	int GetIterationNumber() { return final_iteration_number; }
	cv::Mat GetHomography(int idx) { return cluster_homographies[idx - 1]; }

	void DrawClusters(cv::Mat &img1, cv::Mat &img2, int size);
	void HomographyCompatibilityCheck();

	double GetEnergy() { return final_energy; }
	double GetHomographyThreshold() { return threshold_homography; }

protected:
	std::vector<cv::Point2d> src_points_original, dst_points_original; // The input point correspondences
	std::vector<cv::Point2d> src_points, dst_points; // Point correspondences after filtering
	std::vector<cv::Mat> affinities_original, affinities, homographies; // Input affinities, filtered affinities, point-wise homographies
	cv::Mat fundamental_matrix; // The fundamental matrix
	const double *fundamental_matrix_ptr; // Pointer to the fundamental matrix
	bool log_to_console; // Turn on/off logging to console
	bool degenerate_case; // Is the current case is a degenerate one
	double final_energy; // The resulting energy
	std::vector<std::vector<cv::DMatch>> neighbours; // The neighborhood	
	std::vector<cv::Mat> cluster_homographies; // The resulting homographies
	std::vector<int> labeling; // The resulting labeling
	cv::Mat epipole_1, epipole_2;
	cv::Mat R1, R1t, R2;

	int minimum_inlier_number; // The minimum inlier number of the obtained homographies
	double threshold_fundamental_matrix; // Threshold used for fundamental matrix estimation
	double threshold_homography, sqr_threshold_homography; // Threshold (and its square) used for homography estimation 
	double compatibility_threshold; // Threshold to decide whether a homography is compatible with F or not
	double locality_lambda; // The locality threshold
	double energy_lambda; // The weight of spatial coherence
	double affine_threshold; // Threshold for filtering an affine transformation
	double straightness_threshold; // Threshold for filtering points on a line
	int final_iteration_number; // The final iteration number
	
	void GetFundamentalMatrixAndRefineData(); // Estimate the fundamental matrix and refine data
	void ComputeLocalHomographies(); // Estimate point-wise homographies
	void EstablishStablePointSets(); // Establish stable homographies using the point-wise ones
	void ClusterMergingAndLabeling(); // The main alternating optimization

	void MergingStep(bool &changed); // The mode-seeking step
	void LabelingStep(double &energy, bool changed); // Labeling using alpha-expansion

	void HandleDegenerateCase(); // Handling degenerate cases
	void ComputeInliersOfHomography(int idx); // Get the inliers of a homography by thresholding

	// Apply HAF method to estimate homographies for each correspondence. Bibtex:
	// @article{barath2017theory,
	// 	  title = { A Theory of Point-wise Homography Estimation },
	// 	  author = { Barath, Daniel and Hajder, Levente },
	// 	  journal = { Pattern Recognition Letters },
	// 	  year = { 2017 }
	// }
	void GetHomographyHAF(double a11, double a12, double a21, double a22,
		double x1, double y1, double x2, double y2,
		cv::Mat &H);

	void GetHomographyHAFNonminimal(const cv::Mat &affines,
		const cv::Mat &pts1,
		const cv::Mat &pts2,
		cv::Mat &H,
		bool do_numerical_refinement = true);

	// Homography estimation using three correspondences and the fundamental matrix
	void GetHomography3PT(const cv::Mat &pts1,
		const cv::Mat &pts2,
		cv::Mat &H,
		bool do_numerical_refinement = true);

	// Refinement of point coordinates using the fundamental matrix
	bool OptimalTriangulation(cv::Mat pt1, cv::Mat pt2, cv::Mat &pt1Out, cv::Mat &pt2Out);

	// Calculating the consistency of a local affine transformation with the fundamental matrix and refine
	// @inproceedings{barath2016accurate,
	// 	  title = { Accurate Closed - form Estimation of Local Affine Transformations Consistent with the Epipolar Geometry },
	// 	  author = { Barath, Daniel and Hajder, Levente and cv::Matas, Jiri },
	// 	 booktitle = { 27th British Machine Vision Conference },
	// 	  year = { 2016 }
	// }
	void GetAffineConsistency(cv::Mat fundamental_matrix_transposed, cv::Mat A, cv::Mat _pt1, cv::Mat pt2, double &scaleError, double &angularError, double &distanceError);
	double GetBetaScale(cv::Mat fundamental_matrix_transposed, cv::Mat A, cv::Mat pt1, cv::Mat pt2);
	void GetOptimalAffineTransformation(cv::Mat A, cv::Mat fundamental_matrix_transpose, cv::Mat pt1, cv::Mat pt2, cv::Mat &optimalA);
};


