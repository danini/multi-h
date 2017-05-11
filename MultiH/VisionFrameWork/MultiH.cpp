#include "stdafx.h"
#include "MultiH.h"
#include "moduls/alpha_expansion/GCoptimization.h"
#include "moduls/mode_seeking/median_shift.h"
#include <chrono>
#include "moduls\homographies\Homography_RefineHAFCallback.h"
#include "moduls\homographies\Homography_Refine3PTCallback.h"

MultiH::MultiH(double _thr_fund_mat, double _thr_hom, double _locality, double _lambda, int _minimum_inlier_number) :
	threshold_fundamental_matrix(_thr_fund_mat),
	threshold_homography(_thr_hom),
	sqr_threshold_homography(_thr_hom * _thr_hom),
	locality_lambda(_locality),
	energy_lambda(_lambda),
	minimum_inlier_number(_minimum_inlier_number),
	affine_threshold(DEFAULT_AFFINE_THRESHOLD),
	straightness_threshold(DEFAULT_LINENESS_THRESHOLD),
	log_to_console(LOG_TO_CONSOLE)
{
}

MultiH::~MultiH()
{
	Release();
}

void MultiH::Release()
{
}

bool MultiH::Process(std::vector<cv::Point2d> _srcPoints, std::vector<cv::Point2d> _dstPoints, std::vector<cv::Mat> _affines)
{
	printf("[Multi-H] Processing has been started.\n");
	src_points_original = _srcPoints;
	dst_points_original = _dstPoints;
	affinities_original = _affines;

	return Process();
}

bool MultiH::Process()
{
	if (src_points_original.size() < 8 || 
		dst_points_original.size() != src_points_original.size() || 
		affinities_original.size() != src_points_original.size())
	{
		std::cerr << "Error: Features are not set!\n";
		return false;
	}

	GetFundamentalMatrixAndRefineData(); // Estimate fundamental matrix
	
	// If fundamental matrix is not estimable run RANSAC homography fitting.
	if (degenerate_case) 
	{
		HandleDegenerateCase();
	}
	else
	{
		std::chrono::time_point<std::chrono::system_clock> start, end;
		std::chrono::duration<double> elapsed_seconds;
		start = std::chrono::system_clock::now();

		ComputeLocalHomographies(); // Estimate point-wise homographies
		end = std::chrono::system_clock::now();
		elapsed_seconds = end - start;
		printf("[Multi-H] Point-wise homography estimation time = %f secs\n", elapsed_seconds.count());

		start = std::chrono::system_clock::now();
		EstablishStablePointSets(); // Refine local homography exploiting spatial coherence
		end = std::chrono::system_clock::now();
		elapsed_seconds = end - start;
		printf("[Multi-H] Stable cluster estimation time = %f secs\n", elapsed_seconds.count());

		ClusterMergingAndLabeling();

		if (cluster_homographies.size() > 1)
		{
			int cluster_number_before_filtering = cluster_homographies.size();
			start = std::chrono::system_clock::now();
			HomographyCompatibilityCheck();
			end = std::chrono::system_clock::now();
			elapsed_seconds = end - start;
			printf("[Multi-H] Compatibility check time = %f secs (%d clusters removed from %d)\n", elapsed_seconds.count(), cluster_number_before_filtering - cluster_homographies.size(), cluster_number_before_filtering);
		}

		if (cluster_homographies.size() <= 1)
		{
			labeling.clear();
			labeling.resize(src_points.size(), -1);
			cluster_homographies.resize(0);
			HandleDegenerateCase();
		}
	}

	return true;
}

void MultiH::HomographyCompatibilityCheck()
{
	std::vector<std::vector<cv::Point2d>> src_points_per_cluster(cluster_homographies.size());
	std::vector<std::vector<cv::Point2d>> dst_points_per_cluster(cluster_homographies.size());
	std::vector<bool> remove_cluster(cluster_homographies.size(), false);

	for (int i = 0; i < labeling.size(); ++i)
	{
		int l = labeling[i];
		if (l > -1)
		{
			src_points_per_cluster[l].push_back(src_points[i]);
			dst_points_per_cluster[l].push_back(dst_points[i]);
		}
	}

	// A cross-validation-based method to determine whether a homography is stable or not
#if USE_CONCURRENCY
	concurrency::parallel_for(0, static_cast<int>(cluster_homographies.size()), [&](int i)
#else
	for (int i = 0; i < cluster_homographies.size(); ++i)
#endif
	{
		bool remove = false;

		std::vector<cv::Point2d> *src_points = &src_points_per_cluster[i];
		std::vector<cv::Point2d> *dst_points = &dst_points_per_cluster[i];
		const int N = src_points->size();
		const int trials = MAX(501, MIN(501, N * (N - 1) * (N - 2) / 6)); // n choose 3
		double mss_src_points_vec[6], mss_dst_points_vec[6];
		cv::Mat mss_src_points(3, 2, CV_64F, mss_src_points_vec);
		cv::Mat mss_dst_points(3, 2, CV_64F, mss_dst_points_vec);
		std::vector<double> distances(trials);

		if (N >= MAX(minimum_inlier_number, 4))
		{
			std::vector<double> distance_of_points(src_points->size());
			for (int t = 0; t < trials; ++t)
			{
				for (int j = 0; j < 3; ++j)
				{
					int p_idx = j * 2;
					int idx = (src_points->size() - 1) * (static_cast<double>(rand()) / RAND_MAX);

					mss_src_points_vec[p_idx] = src_points->at(idx).x;
					mss_src_points_vec[p_idx + 1] = src_points->at(idx).y;
					mss_dst_points_vec[p_idx] = dst_points->at(idx).x;
					mss_dst_points_vec[p_idx + 1] = dst_points->at(idx).y;

					src_points->erase(src_points->begin() + idx);
					dst_points->erase(dst_points->begin() + idx);
				}

				cv::Mat H(3, 3, CV_64F);
				GetHomography3PT(mss_src_points, mss_dst_points, H, false);
				const double * h = (double *)H.data;

				double s, x1, y1, dx, dy;
				double mean_distance_per_trial = 0;
				for (int j = 0; j < src_points->size(); ++j)
				{
					const double ox1 = src_points->at(j).x;
					const double oy1 = src_points->at(j).y;

					s = *(h + 6) * ox1 + *(h + 7) * oy1 + *(h + 8);
					x1 = (*(h)* ox1 + *(h + 1) * oy1 + *(h + 2)) / s;
					y1 = (*(h + 3)* ox1 + *(h + 4) * oy1 + *(h + 5)) / s;

					dx = dst_points->at(j).x - x1;
					dy = dst_points->at(j).y - y1;

					distance_of_points[j] = dx*dx + dy*dy;
					//mean_distance_per_trial += dx*dx + dy*dy;
				}
				//mean_distance_per_trial = mean_distance_per_trial / src_points->size();
				std::sort(distance_of_points.begin(), distance_of_points.end());
				distances[t] = src_points->size() % 2 ? distance_of_points[src_points->size() / 2] : 0.5 * (distance_of_points[src_points->size() / 2] + distance_of_points[src_points->size() / 2 + 1]); //mean_distance_per_trial;

				src_points->resize(N);
				dst_points->resize(N);
				for (int j = 0; j < 3; ++j)
				{
					int p_idx_1 = j * 2;
					int p_idx_2 = N - j - 1;

					src_points->at(p_idx_2).x = mss_src_points_vec[p_idx_1];
					src_points->at(p_idx_2).y = mss_src_points_vec[p_idx_1 + 1];
					dst_points->at(p_idx_2).x = mss_dst_points_vec[p_idx_1];
					dst_points->at(p_idx_2).y = mss_dst_points_vec[p_idx_1 + 1];
				}
			}

			std::sort(distances.begin(), distances.end());
			double median = trials % 2 ? distances[trials / 2] : 0.5 * (distances[trials / 2] + distances[trials / 2 + 1]);

			remove = median > sqr_threshold_homography * 81.0 / 16.0;
			//if (log_to_console)
				printf("Median distance = %f (%d, %f)\n", median, i, sqr_threshold_homography * 81.0 / 16.0);
		}
		else if (N < minimum_inlier_number)
			remove = true;

		remove_cluster[i] = remove;
	}
#if USE_CONCURRENCY
	);
#endif

	for (int i = cluster_homographies.size() - 1; i >= 0; --i)
	{
		if (remove_cluster[i])
		{
			for (int j = 0; j < labeling.size(); ++j)
			{
				if (labeling[j] == i)
					labeling[j] = -1;
				else if (labeling[j] > i)
					--labeling[j];
			}
			cluster_homographies.erase(cluster_homographies.begin() + i);
		}
	}
}

void MultiH::ClusterMergingAndLabeling()
{
	// Determine neighbourhood
	std::chrono::time_point<std::chrono::system_clock> start, end;
	std::chrono::duration<double> elapsed_seconds;
	start = std::chrono::system_clock::now();

	int N = static_cast<int>(src_points.size());

	cv::Mat pointVectors = (cv::Mat_<float>(N, 4));
	float *pv_ptr = (float *)pointVectors.data;

#if USE_CONCURRENCY
	concurrency::parallel_for(0, N, [&](int i)
#else
	for (int i = 0; i < N; ++i)
#endif
	{
		const int idx = i * 4;
		*(pv_ptr + idx) = static_cast<float>(src_points.at(i).x);
		*(pv_ptr + idx + 1) = static_cast<float>(src_points.at(i).y);
		*(pv_ptr + idx + 2) = static_cast<float>(dst_points.at(i).x);
		*(pv_ptr + idx + 3) = static_cast<float>(dst_points.at(i).y);
	}
#if USE_CONCURRENCY
		);
#endif

	cv::FlannBasedMatcher flann;
	flann.radiusMatch(pointVectors, pointVectors, neighbours, 1.0 / locality_lambda);

	end = std::chrono::system_clock::now();
	elapsed_seconds = end - start;

	printf("[Multi-H] Adjacency-matrix calculation time = %f secs\n", elapsed_seconds.count());

	// Second mean shift round
	start = std::chrono::system_clock::now();
	int iteration_number = 0;
	labeling.resize(N, -1);
	double lastEnergy = INT_MAX;
	int not_changed_number = 0;
	
	while (iteration_number++ < MAX_ITERATION_NUMBER)
	{
		printf("[Multi-H] Optimization started... Iteration %d (max %d)\r", iteration_number, MAX_ITERATION_NUMBER);

		bool changed = false;
		
		MergingStep(changed);

		if (changed)
			not_changed_number = 0;
		else
			++not_changed_number;

		if (cluster_homographies.size() == 1)
		{
			labeling.resize(N, -1);
			ComputeInliersOfHomography(0);
			break;
		}
		else if (cluster_homographies.size() == 0)
			break;
		
		double energy;
		LabelingStep(energy, changed);

		if (log_to_console)
			printf("Iteration %d.   Number of clusters = %d   Energy = %f\n", iteration_number, cluster_homographies.size(), energy);
		
		if ((!changed && abs(lastEnergy - energy) < CONVERGENCE_THRESHOLD) || not_changed_number > 10)
		{
			final_energy = energy;
			if (log_to_console)
				printf("Number of clusters = %d   Energy = %f\n", cluster_homographies.size(), energy);
			break;
		}

		lastEnergy = energy;
	}
	printf("[Multi-H] Optimization started... Iteration %d\n", iteration_number);

	end = std::chrono::system_clock::now();
	elapsed_seconds = end - start;

	printf("[Multi-H] Alternating optimization time = %f secs\n", elapsed_seconds.count());
	final_iteration_number = iteration_number - 1;
}

void MultiH::DrawClusters(cv::Mat &img1, cv::Mat &img2, int size)
{
	std::vector<cv::Scalar> colors(cluster_homographies.size() + 1);

	colors[0] = cv::Scalar(0, 0, 0);
	for (int i = 1; i < cluster_homographies.size() + 1; ++i)
		colors[i] = cv::Scalar(255 * (float)rand() / RAND_MAX, 255 * (float)rand() / RAND_MAX, 255 * (float)rand() / RAND_MAX);

	colors[1] = cv::Scalar(255, 0, 0);
	colors[2] = cv::Scalar(255, 255, 255);
	colors[3] = cv::Scalar(0, 0, 255);
	colors[4] = cv::Scalar(255, 255, 0);
	colors[5] = cv::Scalar(255, 0, 255);
	colors[6] = cv::Scalar(0, 255, 255);
	colors[7] = cv::Scalar(0, 255, 0);
	colors[8] = cv::Scalar(127, 255, 0);
	colors[9] = cv::Scalar(0, 255, 127);
	colors[10] = cv::Scalar(127, 127, 0);
	colors[11] = cv::Scalar(63, 255, 127);
	
	for (int i = 0; i < labeling.size(); ++i)
	{
		if (labeling[i] == -1)
			continue;

		if (cluster_homographies.size() == 1)
		{
			circle(img1, src_points_original[i], size, colors[labeling[i] + 1], -1);
			circle(img2, dst_points_original[i], size, colors[labeling[i] + 1], -1);
		}
		else
		{
			circle(img1, src_points[i], size, colors[labeling[i] + 1], -1);
			circle(img2, dst_points[i], size, colors[labeling[i] + 1], -1);
		}
	}
}

void MultiH::MergingStep(bool &changed)
{
	const int N = static_cast<int>(cluster_homographies.size());
	cv::Mat featureVectors(N, 6, CV_32F);
	float *fv_ptr = (float *)featureVectors.data;

#if USE_CONCURRENCY
	concurrency::parallel_for(0, N, [&](int i)
#else
	for (int i = 0; i < N; ++i)
#endif
	{
		const double *h = (double *)cluster_homographies[i].data;
		const int idx = i * 6;

		// Image of point [0,0,1]
		const double s1 = *(h + 8);
		const double x1 = *(h + 2) / s1;
		const double y1 = *(h + 5) / s1;

		// Image of point [1,0,1]
		const double s2 = *(h + 6) + *(h + 8);
		const double x2 = (*(h + 0) + *(h + 2)) / s2;
		const double y2 = (*(h + 3) + *(h + 5)) / s2;

		// Image of point [0,1,1]
		const double s3 = *(h + 7) + *(h + 8);
		const double x3 = (*(h + 1) + *(h + 2)) / s3;
		const double y3 = (*(h + 4) + *(h + 5)) / s3;

		*(fv_ptr + idx) = static_cast<float>(x1);
		*(fv_ptr + idx + 1) = static_cast<float>(y1);
		*(fv_ptr + idx + 2) = static_cast<float>(x2);
		*(fv_ptr + idx + 3) = static_cast<float>(y2);
		*(fv_ptr + idx + 4) = static_cast<float>(x3);
		*(fv_ptr + idx + 5) = static_cast<float>(y3);

}
#if USE_CONCURRENCY
	);
#endif

	MedianShift median_shift;
	median_shift.Initialize(&featureVectors, false, 2, 10000, false, false, MedianShift::TUKEY_MEDIAN);
	median_shift.Run(50, threshold_homography);

	cv::Mat resulting_clusters;
	median_shift.GetClusters(resulting_clusters);
	
	std::vector<cv::Mat> homographies;
	homographies.reserve(cluster_homographies.size());

	double pts1_coords[] = { 0, 0, 1, 0, 0, 1 };
	double pts2_coords[6];
	cv::Mat pts1(3, 2, CV_64F, pts1_coords);
	cv::Mat pts2(3, 2, CV_64F, pts2_coords);

	float *resulting_clusters_ptr = (float *)resulting_clusters.data;

	for (int i = 0; i < resulting_clusters.rows; ++i)
	{
		int idx = i * 6;

		pts2_coords[0] = (double)*(resulting_clusters_ptr + idx);
		pts2_coords[1] = (double)*(resulting_clusters_ptr + idx + 1);
		pts2_coords[2] = (double)*(resulting_clusters_ptr + idx + 2);
		pts2_coords[3] = (double)*(resulting_clusters_ptr + idx + 3);
		pts2_coords[4] = (double)*(resulting_clusters_ptr + idx + 4);
		pts2_coords[5] = (double)*(resulting_clusters_ptr + idx + 5);

		cv::Mat H(3, 3, CV_64F);
		GetHomography3PT(pts1, pts2, H);
		const double *h = (double *)H.data;

		std::vector<int> inliers;
		double s, x1, y1, dx, dy;
		for (int j = 0; j < src_points.size(); ++j)
		{
			s = *(h + 6) * src_points.at(j).x + *(h + 7) * src_points.at(j).y + *(h + 8);
			x1 = (*(h)* src_points.at(j).x + *(h + 1) * src_points.at(j).y + *(h + 2)) / s;
			y1 = (*(h + 3)* src_points.at(j).x + *(h + 4) * src_points.at(j).y + *(h + 5)) / s;
			
			dx = dst_points.at(j).x - x1;
			dy = dst_points.at(j).y - y1;

			if (dx*dx + dy*dy < sqr_threshold_homography)
				inliers.push_back(j);
		}

		// Fit line to the points in the first image
		cv::Mat line_parameters = cv::Mat_<double>(inliers.size(), 3);
		double *line_parameters_ptr = (double *)line_parameters.data;
		for (int j = 0; j < inliers.size(); ++j)
		{
			int idx = inliers[j];
			*(line_parameters_ptr++) = src_points.at(idx).x;
			*(line_parameters_ptr++) = src_points.at(idx).y;
			*(line_parameters_ptr++) = 1;
		}

		cv::Mat evalues, evectors;

		line_parameters = line_parameters.t() * line_parameters;
		eigen(line_parameters, evalues, evectors);

		// Remove homography if the points lie on a straight line
		if (evalues.at<double>(2) < straightness_threshold || inliers.size() < 3)
			continue;

		homographies.push_back(H);
	}

	changed = homographies.size() != cluster_homographies.size();
	if (changed)
		cluster_homographies = homographies;
}

inline int dataEnergy(int p, int l, void *data)
{
	MultiH::EnergyDataStruct *myData = (MultiH::EnergyDataStruct *)data;
	const double lambda = myData->one_per_energy_lambda;	// 1 / Lambda

	if (l == 0)
		return round(lambda * myData->truncated_sqr_threshold);

	const std::vector<cv::Mat> * const homographies = myData->homographies;
	const std::vector<cv::Point2d> * const src_points = myData->src_points;
	const std::vector<cv::Point2d> * const dst_points = myData->dst_points;

	const double *h = (double *)homographies->at(l - 1).data;
	const double ox1 = src_points->at(p).x;
	const double oy1 = src_points->at(p).y;
	const double x2 = dst_points->at(p).x;
	const double y2 = dst_points->at(p).y;
	
	const double s1 = *(h + 6) * ox1 + *(h + 7) * oy1 + *(h + 8);
	const double x1 = (*(h)* ox1 + *(h + 1) * oy1 + *(h + 2)) / s1;
	const double y1 = (*(h + 3) * ox1 + *(h + 4) * oy1 + *(h + 5)) / s1;

	const double dx = x1 - x2;
	const double dy = y1 - y2;

	const double distance = dx*dx + dy*dy;	
	const double truncated_threshold = myData->truncated_sqr_threshold;

	if (distance < truncated_threshold)
		return round(lambda * (1.0f - (distance / truncated_threshold)));
	return 2 * round(lambda * myData->truncated_sqr_threshold);
}

inline int smoothnessEnergy(int p1, int p2, int l1, int l2, void *data)
{
	MultiH::EnergyDataStruct *myData = (MultiH::EnergyDataStruct *)data;
	const double lambda = myData->energy_lambda;
	return l1 != l2 ? round(lambda) : 0;
}

void MultiH::LabelingStep(double &energy, bool changed)
{
	energy = 0;
	const int Np = src_points.size();
	const int Nh = cluster_homographies.size();
				
	// Set up alpha-expansion algorithm
	GCoptimizationGeneralGraph *gc_optimizator = new GCoptimizationGeneralGraph(Np, Nh + 1);
	EnergyDataStruct toFn(&src_points, &dst_points, &cluster_homographies, energy_lambda, sqr_threshold_homography);
	gc_optimizator->setDataCost(&dataEnergy, &toFn);
	gc_optimizator->setSmoothCost(&smoothnessEnergy, &toFn);
	
	if (!changed) // Set the previous labeling as initial if the mode-seeking didn't change anything
	{
		for (int i = 0; i < labeling.size(); ++i)
			gc_optimizator->setLabel(i, labeling[i] + 1);
	}

	// Set neighbourhood
	for (int i = 0; i < neighbours.size(); ++i)
	{
		for (int j = 0; j < neighbours[i].size(); ++j)
		{
			int idx = neighbours[i][j].trainIdx;
			if (idx != i)
				gc_optimizator->setNeighbors(i, idx);
		}
	}

	int iteration_number = 0;
	energy = gc_optimizator->expansion(iteration_number, 1000);

	std::vector<cv::Mat> pts1(Nh), pts2(Nh), tmp_affines(Nh);
	std::vector<int> inlier_number(Nh, 0);
	for (int i = 0; i < Np; ++i)
	{
		int l = gc_optimizator->whatLabel(i);
		gc_optimizator->setLabel(i, l - 1);
		if (l == 0)
			continue;

		++inlier_number[l - 1];
	}

	for (int i = 0; i < Nh; ++i)
	{
		pts1[i] = cv::Mat_<double>(inlier_number[i], 2);
		pts2[i] = cv::Mat_<double>(inlier_number[i], 2);
		tmp_affines[i] = cv::Mat_<double>(inlier_number[i], 4);
	}

	std::vector<int> current_indices(Nh, 0);
	for (int i = 0; i < Np; ++i)
	{
		int l = gc_optimizator->whatLabel(i);
		labeling[i] = l;

		if (l == -1)
			continue;

		pts1[l].at<double>(current_indices[l], 0) = src_points.at(i).x;
		pts1[l].at<double>(current_indices[l], 1) = src_points.at(i).y;
		pts2[l].at<double>(current_indices[l], 0) = dst_points.at(i).x;
		pts2[l].at<double>(current_indices[l], 1) = dst_points.at(i).y;

		tmp_affines[l].at<double>(current_indices[l], 0) = affinities.at(i).at<double>(0, 0);
		tmp_affines[l].at<double>(current_indices[l], 1) = affinities.at(i).at<double>(0, 1);
		tmp_affines[l].at<double>(current_indices[l], 2) = affinities.at(i).at<double>(1, 0);
		tmp_affines[l].at<double>(current_indices[l], 3) = affinities.at(i).at<double>(1, 1);

		++current_indices[l];
	}

#if USE_CONCURRENCY
	concurrency::parallel_for(0, static_cast<int>(pts1.size()), [&](int i)
#else
	for (int i = 0; i < pts1.size(); ++i)
#endif
	{
		if (pts1[i].rows == 0)
			return;

		GetHomographyHAFNonminimal(tmp_affines[i], pts1[i], pts2[i], cluster_homographies[i]);		
	}
#if USE_CONCURRENCY
	);
#endif
		
	delete gc_optimizator;
}

void MultiH::EstablishStablePointSets()
{
	// Compute feature std::vector for each homography
	const int N = static_cast<int>(src_points.size());
	cv::Mat featureVectors(N, 10, CV_32F);
	float *fv_ptr = (float *)featureVectors.data;

#if USE_CONCURRENCY
	concurrency::parallel_for(0, N, [&](int i)
#else
	for (int i = 0; i < N; ++i)
#endif
	{
		const double *h = (double *)homographies[i].data;
		const int idx = i * 10;

		// Image of point [0,0,1]
		const double s1 = *(h + 8);
		const double x1 = *(h + 2) / s1;
		const double y1 = *(h + 5) / s1;

		// Image of point [1,0,1]
		const double s2 = *(h + 6) + *(h + 8);
		const double x2 = (*(h + 0) + *(h + 2)) / s2;
		const double y2 = (*(h + 3) + *(h + 5)) / s2;

		// Image of point [0,1,1]
		const double s3 = *(h + 7) + *(h + 8);
		const double x3 = (*(h + 1) + *(h + 2)) / s3;
		const double y3 = (*(h + 4) + *(h + 5)) / s3;
						
		*(fv_ptr + idx) = static_cast<float>(x1);
		*(fv_ptr + idx + 1) = static_cast<float>(x2);
		*(fv_ptr + idx + 2) = static_cast<float>(x3);
		*(fv_ptr + idx + 3) = static_cast<float>(y1);
		*(fv_ptr + idx + 4) = static_cast<float>(y2);
		*(fv_ptr + idx + 5) = static_cast<float>(y3);
		*(fv_ptr + idx + 6) = static_cast<float>(src_points[i].x * locality_lambda);
		*(fv_ptr + idx + 7) = static_cast<float>(src_points[i].y * locality_lambda);
		*(fv_ptr + idx + 8) = static_cast<float>(dst_points[i].x * locality_lambda);
		*(fv_ptr + idx + 9) = static_cast<float>(dst_points[i].y * locality_lambda);

	}
#if USE_CONCURRENCY
	);
#endif

	cv::Mat clusters;
	std::vector<std::vector<int>> clusterPoints;

	MedianShift median_shift;
	median_shift.Initialize(&featureVectors, false, 2, 10000, true, false, MedianShift::TUKEY_MEDIAN);
	median_shift.Run(50, threshold_homography);

	median_shift.GetClusters(clusters);
	median_shift.GetClusterPoints(clusterPoints);

	// Filter the obtained clusters and apply LSQ homography fitting
	for (int i = 0; i < clusters.rows; ++i)
	{
		const int Ni = clusterPoints[i].size();
		if (Ni < 3) 
			continue;

		// Compute the centroid and fit line to the points
		cv::Mat pts1 = cv::Mat_<double>(Ni, 2), pts2 = cv::Mat_<double>(Ni, 2);
		double *pts1_ptr = (double *)pts1.data, *pts2_ptr = (double *)pts2.data;
		for (int j = 0; j < Ni; ++j)
		{
			const int idx = clusterPoints[i][j];

			*(pts1_ptr++) = src_points[idx].x;
			*(pts1_ptr++) = src_points[idx].y;
			*(pts2_ptr++) = dst_points[idx].x;
			*(pts2_ptr++) = dst_points[idx].y;
		}

		// Fit homography in LSQ sense to the points
		cv::Mat H(3, 3, CV_64F);
		GetHomography3PT(pts1, pts2, H);

		cluster_homographies.push_back(H);
	}

	clusterPoints.resize(0);

	if (log_to_console)
		printf("[Multi-H] Number of stable, local clusters = %d\n", cluster_homographies.size());
}

void MultiH::ComputeLocalHomographies()
{
	homographies.resize(affinities.size());

#if USE_CONCURRENCY
	concurrency::parallel_for(0, static_cast<int>(src_points.size()), [&](int i)
#else
	for (int i = 0; i < src_points.size(); ++i)
#endif
	{
		// Add to class containers
		const double *a = (double *)affinities[i].data;
		homographies[i] = cv::Mat(3, 3, CV_64F);

		GetHomographyHAF(*a, *(a + 1), *(a + 2), *(a + 3),
			src_points[i].x, src_points[i].y, dst_points[i].x, dst_points[i].y,
			homographies[i]);
	}
#if USE_CONCURRENCY
	);
#endif
}

void MultiH::HandleDegenerateCase()
{
	if (log_to_console)	
		printf("Handle degenerate case. Apply RANSAC homography estimation to the point correspondences\n");
	
	std::vector<uchar> mask;
	cv::Mat H = findHomography(src_points_original, dst_points_original, CV_RANSAC, threshold_homography, mask);

	labeling.resize(src_points_original.size());
#if USE_CONCURRENCY
	concurrency::parallel_for(0, static_cast<int>(src_points_original.size()), [&](int i)
#else
	for (int i = 0; i < src_points_original.size(); ++i)
#endif
	{
		labeling[i] = mask[i] ? 0 : -1;
	}
#if USE_CONCURRENCY
	);
#endif

	cluster_homographies.push_back(H);
}

void MultiH::ComputeInliersOfHomography(int idx)
{
	const int N = static_cast<int>(src_points.size());

#if USE_CONCURRENCY
	concurrency::parallel_for(0, N, [&](int i)
#else
	for (int i = 0; i < N; ++i)
#endif
	{
		const double *h = (double *)cluster_homographies[idx].data;

		const double s = *(h + 6) * src_points[i].x + *(h + 7) * src_points[i].y + *(h + 8);
		const double x1 = (*(h)* src_points[i].x + *(h + 1) * src_points[i].y + *(h + 2)) / s;
		const double y1 = (*(h + 3)* src_points[i].x + *(h + 4) * src_points[i].y + *(h + 5)) / s;
		
		const double dx = x1 - dst_points[i].x; 
		const double dy = y1 - dst_points[i].y;

		if (dx*dx + dy*dy < sqr_threshold_homography)
			labeling[i] = idx;
	}
#if USE_CONCURRENCY
	);
#endif
}

void MultiH::GetFundamentalMatrixAndRefineData()
{
	// Estimate fundamental matrix
	// TODO: change to LO-RANSAC
	std::vector<uchar> mask;
	fundamental_matrix = findFundamentalMat(src_points_original, dst_points_original, CV_FM_RANSAC, threshold_fundamental_matrix, 0.99, mask);
	fundamental_matrix_ptr = (double *)fundamental_matrix.data;

	// TODO: handle degenerate cases 
	if (norm(fundamental_matrix) < 1e-5)
	{
		degenerate_case = true;
		printf("[Multi-H] Degenerate case, the fundamental matrix cannot be estimated.\n");
		return;
	}

	cv::Mat Ft = fundamental_matrix.t();

	// Compute the epipole on the second image
	cv::Mat FFt = fundamental_matrix * Ft;
	cv::Mat evalues1, evectors1;
	eigen(FFt, evalues1, evectors1);
	epipole_2 = evectors1.row(evectors1.rows - 1);
	epipole_2 = epipole_2 / epipole_2.at<double>(2);

	cv::Mat FtF = Ft * fundamental_matrix;
	cv::Mat evalues2, evectors2;
	eigen(FtF, evalues2, evectors2);
	epipole_1 = evectors2.row(evectors2.rows - 1);
	epipole_1 = epipole_1 / epipole_1.at<double>(2);

	R1 = (cv::Mat_<double>(3, 3, CV_64F) << epipole_1.at<double>(0), epipole_1.at<double>(1), 0, -epipole_1.at<double>(1), epipole_1.at<double>(0), 0, 0, 0, 1);
	R2 = (cv::Mat_<double>(3, 3, CV_64F) << -epipole_2.at<double>(0), -epipole_2.at<double>(1), 0, epipole_2.at<double>(1), -epipole_2.at<double>(0), 0, 0, 0, 1);
	R1t = R1.t();

	degenerate_case = false;

	for (int i = 0; i < src_points_original.size(); ++i)
	{
		if (mask[i])
		{
			cv::Mat a = (cv::Mat)src_points_original[i];
			cv::Mat b = (cv::Mat)dst_points_original[i];

			// Apply Hartley & Sturm optimization to the original point locations
			cv::Mat c, d;
			bool error = !OptimalTriangulation(a, b, c, d);
			if (!error)
			{
				cv::Mat A = (cv::Mat_<double>(2, 2) << affinities_original[i].at<double>(0, 0), affinities_original[i].at<double>(0, 1),
					affinities_original[i].at<double>(1, 0), affinities_original[i].at<double>(1, 1));

				// Remove poor quality affine transformations
				double scaleError, angularError, distanceError;
				GetAffineConsistency(Ft, A, c, d, scaleError, angularError, distanceError);

				if (distanceError > 1.0)
					continue;

				// Get optimal affine transformation
				cv::Mat optA;
				GetOptimalAffineTransformation(A, Ft, c, d, optA);
				
				affinities.push_back(optA);
				src_points.push_back(cv::Point2d(c.at<double>(0), c.at<double>(1)));
				dst_points.push_back(cv::Point2d(d.at<double>(0), d.at<double>(1)));
			}
		}
	}

	printf("[Multi-H] %d points kept from the initial %d after filtering.\n", src_points.size(), src_points_original.size());

	if (src_points.size() < 8)
	{
		degenerate_case = true;
		printf("[Multi-H] Degenerate case, not enough points remained.\n");
		return;
	}
}

void MultiH::GetHomographyHAF(double a11, double a12, double a21, double a22,
	double x1, double y1, double x2, double y2,
	cv::Mat &H)
{
	// Coefficient matrix
	cv::Mat A(6, 4, CV_64F);
	double * A_ptr = (double *)A.data;

	// Fill the coefficient matrix
	*(A_ptr++) = a11 * x1 + x2 - epipole_2.at<double>(0);
	*(A_ptr++) = a11 * y1;
	*(A_ptr++) = a11;
	*(A_ptr++) = -*(fundamental_matrix_ptr + 3);

	*(A_ptr++) = a12 * x1;
	*(A_ptr++) = a12 * y1 + x2 - epipole_2.at<double>(0);
	*(A_ptr++) = a12;
	*(A_ptr++) = -*(fundamental_matrix_ptr + 4);

	*(A_ptr++) = a21 * x1 + y2 - epipole_2.at<double>(1);
	*(A_ptr++) = a21 * y1;
	*(A_ptr++) = a21;
	*(A_ptr++) = *(fundamental_matrix_ptr);

	*(A_ptr++) = a22 * x1;
	*(A_ptr++) = a22 * y1 + y2 - epipole_2.at<double>(1);
	*(A_ptr++) = a22;
	*(A_ptr++) = *(fundamental_matrix_ptr + 1);

	*(A_ptr++) = (epipole_2.at<double>(0)*x1 - x2*x1);
	*(A_ptr++) = (epipole_2.at<double>(0)*y1 - x2*y1);
	*(A_ptr++) = (epipole_2.at<double>(0) - x2);
	*(A_ptr++) = ((x1 * *(fundamental_matrix_ptr + 3) + y1 * *(fundamental_matrix_ptr + 4) + *(fundamental_matrix_ptr + 5)));

	*(A_ptr++) = (epipole_2.at<double>(1)*x1 - y2*x1);
	*(A_ptr++) = (epipole_2.at<double>(1)*y1 - y2*y1);
	*(A_ptr++) = (epipole_2.at<double>(1) - y2);
	*(A_ptr++) = (-(x1 * *(fundamental_matrix_ptr) + y1 * *(fundamental_matrix_ptr + 1) + *(fundamental_matrix_ptr + 2)));

	// Calculate the optimal solution (in LS sense) as the eigen std::vector corresponds to the lowest eigen value
	A = A.t() * A;

	cv::Mat EVal, EVec;
	eigen(A, EVal, EVec);

	cv::Mat res = EVec.row(3);

	double *H_ptr = (double *)H.data;

	*(H_ptr + 6) = res.at<double>(0);
	*(H_ptr + 7) = res.at<double>(1);
	*(H_ptr + 8) = res.at<double>(2);
	double lambda = res.at<double>(3);

	*(H_ptr + 3) = epipole_2.at<double>(1) * *(H_ptr + 6) - lambda * *(fundamental_matrix_ptr);
	*(H_ptr + 4) = epipole_2.at<double>(1) * *(H_ptr + 7) - lambda * *(fundamental_matrix_ptr + 1);
	*(H_ptr + 5) = epipole_2.at<double>(1) * *(H_ptr + 8) - lambda * *(fundamental_matrix_ptr + 2);
	*(H_ptr + 0) = epipole_2.at<double>(0) * *(H_ptr + 6) + lambda * *(fundamental_matrix_ptr + 3);
	*(H_ptr + 1) = epipole_2.at<double>(0) * *(H_ptr + 7) + lambda * *(fundamental_matrix_ptr + 4);
	*(H_ptr + 2) = epipole_2.at<double>(0) * *(H_ptr + 8) + lambda * *(fundamental_matrix_ptr + 5);
	H = H / *(H_ptr + 8);
}

void MultiH::GetHomographyHAFNonminimal(const cv::Mat &affines,
	const cv::Mat &pts1,
	const cv::Mat &pts2,
	cv::Mat &H,
	bool do_numerical_refinement)
{
	const int N = affines.rows;

	// Coefficient matrix
	cv::Mat A(N * 6, 4, CV_64F);
	double * A_ptr = (double *)A.data;

	// Fill the coefficient matrix
	for (int i = 0; i < N; ++i)
	{
		const double a11 = affines.at<double>(i, 0);
		const double a12 = affines.at<double>(i, 1);
		const double a21 = affines.at<double>(i, 2);
		const double a22 = affines.at<double>(i, 3);

		const double x1 = pts1.at<double>(i, 0);
		const double y1 = pts1.at<double>(i, 1);
		const double x2 = pts2.at<double>(i, 0);
		const double y2 = pts2.at<double>(i, 1);

		*(A_ptr++) = a11 * x1 + x2 - epipole_2.at<double>(0);
		*(A_ptr++) = a11 * y1;
		*(A_ptr++) = a11;
		*(A_ptr++) = -*(fundamental_matrix_ptr + 3);

		*(A_ptr++) = a12 * x1;
		*(A_ptr++) = a12 * y1 + x2 - epipole_2.at<double>(0);
		*(A_ptr++) = a12;
		*(A_ptr++) = -*(fundamental_matrix_ptr + 4);

		*(A_ptr++) = a21 * x1 + y2 - epipole_2.at<double>(1);
		*(A_ptr++) = a21 * y1;
		*(A_ptr++) = a21;
		*(A_ptr++) = *(fundamental_matrix_ptr);

		*(A_ptr++) = a22 * x1;
		*(A_ptr++) = a22 * y1 + y2 - epipole_2.at<double>(1);
		*(A_ptr++) = a22;
		*(A_ptr++) = *(fundamental_matrix_ptr + 1);

		*(A_ptr++) = (epipole_2.at<double>(0)*x1 - x2*x1);
		*(A_ptr++) = (epipole_2.at<double>(0)*y1 - x2*y1);
		*(A_ptr++) = (epipole_2.at<double>(0) - x2);
		*(A_ptr++) = ((x1 * *(fundamental_matrix_ptr + 3) + y1 * *(fundamental_matrix_ptr + 4) + *(fundamental_matrix_ptr + 5)));

		*(A_ptr++) = (epipole_2.at<double>(1)*x1 - y2*x1);
		*(A_ptr++) = (epipole_2.at<double>(1)*y1 - y2*y1);
		*(A_ptr++) = (epipole_2.at<double>(1) - y2);
		*(A_ptr++) = (-(x1 * *(fundamental_matrix_ptr) + y1 * *(fundamental_matrix_ptr + 1) + *(fundamental_matrix_ptr + 2)));
	}

	// Calculate the optimal solution (in LS sense) as the eigen std::vector corresponds to the lowest eigen value
	A = A.t() * A;

	cv::Mat EVal, EVec;
	eigen(A, EVal, EVec);

	cv::Mat res = EVec.row(3);

	double *H_ptr = (double *)H.data;

	*(H_ptr + 6) = res.at<double>(0);
	*(H_ptr + 7) = res.at<double>(1);
	*(H_ptr + 8) = res.at<double>(2);
	double lambda = res.at<double>(3);

	*(H_ptr + 3) = epipole_2.at<double>(1) * *(H_ptr + 6) - lambda * *(fundamental_matrix_ptr);
	*(H_ptr + 4) = epipole_2.at<double>(1) * *(H_ptr + 7) - lambda * *(fundamental_matrix_ptr + 1);
	*(H_ptr + 5) = epipole_2.at<double>(1) * *(H_ptr + 8) - lambda * *(fundamental_matrix_ptr + 2);
	*(H_ptr + 0) = epipole_2.at<double>(0) * *(H_ptr + 6) + lambda * *(fundamental_matrix_ptr + 3);
	*(H_ptr + 1) = epipole_2.at<double>(0) * *(H_ptr + 7) + lambda * *(fundamental_matrix_ptr + 4);
	*(H_ptr + 2) = epipole_2.at<double>(0) * *(H_ptr + 8) + lambda * *(fundamental_matrix_ptr + 5);

	if (do_numerical_refinement)
		RefineHomographyHAF<double>(pts1, pts2, affines, epipole_2, fundamental_matrix, H);
}

void MultiH::GetHomography3PT(const cv::Mat &pts1,
	const cv::Mat &pts2,
	cv::Mat &H,
	bool do_numerical_refinement)
{
	// Setup the coefficient matrix
	const int N = pts1.rows;

	// Normalize the data
	cv::Mat T1, T2;
	cv::Mat norm_pts1, norm_pts2;
	NormalizePoints<double>(pts1, norm_pts1, T1);
	NormalizePoints<double>(pts2, norm_pts2, T2);

	cv::Mat normalized_fundamental_matrix = T2.inv().t() * fundamental_matrix * T1.inv();
	const double * normalized_fundamental_matrix_ptr = (double *)normalized_fundamental_matrix.data;

	cv::Mat normalized_epipole_2;
	cv::Mat FFt = normalized_fundamental_matrix * normalized_fundamental_matrix.t();
	cv::Mat evalues, evectors;
	eigen(FFt, evalues, evectors);
	normalized_epipole_2 = evectors.row(evectors.rows - 1);
	normalized_epipole_2 = normalized_epipole_2 / normalized_epipole_2.at<double>(2);

	cv::Mat A(2 * N, 3, CV_64F);
	cv::Mat b(2 * N, 1, CV_64F);
	double *A_ptr = (double *)A.data;
	double *b_ptr = (double *)b.data;

	for (unsigned int i = 0; i < N; ++i)
	{
		*(A_ptr++) = normalized_epipole_2.at<double>(0) * norm_pts1.at<double>(i, 0) - norm_pts2.at<double>(i, 0)*norm_pts1.at<double>(i, 0);
		*(A_ptr++) = normalized_epipole_2.at<double>(0) * norm_pts1.at<double>(i, 1) - norm_pts2.at<double>(i, 0)*norm_pts1.at<double>(i, 1);
		*(A_ptr++) = normalized_epipole_2.at<double>(0) - norm_pts2.at<double>(i, 0);

		*(A_ptr++) = normalized_epipole_2.at<double>(1)*norm_pts1.at<double>(i, 0) - norm_pts2.at<double>(i, 1)*norm_pts1.at<double>(i, 0);
		*(A_ptr++) = normalized_epipole_2.at<double>(1)*norm_pts1.at<double>(i, 1) - norm_pts2.at<double>(i, 1)*norm_pts1.at<double>(i, 1);
		*(A_ptr++) = normalized_epipole_2.at<double>(1) - norm_pts2.at<double>(i, 1);

		*(b_ptr++) = -(norm_pts1.at<double>(i, 0) * *(normalized_fundamental_matrix_ptr + 3) + norm_pts1.at<double>(i, 1) * *(normalized_fundamental_matrix_ptr + 4) + *(normalized_fundamental_matrix_ptr + 5));
		*(b_ptr++) = (norm_pts1.at<double>(i, 0) * *(normalized_fundamental_matrix_ptr) +norm_pts1.at<double>(i, 1) * *(normalized_fundamental_matrix_ptr + 1) + *(normalized_fundamental_matrix_ptr + 2));
	}

	cv::Mat res = A.inv(cv::DECOMP_SVD) * b;

	double *H_ptr = (double *)H.data;
	*(H_ptr + 6) = res.at<double>(0);
	*(H_ptr + 7) = res.at<double>(1);
	*(H_ptr + 8) = res.at<double>(2);

	*(H_ptr + 3) = normalized_epipole_2.at<double>(1) * *(H_ptr + 6) - *(normalized_fundamental_matrix_ptr);
	*(H_ptr + 4) = normalized_epipole_2.at<double>(1) * *(H_ptr + 7) - *(normalized_fundamental_matrix_ptr + 1);
	*(H_ptr + 5) = normalized_epipole_2.at<double>(1) * *(H_ptr + 8) - *(normalized_fundamental_matrix_ptr + 2);
	*(H_ptr + 0) = normalized_epipole_2.at<double>(0) * *(H_ptr + 6) + *(normalized_fundamental_matrix_ptr + 3);
	*(H_ptr + 1) = normalized_epipole_2.at<double>(0) * *(H_ptr + 7) + *(normalized_fundamental_matrix_ptr + 4);
	*(H_ptr + 2) = normalized_epipole_2.at<double>(0) * *(H_ptr + 8) + *(normalized_fundamental_matrix_ptr + 5);

	if (do_numerical_refinement)
		RefineHomography3PT<double>(norm_pts1, norm_pts2, normalized_epipole_2, normalized_fundamental_matrix, H);
	H = T2.inv() * H * T1;
}

void MultiH::GetAffineConsistency(cv::Mat fundamental_matrix_transposed, cv::Mat A, cv::Mat pt1, cv::Mat pt2, double &scaleError, double &angularError, double &distanceError)
{
	// Get the epipolar lines
	cv::Mat l1 = fundamental_matrix_transposed * pt2;
	cv::Mat l2 = fundamental_matrix * pt1;

	l1 = l1 / l1.at<double>(2);
	l2 = l2 / l2.at<double>(2);

	// Get the normal directions
	cv::Mat n1 = (cv::Mat_<double>(3, 1) << l1.at<double>(0), l1.at<double>(1), 0);
	cv::Mat n2 = (cv::Mat_<double>(3, 1) << l2.at<double>(0), l2.at<double>(1), 0);
	n1 = n1 / norm(n1);
	n2 = n2 / norm(n2);

	// Compute optimal scale
	double beta = GetBetaScale(fundamental_matrix_transposed, A, pt1, pt2);

	// Transform the normals with the affine transformation
	cv::Mat r1 = A.inv().t() * (cv::Mat_<double>(2, 1) << n1.at<double>(0), n1.at<double>(1));
	cv::Mat r2 = beta * (cv::Mat_<double>(2, 1) << n2.at<double>(0), n2.at<double>(1));

	// Get the scale difference
	scaleError = norm(r1) - beta;

	// Get the difference of the resulting std::vectors
	distanceError = norm(r1 - r2);

	// Get the angular error
	r1 = r1 / norm(r1);
	r2 = r2 / norm(r2);

	angularError = acos(r1.dot(r2));
}

double MultiH::GetBetaScale(cv::Mat fundamental_matrix_transposed, cv::Mat A, cv::Mat pt1, cv::Mat pt2)
{
	// Compute the epipolar lines
	cv::Mat l1 = fundamental_matrix_transposed * pt2;
	cv::Mat l2 = fundamental_matrix * pt1;

	double x_new1 = pt1.at<double>(0) + static_cast<double>(1.0);
	double y_new1 = -(l1.at<double>(0) * x_new1 + l1.at<double>(2)) / l1.at<double>(1);

	cv::Mat dx1 = (cv::Mat_<double>(3, 1) << x_new1, y_new1, 1) - pt1;

	double x_new2 = pt2.at<double>(0) + 1.0;
	double y_new2 = -(l2.at<double>(0) * x_new2 + l2.at<double>(2)) / l2.at<double>(1);
	cv::Mat dx2 = (cv::Mat_<double>(3, 1) << x_new2, y_new2, 1) - pt2;

	dx1 = dx1 / norm(dx1);
	dx2 = dx2 / norm(dx2);

	double beta = abs(sqrt(l2.at<double>(0) * l2.at<double>(0) + l2.at<double>(1) * l2.at<double>(1)) /
		((-*(fundamental_matrix_ptr) * dx1.at<double>(1) + *(fundamental_matrix_ptr + 1) * dx1.at<double>(0)) * pt2.at<double>(0) +
		(-*(fundamental_matrix_ptr + 3) * dx1.at<double>(1) + *(fundamental_matrix_ptr + 4) * dx1.at<double>(0)) * pt2.at<double>(1) - *(fundamental_matrix_ptr + 6)*dx1.at<double>(1) + *(fundamental_matrix_ptr + 7)*dx1.at<double>(0)));
	return beta;
}

bool MultiH::OptimalTriangulation(cv::Mat pt1, cv::Mat pt2, cv::Mat &pt1Out, cv::Mat &pt2Out)
{
	cv::Mat T1 = (cv::Mat_<double>(3, 3, CV_64F) << 1, 0, -pt1.at<double>(0), 0, 1, -pt1.at<double>(1), 0, 0, 1);
	cv::Mat T2 = (cv::Mat_<double>(3, 3, CV_64F) << 1, 0, -pt2.at<double>(0), 0, 1, -pt2.at<double>(1), 0, 0, 1);
	cv::Mat F2 = T2.t().inv() * fundamental_matrix * T1.inv();

	double s1 = epipole_1.at<double>(0) * epipole_1.at<double>(0) + epipole_1.at<double>(1) * epipole_1.at<double>(1);
	double s2 = epipole_2.at<double>(0) * epipole_2.at<double>(0) + epipole_2.at<double>(1) * epipole_2.at<double>(1);

	cv::Mat F3 = R2 * F2 * R1t;

	double f1 = epipole_1.at<double>(2);
	double f2 = epipole_2.at<double>(2);

	double a = F3.at<double>(1, 1);
	double b = F3.at<double>(1, 2);
	double c = F3.at<double>(2, 1);
	double d = F3.at<double>(2, 2);

	// Create polinomial :
	double t6 = -a*c*(f1*f1*f1*f1)*(a*d - b*c);
	double t5 = (a*a + f2*f2*c*c)*(a*a + f2*f2*c*c) - (a*d + b*c)*(f1*f1*f1*f1)*(a*d - b*c);
	double t4 = 2 * (a*a + f2*f2*c*c)*(2 * a*b + 2 * c*d*f2*f2) - d*b*(f1*f1*f1*f1)*(a*d - b*c) - 2 * a*c*f1*f1*(a*d - b*c);
	double t3 = (2 * a*b + 2 * c*d*f2*f2)*(2 * a*b + 2 * c*d*f2*f2) + 2 * (a*a + f2*f2*c*c)*(b*b + f2*f2*d*d) - 2 * f1*f1*(a*d - b*c)*(a*d + b*c);
	double t2 = 2 * (2 * a*b + 2 * c*d*f2*f2)*(b*b + f2*f2*d*d) - 2 * (f1*f1*a*d - f1*f1*b*c)*b*d - a*c*(a*d - b*c);
	double t1 = (b*b + f2*f2*d*d)*(b*b + f2*f2*d*d) - (a*d + b*c)*(a*d - b*c);
	double t0 = -(a*d - b*c)*b*d;

	cv::Mat coeffs = (cv::Mat_<double>(1, 7, CV_64F) << t0, t1, t2, t3, t4, t5, t6);
	cv::Mat roots;

	double bestS = static_cast<double>(INT_MAX);
	double bestT = 0;

	solvePoly(coeffs, roots);

	for (unsigned short i = 0; i < roots.rows; ++i)
	{
		double val;

		if (abs(roots.at<double>(i, 1)) <= 1e-10) // Is real
		{
			double t = roots.at<double>(i, 0);
			val = t*t / (1 + f1*f1*t*t) + ((c*t + d) * (c*t + d)) / ((a*t + b) * (a*t + b) + f2*f2*((c*t + d) * (c*t + d)));

			if (val < bestS)
			{
				bestS = val;
				bestT = t;
			}
		}
	}

	double valInf = 1 / (f1*f1) + (c*c) / (a*a + f2*f2 * c*c);
	if (valInf < bestS)
	{
		pt1Out = pt1;
		pt2Out = pt2;
		return false;
	}

	cv::Mat point1 = (cv::Mat_<double>(3, 1, CV_64F) << 0, bestT, 1);
	cv::Mat line2 = F3*point1;
	cv::Mat point2 = (cv::Mat_<double>(3, 1, CV_64F) << -line2.at<double>(0)*line2.at<double>(2), -line2.at<double>(1)*line2.at<double>(2), line2.at<double>(0)* line2.at<double>(0) + line2.at<double>(1) * line2.at<double>(1));
	point2 = point2 * (1.0 / point2.at<double>(2));

	cv::Mat u2 = (R1*T1).inv() * point1;
	cv::Mat v2 = (R2*T2).inv() * point2;

	pt1Out = u2;
	pt2Out = v2;
	return true;
}

void MultiH::GetOptimalAffineTransformation(cv::Mat A, cv::Mat fundamental_matrix_transpose, cv::Mat pt1, cv::Mat pt2, cv::Mat &optimalA)
{
	// Compute the epipolar lines
	cv::Mat l1 = fundamental_matrix_transpose * pt2;
	cv::Mat l2 = fundamental_matrix * pt1;

	l1 = l1 / l1.at<double>(2);
	l2 = l2 / l2.at<double>(2);

	cv::Mat n1 = (cv::Mat_<double>(2, 1) << l1.at<double>(0), l1.at<double>(1));
	cv::Mat n2 = (cv::Mat_<double>(2, 1) << l2.at<double>(0), l2.at<double>(1));
	n1 = n1 / norm(n1);
	n2 = n2 / norm(n2);

	// Calculation of scale beta
	double beta = GetBetaScale(fundamental_matrix_transpose, A, pt1, pt2);

	if (n1.dot(n2) < 0)
		n2 = -n2;

	// Computation of the optimal affine transformation
	cv::Mat C = (cv::Mat_<double>(6, 6) << 1, 0, 0, 0, -beta * n2.at<double>(0), 0,
		0, 1, 0, 0, 0, -beta * n2.at<double>(0),
		0, 0, 1, 0, -beta * n2.at<double>(1), 0,
		0, 0, 0, 1, 0, -beta * n2.at<double>(1),
		-beta * n2.at<double>(0), 0, -beta * n2.at<double>(1), 0, 0, 0,
		0, -beta * n2.at<double>(0), 0, -beta * n2.at<double>(1), 0, 0);

	cv::Mat b = (cv::Mat_<double>(6, 1) << A.at<double>(0, 0), A.at<double>(0, 1), A.at<double>(1, 0), A.at<double>(1, 1), -n1.at<double>(0), -n1.at<double>(1));
	cv::Mat x = C.inv() * b;

	optimalA = (cv::Mat_<double>(2, 2) << x.at<double>(0), x.at<double>(1),
		x.at<double>(2), x.at<double>(3));
}
