#pragma once

#include "cv.h"
#include "ppl.h"
#include <iostream>
#include <opencv2/features2d/features2d.hpp>

class MedianShift
{
public:
	enum MODE_TYPE { MEAN, TUKEY_MEDIAN, WEISZFELD_MEDIAN };

	MedianShift();
	~MedianShift();

	void Initialize(cv::Mat const * _points, bool _estimate_bandwidth = true, int _knn_param = 10, int _map_precision = 10000, bool _do_assign_points = false, bool _adaptive = true, MODE_TYPE _mode_type = MODE_TYPE::WEISZFELD_MEDIAN, bool _use_anisotropic_scale = false);
	void Run(int _max_iter_number = 50, float _band_width = -1.0f);

	int GetClusterNumber() { return clusters.rows; }
	void GetClusters(cv::Mat &_clusters) { _clusters = clusters; }
	float GetBandWidth() { return band_width; }
	float GetMeanDistance() { return mean_distance; }
	const int * const GetTukeyDephts() { return tukey_depths; };

	void GetClusterPoints(std::vector<std::vector<int>> &_cluster_points) { _cluster_points = cluster_points; }
	int GetClusterCardinality(int idx) { return cluster_points[idx].size(); }

protected:
	int *tukey_depths;
	bool use_anisotropic_scale;
	MODE_TYPE mode_type;
	float mean_distance;
	float epsilon;
	cv::Mat const * points;
	cv::Mat clusters, temp_clusters;
	int max_iter_number;
	int knn_param;
	std::vector<std::vector<cv::DMatch>> neighbours;
	std::vector<int> map_to_cluster;
	std::vector<std::vector<int>> cluster_points;
	float average_distance;
	float band_width;
	int dimension;
	cv::Mat hyperplanes;
	int map_precision;
	bool do_assign_points;
	bool adaptive;
	std::vector<float> bandwidths;

	void Step();
	void Mean(std::vector<cv::DMatch> const &neighbours, cv::Mat &mean);
	void TukeyMedian(std::vector<cv::DMatch> const &neighbours, cv::Mat &median, int &median_depth);
	void WeiszfeldMedian(std::vector<cv::DMatch> const &neighbours, cv::Mat &median, int iterations);
	void Unique(const cv::Mat& input, std::vector<int> &output, bool sort = false);
};

