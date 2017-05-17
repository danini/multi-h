#pragma once

#include "cv.h"
#include "ppl.h"
#include <iostream>
#include <vector>
#include <opencv2/features2d/features2d.hpp>

using namespace cv;
using namespace std;
using namespace concurrency;

class MedianShift
{
public:
	enum MODE_TYPE {MEAN, TUKEY_MEDIAN, WEISZFELD_MEDIAN};

	MedianShift();
	~MedianShift();

	void Initialize(Mat const * _points, bool _estimate_bandwidth = true, int _knn_param = 10, int _map_precision = 10000, bool _do_assign_points = false, bool _adaptive = true, MODE_TYPE _mode_type = MODE_TYPE::WEISZFELD_MEDIAN, bool _use_anisotropic_scale = false);
	void Run(int _max_iter_number = 50, float _band_width = -1.0f);

	int GetClusterNumber() { return clusters.rows; }
	vector<vector<int>> *GetClusterPoints() { return &cluster_points; }
	void GetClusters(Mat &_clusters) { _clusters = clusters; }
	float GetBandWidth() { return band_width; }
	float GetMeanDistance() { return mean_distance; }
	const int * const GetTukeyDephts() { return tukey_depths; };

	int GetClusterCardinality(int idx) { return adaptive ? cluster_points[idx].size() : -1; }

protected:
	int *tukey_depths;
	bool use_anisotropic_scale;
	MODE_TYPE mode_type;
	float mean_distance;
	float epsilon;
	Mat const * points;
	Mat clusters, temp_clusters;
	int max_iter_number;
	int knn_param;
	vector<vector<DMatch>> neighbours;
	vector<int> map_to_cluster;
	vector<vector<int>> cluster_points;
	float average_distance;
	float band_width;
	int dimension;
	Mat hyperplanes;
	int map_precision;
	bool do_assign_points;
	bool adaptive;
	vector<float> bandwidths;

	void Step(float &change);
	void Mean(vector<DMatch> const &neighbours, Mat &mean);
	void TukeyMedian(vector<DMatch> const &neighbours, Mat &median, int &median_depth);
	void WeiszfeldMedian(vector<DMatch> const &neighbours, Mat &median, int iterations);
	void Unique(const cv::Mat& input, std::vector<int> &output, std::vector<int> &cardinalities, bool sort = false);
};

