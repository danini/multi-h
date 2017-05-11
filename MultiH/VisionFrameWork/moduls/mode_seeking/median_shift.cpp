#include "median_shift.h"
#include <map>

MedianShift::MedianShift() : epsilon(1e-5),
	tukey_depths(NULL)
{
}


MedianShift::~MedianShift()
{
	delete tukey_depths;
}

void MedianShift::Initialize(cv::Mat const * _points, bool _estimate_bandwidth, int _knn_param, int _map_precision, bool _do_assign_points, bool _adaptive, MODE_TYPE _mode_type, bool _use_anisotropic_scale)
{
	if (_points->rows == 0)
		return;

	use_anisotropic_scale = _use_anisotropic_scale;
	points = _points;
	clusters = points->clone();
	temp_clusters = cv::Mat(clusters.size(), clusters.type());
	knn_param = _knn_param;
	map_precision = _map_precision;
	dimension = clusters.cols;
	adaptive = _adaptive;
	mode_type = _mode_type;
	map_to_cluster.resize(_points->rows);

	hyperplanes = cv::Mat::eye(dimension, dimension, points->type());

	do_assign_points = _do_assign_points;

	if (_estimate_bandwidth)
	{
		cv::FlannBasedMatcher flann;
		flann.knnMatch(clusters, clusters, neighbours, knn_param);

		average_distance = 0;
		for (int i = 0; i < neighbours.size(); ++i)
		{
			float dist = 0;
			for (int j = 0; j < knn_param; ++j)
				dist += neighbours[i][j].distance;
			average_distance += dist / knn_param;
		}

		average_distance = 0.1f * average_distance / neighbours.size();
	}

	if (adaptive)
	{
		if (clusters.rows < knn_param)
		{
			bandwidths.resize(clusters.rows, 0);
			band_width = 0;
			return;
		}

		//std::cout << clusters << endl;
		cv::FlannBasedMatcher flann;
		flann.knnMatch(clusters, clusters, neighbours, knn_param);

		bandwidths.resize(clusters.rows, 0);

		std::vector<float> distances(neighbours.size());
		mean_distance = 0;
		float maxDistance = 0;
		for (int i = 0; i < neighbours.size(); ++i)
		{
			float dist = neighbours[i][knn_param - 1].distance;
			bandwidths[i] = dist;
			maxDistance = MAX(maxDistance, dist);
			distances[i] = dist;
		}
		std::sort(distances.begin(), distances.end());
		mean_distance = distances[round(distances.size() / 2)];
		band_width = maxDistance;
	}
}

void MedianShift::Run(int _max_iter_number, float _band_width)
{
	if (clusters.rows < knn_param)
	{
		if (do_assign_points)
		{
			cluster_points.resize(clusters.rows);
			for (int i = 0; i < points->rows; ++i)
				cluster_points[i].push_back(i);
		}
		return;
	}

	max_iter_number = _max_iter_number;

	if (_band_width < 0)
	{
		if (!adaptive)
			band_width = average_distance;
	}
	else
		band_width = _band_width; 

	int last_cluster_number = 0;

	std::vector<int> last_cluster_indices;
	for (int it = 0; it < max_iter_number; ++it)
	{
		Step();

		if (clusters.rows <= 1)
			break;

		std::vector<int> unique_clusters;
		Unique(clusters, unique_clusters);
		if (last_cluster_number == unique_clusters.size())
			break;

		last_cluster_number = unique_clusters.size();

		if (do_assign_points)
		{
			if (it == 0)
			{
				for (int i = 0; i < clusters.rows; ++i)
				{
					for (int k = 0; k < unique_clusters.size(); ++k)
					{
						int idx = unique_clusters[k];
						if (i == idx || norm(clusters.row(i) - clusters.row(idx)) < 1e-6)
						{
							map_to_cluster[i] = k;
							break;
						}
					}
				}
			}
			else
			{
				std::vector<int> localMap(clusters.rows);
				for (int i = 0; i < clusters.rows; ++i)
				{
					for (int k = 0; k < unique_clusters.size(); ++k)
					{
						int idx = unique_clusters[k];
						if (i == idx || norm(clusters.row(i) - clusters.row(idx)) < 1e-6)
						{
							localMap[i] = k;
							break;
						}
					}
				}

				for (int i = 0; i < map_to_cluster.size(); ++i)
				{
					int idx1 = map_to_cluster[i];
					map_to_cluster[i] = localMap[idx1];
				}
			}

			last_cluster_indices = unique_clusters;
		}

		temp_clusters = cv::Mat(unique_clusters.size(), dimension, points->type());

		for (int i = 0; i < unique_clusters.size(); ++i)
			clusters.row(unique_clusters[i]).copyTo(temp_clusters.row(i));
		clusters.release();
		clusters = temp_clusters.clone();
	}

	if (do_assign_points)
	{
		cluster_points.resize(clusters.rows);
		for (int i = 0; i < points->rows; ++i)
			cluster_points[map_to_cluster[i]].push_back(i);
	}
}

void MedianShift::Step()
{
	if (mode_type == MODE_TYPE::TUKEY_MEDIAN)
	{
		if (tukey_depths != NULL)
			delete tukey_depths;
		tukey_depths = new int[clusters.rows];

		if (clusters.rows == 1)
			tukey_depths[0] = 0;
	}

	if (clusters.rows < 2)
		return;

	std::vector<std::vector<cv::DMatch>> localNeighbours;

	cv::FlannBasedMatcher flann;
	flann.radiusMatch(clusters, clusters, localNeighbours, band_width); 
	
	if (localNeighbours.size() == 0)
		return;

	concurrency::parallel_for(0, clusters.rows, [&](int i)
	//for (int i = 0; i < clusters.rows; ++i)
	{
		int ptNumber = localNeighbours[i].size();
		if (ptNumber == 0)
			return;
		
		if (ptNumber == 1)
			clusters.row(i).copyTo(temp_clusters.row(i));
		else if (mode_type == MODE_TYPE::MEAN)
			Mean(localNeighbours[i], temp_clusters.row(i));
		else if (mode_type == MODE_TYPE::WEISZFELD_MEDIAN)
			WeiszfeldMedian(localNeighbours[i], temp_clusters.row(i), 10);
		else if (mode_type == MODE_TYPE::TUKEY_MEDIAN)
		{
			if (ptNumber <= dimension)
			{
				Mean(localNeighbours[i], temp_clusters.row(i));
				tukey_depths[i] = ptNumber;
			} 
			else
				TukeyMedian(localNeighbours[i], temp_clusters.row(i), tukey_depths[i]);
		}
	});

	clusters.release();
	clusters = temp_clusters.clone();
}

void MedianShift::Mean(std::vector<cv::DMatch> const &neighbours, cv::Mat &mean)
{
	clusters.row(neighbours[0].trainIdx).copyTo(mean);
	int count = 1;
	for (int i = 1; i < neighbours.size(); ++i)
	{
		int idx = neighbours[i].trainIdx;
		if (adaptive && neighbours[i].distance > bandwidths[idx])
			break;
		mean = mean + clusters.row(idx);
		++count;
	}
	mean = mean / count;
}

void MedianShift::WeiszfeldMedian(std::vector<cv::DMatch> const &neighbours, cv::Mat &median, int iterations)
{
	median = cv::Mat::zeros(1, dimension, clusters.type());
	
	for (int i = 0; i < iterations; i++) 
	{
		cv::Mat s1 = median.clone();
		float s2 = 0;

		for (int i = 0; i < neighbours.size(); ++i)
		{
			if (adaptive && neighbours[i].distance > bandwidths[i])
				break;

			cv::Mat point = clusters.row(neighbours[i].trainIdx);

			float distance = norm(point - median);

			if (distance < epsilon)
				continue;

			s1 += point / distance;
			s2 += 1.0 / distance;
		}

		if (s2 > epsilon)
			median = s1 / s2;
		else
		{
			Mean(neighbours, median);
			break;
		}
	}
}

void MedianShift::TukeyMedian(std::vector<cv::DMatch> const &neighbours, cv::Mat &median, int &tukey_depth)
{
	int N = neighbours.size();
	int *depths = new int[N];
	for (int i = 0; i < N; ++i)
		depths[i] = 0;

	for (int i = 0; i < N; ++i)
	{
		int idx = neighbours[i].trainIdx;
		if (adaptive && neighbours[i].distance > bandwidths[idx])
			break;

		cv::Mat point = clusters.row(idx);

		for (int j = 0; j < hyperplanes.rows; ++j)
		{
			float w = 0;
			for (int k = 0; k < dimension; ++k)
				w = w - hyperplanes.at<float>(j, k) * point.at<float>(k);
			
			int c1 = 0, c2 = 0;
			std::vector<int> pts1, pts2;
			pts1.reserve(N);
			pts2.reserve(N);

			for (int k = 0; k < N; ++k)
			{
				float dist = clusters.row(k).dot(hyperplanes.row(j)) + w;

				if (dist < 0)
				{
					pts1.push_back(k);
					c1 = c1 + 1;
				} 
				else
				{
					pts2.push_back(k);
					c2 = c2 + 1;
				}
			}

			if (c1 < c2)
				for (int k = 0; k < pts1.size(); ++k)
					++depths[pts1[k]];
			else
				for (int k = 0; k < pts2.size(); ++k)
					++depths[pts2[k]];
		}
	}

	int min = depths[0];
	int minIdx = 0;
	for (int i = 1; i < N; ++i)
		if (depths[i] < min)
		{
			min = depths[i];
			minIdx = i;
		}

	tukey_depth = min;
	//std::cout << min << endl;
	clusters.row(neighbours[minIdx].trainIdx).copyTo(median);
	delete depths;
}

void MedianShift::Unique(const cv::Mat& input, std::vector<int> &output, bool sort)
{
	if (input.channels() > 1 || input.type() != CV_32F)
	{
		std::cerr << "unique !!! Only works with CV_32F 1-channel cv::Mat" << std::endl;
		output.resize(0);
	}

	output.reserve(input.rows);
	std::map<std::string, std::vector<int>> elements;
	int clusterNumber = 0;
	for (int y = 0; y < input.rows; ++y)
	{
		const cv::Mat &row = input.row(y);
		std::string key = "";

		for (int i = 0; i < dimension; ++i)
			key = key + std::to_string((int)roundf(map_precision * row.at<float>(i))) + ",";
		
		std::map<std::string, std::vector<int>>::iterator it = elements.find(key);
		if (it == elements.end())
		{
			++clusterNumber;
			elements[key] = std::vector<int>(1, y);
			output.push_back(y);
		} 
		else
			elements[key].push_back(y);
	}
}