#pragma once

#include <thread>
#include <cv.h>

template<typename T>
class MeanShiftClustering
{
public:
	MeanShiftClustering() { workersNumber = std::thread::hardware_concurrency(); }
	~MeanShiftClustering() {}

	void Cluster(cv::InputArray _data, T _bandWidth, cv::OutputArray _clusters, std::vector<std::vector<int>> &_clusterPoints);
	
protected:
	int workersNumber;
	
	void GetMinMaxPos(cv::Mat const &_data, cv::Mat &minPos, cv::Mat &maxPos);
	void repmat(cv::Mat const &_data, int num, cv::Mat &result);
};

template<typename T>
void MeanShiftClustering<T>::Cluster(cv::InputArray _data, T _bandWidth, cv::OutputArray _clusters, std::vector<std::vector<int>> &_clusterPoints)
{
	cv::Mat data = _data.getMat();

	int numDim = data.cols;
	int numPts = data.rows;

	int numClust = 0;
	T bandSq = _bandWidth * _bandWidth;

	std::vector<int> initPtInds(numPts);
	int *beenVisitedFlag = new int[numPts];
	std::vector<cv::Mat> clusterVotes;

	concurrency::parallel_for(0, numPts, [&](int i)
	{
		initPtInds[i] = i;
		beenVisitedFlag[i] = 0;
	});

	cv::Mat minPos, maxPos;
	GetMinMaxPos(data, minPos, maxPos);
	
	cv::Mat boundBox = maxPos - minPos;
	T sizeSpace = norm(boundBox);
	T stopThresh = 1e-3*_bandWidth;

	std::vector<cv::Mat> clustCent;

	while (initPtInds.size())
	{
		T rnd = rand() / static_cast<T>(RAND_MAX);
		int tempInd = round(rnd * (initPtInds.size() - 1));
		int stInd = initPtInds[tempInd];

		cv::Mat myMean = data.row(stInd).clone();
		std::vector<int> myMembers;
		cv::Mat thisClusterVotes = cv::Mat_<int>(1, numPts, 0);

		while (1)
		{
			// dist squared from mean to all points still active
			cv::Mat rMat;
			repmat(myMean, numPts, rMat);

			rMat = rMat - data;

			cv::Mat myOldMean = myMean.clone();
			cv::Mat sqDistToAll = cv::Mat::zeros(numPts, 1, data.type());
			std::vector<int> inInds;
			inInds.reserve(numPts);

			myMean = cv::Mat::zeros(1, data.cols, data.type());
			for (int i = 0; i < numPts; ++i)
			{
				for (int j = 0; j < rMat.cols; ++j)
				{
					double dist = rMat.at<T>(i, j) * rMat.at<T>(i, j);
					dist = sqrt(dist);
					sqDistToAll.at<T>(i) += dist;
				}

				if (sqDistToAll.at<T>(i) < bandSq)
				{
					++thisClusterVotes.at<int>(i);
					inInds.push_back(i);

					myMean = myMean + data.row(i);
					beenVisitedFlag[i] = 1;
					myMembers.push_back(i);
				}
			}

			myMean = myMean / inInds.size();
			
			if (norm(myMean - myOldMean) < stopThresh)
			{
				int mergeWith = -1;
				for (int cn = 0; cn < clustCent.size(); ++cn)
				{
					T distToOther = norm(myMean - clustCent[cn]);
					if (distToOther < _bandWidth / 2)
					{
						mergeWith = cn;
						break;
					}
				}

				if (mergeWith > -1)
				{
					clustCent[mergeWith] = 0.5 * (clustCent[mergeWith] + myMean);
					clusterVotes[mergeWith] = clusterVotes[mergeWith] + thisClusterVotes;
				}
				else
				{
					clustCent.push_back(myMean);
					clusterVotes.push_back(thisClusterVotes);
				}
				break;
			}
		}

		initPtInds.resize(0);
		for (int i = 0; i < numPts; ++i)
		{
			if (beenVisitedFlag[i] == 0)
				initPtInds.push_back(i);
		}
	}

	std::vector<int> finalClusterVotes(numPts, 0);
	std::vector<int> finalClusterIdx(numPts, -1);

	for (int r = 0; r < clusterVotes.size(); ++r)
	{
		for (int i = 0; i < numPts; ++i)
		{
			if (finalClusterVotes[i] < clusterVotes[r].at<int>(i))
			{
				finalClusterVotes[i] = clusterVotes[r].at<int>(i);
				finalClusterIdx[i] = r;
			}
		}
	}

	cv::Mat resultingCluster = cv::Mat_<T>(clusterVotes.size(), data.cols);

	_clusterPoints.resize(clusterVotes.size());
	for (int i = 0; i < finalClusterIdx.size(); ++i)
		_clusterPoints[finalClusterIdx[i]].push_back(i);
	for (int i = 0; i < clustCent.size(); ++i)
		clustCent[i].copyTo(resultingCluster.row(i));
	
	_clusters.getMatRef() = resultingCluster;
}

template<typename T>
void MeanShiftClustering<T>::repmat(cv::Mat const &_data, int num, cv::Mat &result)
{
	if (_data.cols == 1)
	{
		result = cv::Mat_<T>(_data.rows, num);
		concurrency::parallel_for(0, num, [&](int i)
		{
			for (int j = 0; j < result.rows; ++j)
				result.at<T>(j, i) = _data.at<T>(j);
		});
	}
	else if (_data.rows == 1)
	{
		result = cv::Mat_<T>(num, _data.cols);
		concurrency::parallel_for(0, num, [&](int i)
		{
			for (int j = 0; j < result.cols; ++j)
				result.at<T>(i, j) = _data.at<T>(j);
		});
	}

}

template<typename T>
void MeanShiftClustering<T>::GetMinMaxPos(cv::Mat const &_data, cv::Mat &minPos, cv::Mat &maxPos)
{
	minPos = cv::Mat_<T>(1, _data.cols);
	maxPos = cv::Mat_<T>(1, _data.cols);
	for (int i = 0; i < _data.rows; ++i)
	{
		for (int dim = 0; dim < _data.cols; ++dim)
		{
			if (i == 0 || minPos.at<T>(dim) > _data.at<T>(i, dim))
				minPos.at<T>(dim) = _data.at<T>(i, dim);
			if (i == 0 || maxPos.at<T>(dim) < _data.at<T>(i, dim))
				maxPos.at<T>(dim) = _data.at<T>(i, dim);
		}
	}
}

