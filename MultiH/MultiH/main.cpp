// VisionFrameWork.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

#include <fstream>
#include <vector>
#include <cv.h>
#include <opencv2\highgui\highgui.hpp>
#include <chrono>
#include <direct.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "moduls/feature_detection/EllipticASiftDetector.h"
#include "moduls/feature_detection/EllipticKeyPoint.h"
#include "MultiH.h"

struct stat info;

void ApplyMultiH(std::string srcPath, 
	std::string dstPath, 
	std::string out_corrPath, 
	std::string in_corrPath, 
	std::string output_srcImagePath, 
	std::string output_dstImagePath,
	int minimum_inlier_number,
	double fundamental_threshold,
	double homography_threshold,
	double locality,
	double spatial_weight);
bool LoadPointsFromFile(std::vector<cv::Point2d> &srcPoints, std::vector<cv::Point2d> &dstPoints, std::vector<cv::Mat> &affinesTmp, const char* file);
bool SavePointsToFile(std::vector<cv::Point2d> &srcPoints, std::vector<cv::Point2d> &dstPoints, std::vector<cv::Mat> &affinesTmp, const char* file);
bool SavePointsToFile(std::vector<cv::Point2d> &srcPoints, std::vector<cv::Point2d> &dstPoints, std::vector<cv::Mat> &affines, std::vector<int> &labels, const char* file);
void DoASIFT(cv::Mat &F, cv::Mat &img1, cv::Mat &img2, std::vector<cv::Point2d> &srcPoints, std::vector<cv::Point2d> &dstPoints, std::vector<cv::Mat> &affines, std::string method);
void PrintUsage();
bool GetTaskName(int idx, std::string &task);

int main(int argc, const char* argv[])
{
	if (argc == 1)
	{
		PrintUsage();
		return -1;
	}
	
	int use_case = atoi(argv[1]);

	std::string task;
	std::string srcImagePath, dstImagePath;
	std::string input_correspondence_path, output_correspondence_path;
	std::string output_srcImagePath, output_dstImagePath;
	bool error = false;

	int minimum_inlier_number = 20;
	double fundamental_threshold = 2.6;
	double homography_threshold = 2.2;
	double locality = 0.005;
	double spatial_weight = 0.5;
	
	if (use_case == 1 && argc == 3) // Using a built-in test with standard parameterization
	{
		int task_idx = atoi(argv[2]);
		error = !GetTaskName(task_idx, task);

		if (error)
		{
			std::cerr << "Not valid task number!\n";
			return -1;
		}

		srcImagePath = "data/" + task + "/" + task + "1.png";
		dstImagePath = "data/" + task + "/" + task + "2.png";
		input_correspondence_path = "results/" + task + "/" + task + "_points_with_no_annotation.txt";
		output_correspondence_path = "results/" + task + "/result_" + task + ".txt";
		output_srcImagePath = "results/" + task + "/out_" + task + "1.png";
		output_dstImagePath = "results/" + task + "/out_" + task + "2.png";
	}
	else if (use_case == 2 && argc == 8) // Using a built-in test with custom parameterization
	{
		int task_idx = atoi(argv[2]);
		error = !GetTaskName(task_idx, task);

		if (error)
		{
			std::cerr << "Not valid task number!\n";
			return -1;
		}

		srcImagePath = "data/" + task + "/" + task + "1.png";
		dstImagePath = "data/" + task + "/" + task + "2.png";
		input_correspondence_path = "results/" + task + "/" + task + "_points_with_no_annotation.txt";
		output_correspondence_path = "results/" + task + "/result_" + task + ".txt";
		output_srcImagePath = "results/" + task + "/out_" + task + "1.png";
		output_dstImagePath = "results/" + task + "/out_" + task + "2.png";

		fundamental_threshold = atof(argv[3]);
		homography_threshold = atof(argv[4]);
		locality = atof(argv[5]);
		spatial_weight = atof(argv[6]);
		minimum_inlier_number = atoi(argv[7]);
	}
	else if (use_case == 3 && argc == 5)
	{
		task = argv[4];

		srcImagePath = argv[2];
		dstImagePath = argv[3];
		input_correspondence_path = "results/" + task + "/" + task + "_points_with_no_annotation.txt";
		output_correspondence_path = "results/" + task + "/result_" + task + ".txt";
		output_srcImagePath = "results/" + task + "/out_" + task + "1.png";
		output_dstImagePath = "results/" + task + "/out_" + task + "2.png";
	}
	else if (use_case == 4 && argc == 10)
	{
		task = argv[4];

		srcImagePath = argv[2];
		dstImagePath = argv[3];
		input_correspondence_path = "results/" + task + "/" + task + "_points_with_no_annotation.txt";
		output_correspondence_path = "results/" + task + "/result_" + task + ".txt";
		output_srcImagePath = "results/" + task + "/out_" + task + "1.png";
		output_dstImagePath = "results/" + task + "/out_" + task + "2.png";

		fundamental_threshold = atof(argv[5]);
		homography_threshold = atof(argv[6]);
		locality = atof(argv[7]);
		spatial_weight = atof(argv[8]);
		minimum_inlier_number = atoi(argv[9]);
	}
	else
	{
		std::cerr << "Invalid parameterization!\n";
		PrintUsage();
		return -1;
	}
	
	// Create the task directory of doesn't exist
	std::string dir = "results/" + task;
	
	if (stat(dir.c_str(), &info) != 0)
		if (_mkdir(dir.c_str()) != 0)
		{
			std::cerr << "Error while creating a new folder in 'results'\n";
			return -1;
		}

	// Print the used parameters
	printf("Used parameters:\n\tFundamental matrix threshold = %f px\n\tHomography threshold = %f px\n\tLocality = %f (1 / px)\n\tSpatial coherence weight = %f\n\tMinimum inlier number = %d\n", 
		fundamental_threshold, 
		homography_threshold,
		locality,
		spatial_weight,
		minimum_inlier_number);

	// Apply Multi-H
	ApplyMultiH(srcImagePath, 
		dstImagePath, 
		output_correspondence_path, 
		input_correspondence_path, 
		output_srcImagePath, 
		output_dstImagePath,
		minimum_inlier_number,
		fundamental_threshold,
		homography_threshold,
		locality,
		spatial_weight);
	
	return 0;
}

bool GetTaskName(int idx, std::string &task)
{
	if (idx < 0 || idx >= 24)
		return false;

	std::string built_in_tasks[] = { "barrsmith", "bonhall", "bonython",
		"boxesandbooks", "elderhalla", "elderhallb",
		"glasscasea", "glasscaseb", "graffiti",
		"hartley", "johnssona", "johnssonb",
		"ladysymon", "library", "napiera",
		"napierb", "neem", "nese",
		"oldclassicswing", "physics", "sene",
		"stairs", "unihouse", "unionhouse" };

	task = built_in_tasks[idx];
	return true;
}

void PrintUsage()
{
	printf("Help of Multi-H\n");
	printf("After running Multi-H it puts every results into a new folder in folder 'results' named as the current test case.\n");
	printf("If you use this code please cite 'Barath, D. and cv::Matas, J. and Hajder, L., Multi-H: Efficient Recovery of Tangent Planes in Stereo Images. 27th British Machine Vision Conference, 2016.'\n");

	printf("Usage 1. - Using a built-in test with standard parameterization\n");
	printf("\tExample: multih.exe 1 0\n");
	printf("\tParam 1.: Usage type = 0 - determines the use case\n");
	printf("\tParam 2.: Task idx = [0; 24] - built-in test id\n");
	printf("\t\t0 - barrsmith,\n\t\t1 - bonhall,\n\t\t3 - bonython,\n\t\t4 - boxesandbooks,\n\t\t5 - elderhalla,\n\t\t6 - elderhallb,\n\t\t7 - glasscasea,\n\t\t8 - glasscaseb, \n\t\t9 - graffiti,\n\t\t10 - hartley,\n\t\t11 - johnssona,\n\t\t12 - johnssonb,\n\t\t13 - ladysymon,\n\t\t14 - library,\n\t\t15 - napiera,\n\t\t16 - napierb,\n\t\t17 - neem,\n\t\t18 - nese,\n\t\t19 - oldclassicswing,\n\t\t20 - physics,\n\t\t21 - sene,\n\t\t22 - stairs,\n\t\t23 - unihouse,\n\t\t24 - unionhouse\n");

	printf("Usage 2. - Using a built-in test with custom parameterization\n");
	printf("\tExample: multih.exe 2 0 3.0 2.4 0.005 0.5 5\n");
	printf("\tParam 1.: Usage type = 1 - determines which parameterization is used\n");
	printf("\tParam 2.: Task idx = [0; 24] - built-in test id\n");
	printf("\tParam 3.: Fundamental matrix threshold (e.g. 3 pixels)\n");
	printf("\tParam 4.: Homography threshold (e.g. 2 pixels)\n");
	printf("\tParam 5.: Locality (e.g. 1 / 100 pixels = 0.01)\n");
	printf("\tParam 6.: Spatial coherence weight = [0,1]\n");
	printf("\tParam 7.: Minimum inlier number >= 0\n");

	printf("Usage 3. - Using custom input with standard parameterization\n");
	printf("\tExample: multih.exe 3 data/johnssona/barrsmith1.png data/johnssona/barrsmith2.png barrsmith\n");
	printf("\tParam 1.: Usage type = 3 - determines the use case\n");
	printf("\tParam 2.: Input source image path\n");
	printf("\tParam 3.: Input destination image path\n");
	printf("\tParam 4.: Test name\n");
	
	printf("Usage 4. - Using custom input with custom parameterization\n");
	printf("\tExample: multih.exe 4 data/johnssona/barrsmith1.png data/johnssona/barrsmith2.png barrsmith 3.0 2.4 0.005 0.5 5\n");
	printf("\tParam 1.: Usage type = 4 - determines the use case\n");
	printf("\tParam 2.: Input source image path\n");
	printf("\tParam 3.: Input destination image path\n");
	printf("\tParam 4.: Test name\n");
	printf("\tParam 5.: Fundamental matrix threshold (e.g. 3 pixels)\n");
	printf("\tParam 6.: Homography threshold (e.g. 2 pixels)\n");
	printf("\tParam 7.: Locality (e.g. 1 / 100 pixels = 0.01)\n");
	printf("\tParam 8.: Spatial coherence weight = [0,1]\n");
	printf("\tParam 9.: Minimum inlier number >= 0\n");
}

void ApplyMultiH(std::string srcPath, 
	std::string dstPath, 
	std::string out_corrPath, 
	std::string in_corrPath,
	std::string output_srcImagePath,
	std::string output_dstImagePath,
	int minimum_inlier_number,
	double fundamental_threshold,
	double homography_threshold,
	double locality,
	double spatial_weight)
{
	std::vector<cv::Point2d> srcPointsOrig, dstPointsOrig;
	std::vector<cv::Mat> origAffines;

	cv::Mat img1 = cv::imread(srcPath);
	cv::Mat img2 = cv::imread(dstPath);

	// Detect and track feature points
	if (!LoadPointsFromFile(srcPointsOrig, dstPointsOrig, origAffines, in_corrPath.c_str()))
	{
		printf("Detect ASIFT Keypoints\n");
		cv::Mat F;
		DoASIFT(F, img1, img2, srcPointsOrig, dstPointsOrig, origAffines, "SIFT");

		if (in_corrPath != "")	
			SavePointsToFile(srcPointsOrig, dstPointsOrig, origAffines, in_corrPath.c_str());
	}
	
	// Apply MultiHAF
	MultiH *multiH = new MultiH(fundamental_threshold, homography_threshold, locality, spatial_weight, minimum_inlier_number);
	multiH->Process(srcPointsOrig, dstPointsOrig, origAffines);
	
	std::vector<int> labeling;
	multiH->GetLabels(labeling);
	int iterationNum = multiH->GetIterationNumber();

	if (multiH->GetClusterNumber() < 1)
	{
		multiH->Release();
		srcPointsOrig.resize(0);
		dstPointsOrig.resize(0);
		origAffines.resize(0);

		std::cerr << "No homographies were found!\n";
		return;
	}

	// Save data to files
	multiH->DrawClusters(img1, img2, 2);
	std::vector<cv::Point2d> src_points, dst_points;
	std::vector<cv::Mat> affinities;
	std::vector<int> labels;

	multiH->GetLabels(labels);
	if (labels.size() == srcPointsOrig.size())
	{
		SavePointsToFile(srcPointsOrig, dstPointsOrig, origAffines, labels, out_corrPath.c_str());
	} 
	else
	{
		multiH->GetSourcePoints(src_points);
		multiH->GetDestinationPoints(dst_points);
		multiH->GetAffinities(affinities);
		SavePointsToFile(src_points, dst_points, affinities, labels, out_corrPath.c_str());
	}

	cv::imwrite(output_srcImagePath, img1);
	cv::imwrite(output_dstImagePath, img2);

	cv::imshow("Image 1", img1);
	cv::imshow("Image 2", img2);

	cv::waitKey(0);

	multiH->Release();
	srcPointsOrig.resize(0);
	dstPointsOrig.resize(0);
	origAffines.resize(0);
}

void LoadAndTrackImages(std::string ptsPath, cv::Mat &img1, cv::Mat &img2, std::vector<cv::Point2d> &srcPoints, std::vector<cv::Point2d> &dstPoints, std::vector<cv::Mat> &affines, std::string method)
{
	std::vector<cv::Point2d> srcPointsTmp;
	std::vector<cv::Point2d> dstPointsTmp;
	std::vector<std::pair<double, double>> anglesTmp;
	std::vector<std::pair<double, double>> sizesTmp;
	
	if (!LoadPointsFromFile(srcPoints, dstPoints, affines, ptsPath.c_str()))
	{
		cv::Mat F;
		DoASIFT(F, img1, img2, srcPoints, dstPoints, affines, method);
		SavePointsToFile(srcPoints, dstPoints, affines, ptsPath.c_str());
	}
	
	std::cout << "Found " << srcPoints.size() << " matches." << std::endl;
}

void DoASIFT(cv::Mat &F, cv::Mat &img1, cv::Mat &img2, std::vector<cv::Point2d> &srcPoints, std::vector<cv::Point2d> &dstPoints, std::vector<cv::Mat> &affines, std::string method)
{	
	std::vector<EllipticKeyPoint> keypoints1, keypoints2;
	std::vector<std::vector< cv::DMatch >> matches_vector;

	/*Run ASIFT and cv::Matching*/
	{
		cv::Mat descriptors1, descriptors2;
		EllipticASiftDetector detector;
		detector.detectAndCompute(img1, keypoints1, descriptors1, method);
		detector.detectAndCompute(img2, keypoints2, descriptors2, method);

		if (descriptors1.type() != CV_32F) {
			descriptors1.convertTo(descriptors1, CV_32F);
		}

		if (descriptors2.type() != CV_32F) {
			descriptors2.convertTo(descriptors2, CV_32F);
		}

		printf("** A %s Keypoints 1.: %d\n", method, keypoints1.size());
		printf("** A %s Keypoints 2.: %d\n", method, keypoints2.size());
		
		cv::FlannBasedMatcher matcher(new cv::flann::KDTreeIndexParams(5), new cv::flann::SearchParams(32));
		matcher.knnMatch(descriptors1, descriptors2, matches_vector, 2); //TOOD k = 2
	}

	srcPoints.clear(); dstPoints.clear();
	affines.clear(); 

	/* Robustify */
	{
		static const double ratio = 0.75;
		for (auto m : matches_vector)
		{
			if (m.size() == 2 && m[0].distance < m[1].distance * ratio)
			{
				auto& kp1 = keypoints1[m[0].queryIdx];
				auto& kp2 = keypoints2[m[0].trainIdx];
				srcPoints.push_back(kp1.pt);
				dstPoints.push_back(kp2.pt);

				cv::Mat_<double> A = kp2.transformation * kp1.transformation.inv();

				affines.push_back(A);
			}
		}
	}
}

bool LoadPointsFromFile(std::vector<cv::Point2d> &srcPoints, std::vector<cv::Point2d> &dstPoints, std::vector<cv::Mat> &affines, const char* file)
{
	std::ifstream infile(file);

	if (!infile.is_open())
		return false;

	double x1, y1, x2, y2;
	double a1, a2, a3, a4;

	while (infile >> x1 >> y1 >> x2 >> y2 >> a1 >> a2 >> a3 >> a4)
	{
		srcPoints.push_back(cv::Point2d(x1, y1));
		dstPoints.push_back(cv::Point2d(x2, y2));
		affines.push_back((cv::Mat_<double>(2,2) << a1, a2, a3, a4));
	}

	infile.close();

	std::vector<uchar> mask;
	findFundamentalMat(srcPoints, dstPoints, CV_FM_RANSAC, 2.0, 0.99, mask);
	for (int i = static_cast<int>(srcPoints.size()) - 1; i >= 0; --i)
	{
		if (!mask[i])
		{
			srcPoints.erase(srcPoints.begin() + i);
			dstPoints.erase(dstPoints.begin() + i);
			affines.erase(affines.begin() + i);
		}
	}
	
	return true;
}

bool SavePointsToFile(std::vector<cv::Point2d> &srcPoints, std::vector<cv::Point2d> &dstPoints, std::vector<cv::Mat> &affines, const char* file)
{
	std::ofstream outfile(file, std::ios::out);

	for (auto i = 0; i < srcPoints.size(); ++i)
	{
		outfile << srcPoints[i].x << " " << srcPoints[i].y << " " << dstPoints[i].x << " " << dstPoints[i].y << " ";
		outfile << affines[i].at<double>(0, 0) << " " << affines[i].at<double>(0, 1) << " " << affines[i].at<double>(1, 0) << " " << affines[i].at<double>(1, 1) << std::endl;
	}

	outfile.close();

	return true;
}

bool SavePointsToFile(std::vector<cv::Point2d> &srcPoints, std::vector<cv::Point2d> &dstPoints, std::vector<cv::Mat> &affines, std::vector<int> &labels, const char* file)
{
	std::ofstream outfile(file, std::ios::out);

	if (!outfile.is_open())
		return false;

	for (auto i = 0; i < srcPoints.size(); ++i)
	{
		outfile << srcPoints[i].x << " " << srcPoints[i].y << " " << dstPoints[i].x << " " << dstPoints[i].y << " " << 
			affines[i].at<double>(0, 0) << " " << affines[i].at<double>(0, 1) << " " << affines[i].at<double>(1, 0) << " " << affines[i].at<double>(1, 1) << " " << 
			labels[i] << std::endl;
	}

	outfile.close();

	return true;
}