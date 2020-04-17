/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <queue>
#include <cmath>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"
#include "matching2D.hpp"
#include "logging.h"

using namespace std;

// select options
const bool PERF_EVAL = true;
// flag to enable visualizations
const bool bVisualize = false;
const bool bFocusOnVehicle = true;

// processes the databuffer and returns a vector with two pairs:
// a pair containing the number of keypoint detections and the number of
// descriptor matches
// and a second pair containing the runtime for the keypoint detection (whole image)
// and the descriptor extraction (based on roi filter)
std::vector<std::pair<double, double>>
processDatabuffer(const std::string& detectorAlgorithm, const std::string& descriptorAlgorithm,
                  std::queue<DataFrame>& dataBuffer, cv::Mat& imgGray, const bool& bVis, const bool& bFocusOnVehicle) {
    // extract 2D keypoints from current image
    vector<cv::KeyPoint> keypoints; // create empty feature list for current image

    double keyPointDetectionTime = 0;
    if (detectorAlgorithm.compare("SHITOMASI") == 0) {
        keyPointDetectionTime = detKeypointsShiTomasi(keypoints, imgGray, false);
    } else if (detectorAlgorithm.compare("HARRIS") == 0) {
        keyPointDetectionTime = detKeypointsHarris(keypoints, imgGray, false);
    } else {
        keyPointDetectionTime = detKeypointsModern(keypoints, imgGray, detectorAlgorithm, false);
    }

    // only keep keypoints on the preceding vehicle
    //bool bFocusOnVehicle = true;
    cv::Rect vehicleRect(535, 180, 180, 150);
    if (bFocusOnVehicle) {
        std::vector<cv::KeyPoint> filteredKeyPoints;
        filteredKeyPoints.reserve(keypoints.size() / 2);
        for (auto kpIt = keypoints.begin(); kpIt != keypoints.end(); ++kpIt) {
            if (vehicleRect.contains(kpIt->pt)) {
                filteredKeyPoints.push_back(*kpIt);
            }
        }
        keypoints = std::move(filteredKeyPoints);
    }

    // optional : limit number of keypoints (helpful for debugging and learning)
    bool bLimitKpts = false;
    if (bLimitKpts) {
        int maxKeypoints = 50;

        if (detectorAlgorithm.compare("SHITOMASI") ==
            0) { // there is no response info, so keep the first 50 as they are sorted in descending quality order
            keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
        }
        cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
        cout << " NOTE: Keypoints have been limited!" << endl;
    }

    // push keypoints and descriptor for current frame to end of data buffer
    int numKeypointDetections = keypoints.size();
    std::cout << "Detected " << numKeypointDetections << " keypoints with " << detectorAlgorithm << std::endl;
    // calculate avg diameter of meaningful keypoint neighborhood
    double avgKeypointSize = 0.;
    if (PERF_EVAL) {
        for (auto& keypoint : keypoints) {
            avgKeypointSize += keypoint.size;
        }
        avgKeypointSize /= keypoints.size();
    }
    // make pair for return value, second value is placeholder for now
    std::pair<double, double> keyPointSize = std::make_pair(avgKeypointSize, 0.);

    dataBuffer.back().keypoints = std::move(keypoints);

    cout << "#2 : DETECT KEYPOINTS done" << endl;

    /* EXTRACT KEYPOINT DESCRIPTORS */

    cv::Mat descriptors;
    double descriptorExtractionTime = descKeypoints(dataBuffer.back().keypoints, dataBuffer.back()._cameraImg,
                                                    descriptors, descriptorAlgorithm);

    std::pair<double, double> runtimePerformance = std::make_pair(keyPointDetectionTime, descriptorExtractionTime);
    // push descriptors for current frame to end of data buffer
    dataBuffer.back().descriptors = descriptors;

    cout << "#3 : EXTRACT DESCRIPTORS done" << endl;
    int numDescriptorMatches = 0;
    if (dataBuffer.size() > 1) // wait until at least two images have been processed
    {

        /* MATCH KEYPOINT DESCRIPTORS */

        vector<cv::DMatch> matches;
        string matcherType = "MAT_BF";        // MAT_BF, MAT_FLANN
        // Binary descriptors:  BRIEF, BRISK, ORB, FREAK and KAZE
        string descriptorType;
        if (descriptorAlgorithm.compare("SIFT") != 0) {
            descriptorType = "DES_BINARY"; // DES_BINARY, DES_HOG
        } else {
            descriptorType = "DES_HOG";
        }
        string selectorType = "SEL_KNN";       // SEL_NN, SEL_KNN

        matchDescriptors(dataBuffer.front().keypoints, dataBuffer.back().keypoints,
                         dataBuffer.front().descriptors, dataBuffer.back().descriptors,
                         matches, descriptorType, matcherType, selectorType);

        // store matches in current data frame
        numDescriptorMatches = matches.size();
        dataBuffer.back().kptMatches = std::move(matches);
        cout << "#4 : MATCH KEYPOINT DESCRIPTORS done" << endl;

        // visualize matches between current and previous image

        if (bVis) {
            cv::Mat matchImg = dataBuffer.back()._cameraImg.clone();
            cv::drawMatches(dataBuffer.front()._cameraImg, dataBuffer.front().keypoints,
                            dataBuffer.back()._cameraImg, dataBuffer.back().keypoints,
                            dataBuffer.back().kptMatches, matchImg,
                            cv::Scalar::all(-1), cv::Scalar::all(-1),
                            vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
            std::ostringstream windowTitle;
            windowTitle <<"Keypoint/Descriptor: [" << detectorAlgorithm << "/" << descriptorAlgorithm <<"]";
            cv::namedWindow(windowTitle.str(), 7);
            cv::imshow(windowTitle.str(), matchImg);
            cout << "Press key to continue to next image" << endl;
            cv::waitKey(0); // wait for key to be pressed
        }

        dataBuffer.pop();
    }
    std::pair<double, double> recallPerformance = std::make_pair(numKeypointDetections, numDescriptorMatches);
    return std::vector<std::pair<double, double>>{recallPerformance, runtimePerformance, keyPointSize};
}


// handles image loading and starts processing of the databuffer
void processImages(const std::string& detectorAlgorithm, const std::string& descriptorAlgorithm, const bool& bVis,
                   const bool& bFocusOnVehicle) {
    // data location
    string dataPath = "../";

    // camera
    string imgBasePath = dataPath + "images/";
    string imgPrefix = "KITTI/2011_09_26/image_00/data/000000"; // left camera, color
    string imgFileType = ".png";
    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 9;   // last file index to load
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

    // misc
    int dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time
    // reimplemented dataBuffer as queue for optimal memory usage
    std::queue<DataFrame> dataBuffer;

    /* MAIN LOOP OVER ALL IMAGES */
    // count detections and matches for logging and evaluation
    double avgNumDetections = 0;
    double avgNumDescriptorMatches = 0;
    double avgDetectionTime = 0.;
    double avgExtractionTime = 0.;
    double avgKeypointSize = 0.;
    for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex++) {
        /* LOAD IMAGE INTO BUFFER */

        // assemble filenames for current index
        ostringstream imgNumber;
        imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
        string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

        // load image from file and convert to grayscale
        cv::Mat img, imgGray;
        img = cv::imread(imgFullFilename);
        cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

        // replaced vector with queue
        dataBuffer.emplace(imgGray);

        cout << "#1 : LOAD IMAGE INTO BUFFER done" << endl;

        if (PERF_EVAL) {
            auto performanceResults = processDatabuffer(detectorAlgorithm, descriptorAlgorithm, dataBuffer, imgGray, bVis,
                                                        bFocusOnVehicle);
            auto& detectionAndMatches = performanceResults.at(0);
            avgNumDetections += detectionAndMatches.first;
            avgNumDescriptorMatches += detectionAndMatches.second;
            auto& runtimePerformance = performanceResults.at(1);
            avgDetectionTime += runtimePerformance.first;
            avgExtractionTime += runtimePerformance.second;
            avgKeypointSize += performanceResults.at(2).first;
        } else {
            // return values are only needed for performance evaluation
            processDatabuffer(detectorAlgorithm, descriptorAlgorithm, dataBuffer, imgGray, bVis,
                              bFocusOnVehicle);
        }
    }
    avgNumDetections /= 10;
    avgNumDescriptorMatches /= 10;
    avgDetectionTime /= 10;
    avgExtractionTime /= 10;
    avgKeypointSize /= 10;
    if (PERF_EVAL) {
        logDetectionAndMatchResults(detectorAlgorithm, avgNumDetections, descriptorAlgorithm, avgNumDescriptorMatches,
                                    avgKeypointSize, bFocusOnVehicle);
        logDetectionAndExtractionPerformance(detectorAlgorithm, avgDetectionTime, descriptorAlgorithm, avgExtractionTime,
                                             bFocusOnVehicle);
    }
}

// handles logic for performance evaluation
int main(int argc, const char* argv[]) {

    std::vector<std::string> detectorAlgorithms;
    std::vector<std::string> descriptorAlgorithms;
    // loop over all detector types in performance evaluation mode
    if (PERF_EVAL) {
        detectorAlgorithms = {"SIFT", "SHITOMASI", "HARRIS", "FAST", "BRISK", "ORB", "AKAZE"};
        descriptorAlgorithms = {"BRISK", "BRIEF", "ORB", "FREAK", "AKAZE", "SIFT"};
        setUpLogfiles();
    } else {
        // change here for a single run
        detectorAlgorithms = {"SIFT"};
        descriptorAlgorithms = {"AKAZE"};
    }
    // loop over all possible keypoint and descriptor combinations in performance evaluation mode
    for (const std::string& detectorAlgorithm : detectorAlgorithms) {
        for (const std::string& descriptorAlgorithm : descriptorAlgorithms) {
            // skip combinations that don't work together
            // SIFT with ORB causes memory overflow
            if ((detectorAlgorithm.compare("SIFT") == 0 && descriptorAlgorithm.compare("ORB") == 0)) {
                continue;
            }
            // AKAZE descriptor only works with AKAZE keypoints
            else if (descriptorAlgorithm.compare("AKAZE") == 0 && detectorAlgorithm.compare("AKAZE") != 0) {
                continue;
            }
            std::cout << "###################################################################" << std::endl;
            std::cout << "Running *" << detectorAlgorithm << "* Keypoint detector with *" << descriptorAlgorithm
                      << "* Descriptor extractor" << std::endl;
            std::cout << "###################################################################" << std::endl;
            processImages(detectorAlgorithm, descriptorAlgorithm, bVisualize, bFocusOnVehicle);
        }
    }

    return 0;
}
