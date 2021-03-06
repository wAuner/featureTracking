#include <numeric>
#include "matching2D.hpp"

using namespace std;

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint>& kPtsSource, std::vector<cv::KeyPoint>& kPtsRef, cv::Mat& descSource,
                      cv::Mat& descRef,
                      std::vector<cv::DMatch>& matches, std::string descriptorType, std::string matcherType,
                      std::string selectorType) {
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    // brute force matching
    if (matcherType.compare("MAT_BF") == 0) {
        int normType;
        // select Norm depending on whether the descriptors are binary or HOG
        if (descriptorType.compare("DES_BINARY") == 0) {
            normType = cv::NORM_HAMMING;
        } else if (descriptorType.compare("DES_HOG") == 0) {
            normType = cv::NORM_L2;
        } else {
            std::cout << "Descriptor type unkown." << std::endl;
            exit(1);
        }
        matcher = cv::BFMatcher::create(normType, crossCheck);
    } else if (matcherType.compare("MAT_FLANN") == 0) {
        if (descSource.type() != CV_32F)
        { // OpenCV bug workaround : convert binary descriptors to floating point due to a bug in current OpenCV implementation
            descSource.convertTo(descSource, CV_32F);
            descRef.convertTo(descRef, CV_32F);
        }
        matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0) { // nearest neighbor (best match)
        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    } else if (selectorType.compare("SEL_KNN") == 0) { // k nearest neighbors (k=2)
        std::vector<std::vector<cv::DMatch>> knnMatches;
        matcher->knnMatch(descSource, descRef, knnMatches, 2);

        // filter matches based on distance ratio
        float minDistanceRatio = 0.8;
        for (auto knnPair : knnMatches) {
            float distanceRatio = knnPair.at(0).distance / knnPair.at(1).distance;
            if (distanceRatio < minDistanceRatio) {
                matches.push_back(knnPair.at(0));
            }
        }
    }
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
// returns the time it took to extract the keypoint descriptors in milliseconds
double descKeypoints(vector<cv::KeyPoint>& keypoints, cv::Mat& img, cv::Mat& descriptors, string descriptorType) {
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0) {

        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    } else if (descriptorType.compare("BRIEF") == 0) {
        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
    } else if (descriptorType.compare("ORB") == 0) {
        extractor = cv::ORB::create();
    } else if (descriptorType.compare("FREAK") == 0) {
        extractor = cv::xfeatures2d::FREAK::create();
    } else if (descriptorType.compare("AKAZE") == 0) {
        extractor = cv::AKAZE::create();
    } else if (descriptorType.compare("SIFT") == 0) {
        extractor = cv::xfeatures2d::SIFT::create();
    } else {
        std::cout << "Provided descriptor algorithm not supported." << std::endl;
        exit(1);
    }

    // perform feature description
    double t = (double) cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double) cv::getTickCount() - t) / cv::getTickFrequency();
    t = 1000 * t / 1.0;
    cout << descriptorType << " descriptor extraction in " << t << " ms" << endl;

    return t;
}

// returns the keypoint detection time
double detKeypointsHarris(vector<cv::KeyPoint>& keyPoints, cv::Mat& img, bool bVis) {
    // Detector parameters
    int blockSize = 2;     // for every pixel, a blockSize × blockSize neighborhood is considered
    int apertureSize = 3;  // aperture parameter for Sobel operator (must be odd)
    int minResponse = 100; // minimum value for a corner in the 8bit scaled response matrix
    double k = 0.04;       // Harris parameter (see equation for details)

    // Detect Harris corners and normalize output
    cv::Mat dst, dst_norm, dst_norm_scaled;
    dst = cv::Mat::zeros(img.size(), CV_32FC1);
    double t = (double) cv::getTickCount();
    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);

    // loop over every pixel in the resulting harris response matrix
    // and check whether there are overlaps
    // keep only maximum if there are overlaps
    float maxOverlap = 0.0;
    for (int row = 0; row < dst_norm.rows; row++) {
        for (int col = 0; col < dst_norm.cols; col++) {
            int pixelResponse = static_cast<int>(dst_norm.at<float>(row, col));
            // start checking if response exceeds minimum threshold
            if (pixelResponse > minResponse) {
                // create KeyPoint instance
                cv::KeyPoint newKeyPoint;
                // remember rows = y && cols = x, needs to be swapped for x,y coordinates
                newKeyPoint.pt = cv::Point2f(col, row);
                // area to consider for NMS
                newKeyPoint.size = 2 * apertureSize;
                newKeyPoint.response = pixelResponse;

                bool overlap = false;

                // compare with other keyPoints to find maximum
                for (auto it = keyPoints.begin(); it != keyPoints.end(); ++it) {
                    float overlap = cv::KeyPoint::overlap(newKeyPoint, *it);
                    // if the points overlap, check which one has the higher respsonse
                    // choose the one with the high response
                    if (overlap > maxOverlap) {
                        overlap = true;
                        if (newKeyPoint.response > (*it).response) {
                            *it = newKeyPoint;
                            break;
                        }
                    }
                }

                // add if no overlap with other points
                if (!overlap) {
                    keyPoints.push_back(newKeyPoint);
                }
            }
        }
    }
    t = ((double) cv::getTickCount() - t) / cv::getTickFrequency();
    t = 1000 * t / 1.0;

    if (bVis) {
        // show result
        cv::Mat nmsImage = dst_norm_scaled.clone();
        cv::drawKeypoints(dst_norm_scaled, keyPoints, nmsImage, cv::Scalar::all(-1),
                          cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::imshow("NMS", nmsImage);
        cv::waitKey(0);
    }
    return t;
}

// Detect keyPoints in image using the traditional Shi-Thomasi detector
// returns the keypoint detection time
double detKeypointsShiTomasi(vector<cv::KeyPoint>& keyPoints, cv::Mat& img, bool bVis) {
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keyPoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    double t = (double) cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it) {

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keyPoints.push_back(newKeyPoint);
    }
    t = ((double) cv::getTickCount() - t) / cv::getTickFrequency();
    t = 1000 * t / 1.0;
    cout << "Shi-Tomasi detection with n=" << keyPoints.size() << " keyPoints in " << t << " ms" << endl;

    // visualize results
    if (bVis) {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keyPoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
    return t;
}

// implements one of the following keypoint detectors: FAST, BRISK, ORB, AKAZE, SIFT
// returns the keypoint detection time
double
detKeypointsModern(std::vector<cv::KeyPoint>& keyPoints, cv::Mat& img, std::string detectorType, bool bVis) {
    cv::Ptr<cv::FeatureDetector> keyPointDetector;
    if (detectorType.compare("FAST") == 0) {
        int threshold = 50;
        double t = (double) cv::getTickCount();
        cv::FAST(img, keyPoints, threshold);
        t = ((double) cv::getTickCount() - t) / cv::getTickFrequency();
        return t;
    } else if (detectorType.compare("BRISK") == 0) {
        keyPointDetector = cv::BRISK::create();
    } else if (detectorType.compare("ORB") == 0) {
        keyPointDetector = cv::ORB::create();
    } else if (detectorType.compare("AKAZE") == 0) {
        keyPointDetector = cv::AKAZE::create();
    } else if (detectorType.compare("SIFT") == 0) {
        keyPointDetector = cv::xfeatures2d::SIFT::create();
    } else {
        std::cout << "Detector type not supported" << std::endl;
        exit(1);
    }
    double t = (double) cv::getTickCount();
    keyPointDetector->detect(img, keyPoints);
    t = ((double) cv::getTickCount() - t) / cv::getTickFrequency();
    t = 1000 * t / 1.0;

    return t;
}


