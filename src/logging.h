//
// Created by mimimint on 4/16/20.
//

#ifndef CAMERA_FUSION_LOGGING_H
#define CAMERA_FUSION_LOGGING_H

#include <ostream>
#include <iostream>

const std::string LOGFILE_1 = "../logs/detectionRecallLogs.csv";
const std::string LOGFILE_2 = "../logs/detectionPerformanceLogs.csv";

// creates file and header for csv analysis
void setUpLogfiles() {
    std::ofstream logfile1;
    logfile1.open(LOGFILE_1, std::ios::out);
    logfile1 << "Keypointdetector,Avg Num Keypoint Detections,Descriptor Algorithm,Avg Num Descriptor Matches,Avg Keypointsize,FocusOnVehicle\n";
    logfile1.close();
    std::ofstream logfile2;
    logfile2.open(LOGFILE_2, std::ios::out);
    logfile2
            << "Keypointdetector,Avg Keypoint Detection Time,Descriptor Algorithm,Avg Descriptor extraction time,FocusOnVehicle\n";
    logfile2.close();
}

// logs number of keypoint detections and descriptor matches
// is performed on cumulated data after all images processed
void logDetectionAndMatchResults(const std::string& detectorAlgorithm, double avgNumDetections,
                                 const std::string& descriptorAlgorithm,
                                 double avgNumDescriptorMatches, double avgKeypointSize, const bool& bFocusOnVehicle) {
    std::ofstream logfile;
    logfile.open(LOGFILE_1, std::ios::out | std::ios::app);
    logfile << detectorAlgorithm << "," << avgNumDetections << "," << descriptorAlgorithm << "," << avgNumDescriptorMatches
            << "," << avgKeypointSize << ","
            << std::boolalpha << bFocusOnVehicle << "\n";
    logfile.close();
}

// logs the time the keypoint detection and descriptor extraction takes
// is performed on cumulated data after all images processed
void logDetectionAndExtractionPerformance(const std::string& detectorAlgorithm, double detectionTime,
                                          const std::string& descriptorAlgorithm,
                                          double extractionTime, const bool& bFocusOnVehicle) {

    std::ofstream logfile;
    logfile.open(LOGFILE_2, std::ios::out | std::ios::app);
    logfile << detectorAlgorithm << "," << detectionTime << "," << descriptorAlgorithm << "," << extractionTime << ","
            << std::boolalpha << bFocusOnVehicle << "\n";
    logfile.close();
}

#endif //CAMERA_FUSION_LOGGING_H
