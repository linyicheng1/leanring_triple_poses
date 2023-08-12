#ifndef LEARNING_TRIPLE_POSES_LEARNING_TRIPLE_POSES_H
#define LEARNING_TRIPLE_POSES_LEARNING_TRIPLE_POSES_H

#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <Eigen/Core>
#include <string>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include "homotopy.h"

class LearningTriplePoses {
public:
    LearningTriplePoses(const std::string& model_path, const std::string& set_path);
    bool solve(const std::vector<Eigen::Vector2d>& pts1,
               const std::vector<Eigen::Vector2d>& pts2,
               const std::vector<Eigen::Vector2d>& pts3,
               std::vector<float>& depth1,
               Eigen::Matrix3d& R01, Eigen::Vector3d& t01,
               Eigen::Matrix3d& R02, Eigen::Vector3d& t02);

    bool ransac_solve(const std::vector<Eigen::Vector2d>& pts1,
                      const std::vector<Eigen::Vector2d>& pts2,
                      const std::vector<Eigen::Vector2d>& pts3,
                      Eigen::Matrix3d& R01, Eigen::Vector3d& t01,
                      Eigen::Matrix3d& R02, Eigen::Vector3d& t02,
                      std::vector<char>& inliers);


private:
    track_settings settings;
    std::vector<std::vector<double>> anchors;
    std::vector<std::vector<double>> start_a;
    std::vector<std::vector<double>> depths_a;
    std::vector<std::vector<float>> ws;
    std::vector<std::vector<float>> bs;
    std::vector<std::vector<float>> ps;
    std::vector<int> a_;
    std::vector<int> b_;

    double problem[24];
    double params[48];
    float depth[12];
    double solution[12];


    void load_NN(std::string model_dir, std::vector<std::vector<float>> &ws, std::vector<std::vector<float>> &bs, std::vector<std::vector<float>> &ps, std::vector<int> &a_, std::vector<int> &b_);
    bool load_anchors(std::string data_file,
                      std::vector<std::vector<double>> &problems,
                      std::vector<std::vector<double>> &start,
                      std::vector<std::vector<double>> &depths);

    void pose_estimation_3d3d(const std::vector<Eigen::Vector3d>& pts1,
                              const std::vector<Eigen::Vector3d>& pts2,
                              Eigen::Matrix3d &R, Eigen::Vector3d &t);
    static int projection(const std::vector<Eigen::Vector2d>& pts1,
                   const std::vector<Eigen::Vector2d>& pts2,
                   const Eigen::Matrix3d& R, const Eigen::Vector3d& t,
                   std::vector<char>& inliers);
};


#endif //LEARNING_TRIPLE_POSES_LEARNING_TRIPLE_POSES_H
