#include <iostream>
#include "learning_triple_poses.h"

void read_matches(const std::string& path,
                  std::vector<Eigen::Vector2d>& pts1,
                  std::vector<Eigen::Vector2d>& pts2,
                  std::vector<Eigen::Vector2d>& pts3) {
    std::ifstream fin(path);
    while (!fin.eof()) {
        Eigen::Vector2d pt1, pt2, pt3;
        fin >> pt1[0] >> pt1[1] >> pt2[0] >> pt2[1] >> pt3[0] >> pt3[1];
        pts1.emplace_back(pt1);
        pts2.emplace_back(pt2);
        pts3.emplace_back(pt3);
    }

//    Eigen::Matrix3f R0, R1, R2, R01, R02;
//    Eigen::Vector3f t0, t1, t2, t01, t02;

//    R0(0, 0) = 8.656549e-02; R0(0, 1) = 2.535974e-02; R0(0, 2) = -9.959234e-01; t0[0] = -7.834980e+00;
//    R0(1, 0) = -4.844174e-02; R0(1, 1) = 9.986006e-01; R0(1, 2) = 2.121737e-02; t0[1] = -7.122577e+00;
//    R0(2, 0) = 9.950677e-01; R0(2, 1) = 4.640756e-02; R0(2, 2) = 8.767281e-02; t0[2] = 2.416234e+02;
//
//    R1(0, 0) = 1.937520e-01; R1(0, 1) = 3.543131e-02; R1(0, 2) = -9.804106e-01; t1[0] = -9.050632e+00;
//    R1(1, 0) = -4.849689e-02; R1(1, 1) = 9.984717e-01; R1(1, 2) = 2.649992e-02; t1[1] = -7.098001e+00;
//    R1(2, 0) = 9.798511e-01; R1(2, 1) = 4.241244e-02; R1(2, 2) = 1.951742e-01; t1[2] = 2.418908e+02;
//
//    R2(0, 0) = 3.732099e-01; R2(0, 1) = 4.373391e-02; R2(0, 2) = -9.267156e-01; t2[0] = -1.001000e+01;
//    R2(1, 0) = -3.828661e-02; R2(1, 1) = 9.987634e-01; R2(1, 2) = 3.171512e-02; t2[1] = -7.076031e+00;
//    R2(2, 0) = 9.269565e-01; R2(2, 1) = 2.364440e-02; R2(2, 2) = 3.744228e-01; t2[2] = 2.424854e+02;
//
////    8.656549e-02 2.535974e-02 -9.959234e-01 -7.834980e+00 -4.844174e-02 9.986006e-01 2.121737e-02 -7.122577e+00 9.950677e-01 4.640756e-02 8.767281e-02 2.416234e+02
////    1.937520e-01 3.543131e-02 -9.804106e-01 -9.050632e+00 -4.849689e-02 9.984717e-01 2.649992e-02 -7.098001e+00 9.798511e-01 4.241244e-02 1.951742e-01 2.418908e+02
////    3.732099e-01 4.373391e-02 -9.267156e-01 -1.043713e+01 -3.828661e-02 9.987634e-01 3.171512e-02 -7.076031e+00 9.269565e-01 2.364440e-02 3.744228e-01 2.424854e+02
//    R01 = R0.transpose() * R1;
//    t01 = R0.transpose() * (t1 - t0);
//    R02 = R0.transpose() * R2;
//    t02 = R0.transpose() * (t2 - t0);
//
//    float scale = t01.norm();
//    t01 = t01 / scale;
//    t02 = t02 / scale;
//
//    std::cout << "R01: " << std::endl << R01 << std::endl;
//    std::cout << "t01: " << std::endl << t01 << std::endl;
//    std::cout << "R02: " << std::endl << R02 << std::endl;
//    std::cout << "t02: " << std::endl << t02 << std::endl;

}

int main(int argc, char** argv) {

    if (argc != 4) {
        std::cout << "Usage: ./learning_triple_poses <model_path> <set_path> <data_path>" << std::endl;
        return -1;
    }

    // init
    LearningTriplePoses learningTriplePoses(argv[1], argv[2]);
    std::vector<Eigen::Vector2d> pts1, pts2, pts3;
    Eigen::Matrix3d R01, R02;
    Eigen::Vector3d t01, t02;
    std::vector<char> inliers;

    // read matches
    read_matches(argv[3], pts1, pts2, pts3);

    // solve
    learningTriplePoses.ransac_solve(pts1, pts2, pts3, R01, t01, R02, t02, inliers);

    // output
    std::cout << "R01: " << std::endl << R01 << std::endl;
    std::cout << "t01: " << std::endl << t01 << std::endl;
    std::cout << "R02: " << std::endl << R02 << std::endl;
    std::cout << "t02: " << std::endl << t02 << std::endl;

    return 0;
}
