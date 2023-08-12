#include <opencv2/opencv.hpp>
#include <fstream>


/**
 * @brief SIFT feature detection
 * @param img
 * @return
 */
void  detect_kps(const cv::Mat& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) {
    cv::Ptr<cv::FeatureDetector> detector = cv::SIFT::create();
    detector->detect(img, keypoints);
    detector->compute(img, keypoints, descriptors);
}

/**
 * @brief
 * @param descriptors1
 * @param descriptors2
 * @return
 */
std::vector<cv::DMatch> feature_matching(std::vector<cv::KeyPoint>& kps1,
                                         std::vector<cv::KeyPoint>& kps2,
                                         cv::Mat& descriptors1,
                                         cv::Mat& descriptors2) {
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce");
    std::vector<cv::DMatch> matches, good_matches;
    matcher->match(descriptors1, descriptors2, matches);
    // filter matches by f matrix
    std::vector<cv::Point2f> pts1, pts2;
    std::vector<uchar> status12, status13;
    for ( auto& m:matches ) {
        pts1.push_back( kps1[m.queryIdx].pt );
        pts2.push_back( kps2[m.trainIdx].pt );
    }
    cv::findFundamentalMat(pts1, pts2, cv::FM_RANSAC, 3, 0.99, status12);
    for ( int i=0; i<status12.size(); i++ ) {
        if ( status12[i] ) {
            good_matches.push_back(matches[i]);
        }
    }
    return good_matches;
}

/**
 * @brief find tri-matches from two matches list
 * @param matches12
 * @param matches13
 * @return
 */
std::vector<std::vector<int>> tri_matches(std::vector<cv::DMatch>& matches12, std::vector<cv::DMatch>& matches13) {
    std::vector<std::vector<int>> tri_matches;
    for ( auto& m12:matches12 ) {
        for ( auto& m13:matches13 ) {
            if ( m12.queryIdx == m13.queryIdx ) {
                std::vector<int> tri_match;
                tri_match.push_back(m12.queryIdx);
                tri_match.push_back(m12.trainIdx);
                tri_match.push_back(m13.trainIdx);
                tri_matches.push_back(tri_match);
            }
        }
    }
    return tri_matches;
}

void show_tri_matches(cv::Mat img1, cv::Mat img2, cv::Mat img3,
                      std::vector<cv::KeyPoint>& kps1, std::vector<cv::KeyPoint>& kps2, std::vector<cv::KeyPoint>& kps3,
                      std::vector<std::vector<int>>& tri_matches) {
    cv::Mat show1, show2, show3;
    show1 = img1.clone(); show2 = img2.clone(); show3 = img3.clone();

    // show matched key points
    for ( auto& m:tri_matches ) {
        cv::circle(show1, kps1[m[0]].pt, 3, cv::Scalar(0, 0, 255), -1);
        cv::circle(show2, kps2[m[1]].pt, 3, cv::Scalar(0, 0, 255), -1);
        cv::circle(show3, kps3[m[2]].pt, 3, cv::Scalar(0, 0, 255), -1);
    }

    // cat images
    cv::Mat cat12, cat;
    cv::hconcat(show1, show2, cat12);
    cv::hconcat(cat12, show3, cat);
    // draw lines
    for ( auto& m:tri_matches ) {
        cv::Point2f p1 = kps1[m[0]].pt;
        cv::Point2f p2 = kps2[m[1]].pt;
        cv::Point2f p3 = kps3[m[2]].pt;
        cv::line(cat, p1, p2 + cv::Point2f((float)img1.cols, 0), cv::Scalar(0, 255, 0), 1);
        cv::line(cat, p2 + cv::Point2f((float)img1.cols, 0), p3 + cv::Point2f((float)img1.cols + (float)img2.cols, 0), cv::Scalar(0, 255, 0), 1);
    }
    // show
    cv::imshow("matches", cat);
}

void save_matches_txt(const std::string& save_path,
                        const std::vector<cv::KeyPoint>& kps1,
                        const std::vector<cv::KeyPoint>& kps2,
                        const std::vector<cv::KeyPoint>& kps3,
                        std::vector<std::vector<int>>& tri_matches) {
    float fx = 718.856, fy = 718.856, cx = 607.1928, cy = 185.2157;

    std::ofstream fout(save_path);

    for ( auto& m:tri_matches ) {
        fout << (kps1[m[0]].pt.x - cx) / fx << " " << (kps1[m[0]].pt.y - cy) / fy << " "
             << (kps2[m[1]].pt.x - cx) / fx << " " << (kps2[m[1]].pt.y - cy) / fy << " "
             << (kps3[m[2]].pt.x - cx) / fx << " " << (kps3[m[2]].pt.y - cy) / fy << std::endl;

//        std::cout << kps1[m[0]].pt.x << " " << kps1[m[0]].pt.y << " "
//                  << kps2[m[1]].pt.x << " " << kps2[m[1]].pt.y << " "
//                  << kps3[m[2]].pt.x << " " << kps3[m[2]].pt.y << std::endl;
    }
    fout.close();
}

int main(int argc, char** argv) {

    if ( argc != 5 ) {
        std::cout << "usage: ./sift img1_dir img2_dir img3_dir output.txt" << std::endl;
        return 1;
    }

    const std::string img1_dir = argv[1];
    const std::string img2_dir = argv[2];
    const std::string img3_dir = argv[3];
    const std::string save_path = argv[4];

    // load images
    cv::Mat img1 = cv::imread(img1_dir);
    cv::Mat img2 = cv::imread(img2_dir);
    cv::Mat img3 = cv::imread(img3_dir);

    // detect key points
    std::vector<cv::KeyPoint> kps1, kps2, kps3;
    cv::Mat des1, des2, des3;
    detect_kps(img1, kps1, des1);
    detect_kps(img2, kps2, des2);
    detect_kps(img3, kps3, des3);

    // match key points
    std::vector<cv::DMatch> matches12 = feature_matching(kps1, kps2, des1, des2);
    std::vector<cv::DMatch> matches13 = feature_matching(kps1, kps3, des1, des3);

    // tri matches
    std::vector<std::vector<int>> ids = tri_matches(matches12, matches13);
    save_matches_txt(save_path, kps1, kps2, kps3, ids);

    // show tri matches
    show_tri_matches(img1, img2, img3, kps1, kps2, kps3, ids);
    cv::waitKey(0);
}

