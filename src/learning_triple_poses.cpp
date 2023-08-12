#include "learning_triple_poses.h"
#include <random>

LearningTriplePoses::LearningTriplePoses(const std::string &model_folder, const std::string &set_path) {
    // Load settings
    bool succ_load = load_settings(set_path, settings);
    if (!succ_load) {
        std::cout << "Failed to load settings from " << set_path << std::endl;
        return;
    }
    // Load the anchors
    succ_load = load_anchors(model_folder+"/anchors.txt",anchors,start_a,depths_a);
    if (!succ_load) {
        std::cout << "Failed to load anchors from " << model_folder+"/anchors.txt" << std::endl;
        return;
    }
    // Load the NN
    load_NN(model_folder, ws, bs, ps, a_, b_);
}

bool LearningTriplePoses::ransac_solve(const std::vector<Eigen::Vector2d> &pts1, const std::vector<Eigen::Vector2d> &pts2,
                                       const std::vector<Eigen::Vector2d> &pts3, Eigen::Matrix3d &R01, Eigen::Vector3d &t01,
                                       Eigen::Matrix3d &R02, Eigen::Vector3d &t02, std::vector<char> &inliers) {
    int best_inliers = 0;
    int max_iter = pts1.size() / 4;
    std::default_random_engine e(time(nullptr));
    std::uniform_int_distribution<int> u(0, (int)pts1.size() - 1);

    std::vector<int> idx(4);
//    idx[0] = -4; idx[1] = -3; idx[2] = -2; idx[3] = -1;
    for (int i = 0; i < max_iter; ++i) {
        // 1. randomly select 4 points
        std::vector<Eigen::Vector2d> pt1, pt2, pt3;
        while (idx[0] == idx[1] || idx[0] == idx[2] || idx[0] == idx[3] ||
               idx[1] == idx[2] || idx[1] == idx[3] || idx[2] == idx[3]) {
            for (int & j : idx) {
                j = u(e);
            }
        }
//        for (int & j : idx) {
//            j += 4;
//        }
        for (int j = 0; j < 4; ++j) {
            pt1.push_back(pts1[idx[j]]);
            pt2.push_back(pts2[idx[j]]);
            pt3.push_back(pts3[idx[j]]);
        }
        idx[0] = idx[1] = idx[2] = idx[3] = 0;

        // 2. solve the problem
        Eigen::Matrix3d R01_, R02_;
        Eigen::Vector3d t01_, t02_;
        std::vector<float> tmp;
        bool succ = solve(pt1, pt2, pt3, tmp, R01_, t01_, R02_, t02_);
        if (!succ) {
            continue;
        }

        // 3. count the inliers
        std::vector<char> inliers_, inliers01, inliers02;
        projection(pts1, pts2, R01_, t01_, inliers01);
        projection(pts1, pts3, R02_, t02_, inliers02);
        int num_inliers = 0;
        for (int j = 0; j < inliers01.size(); ++j) {
            if (inliers01[j] && inliers02[j]) {
                inliers_.emplace_back(1);
                num_inliers++;
            } else {
                inliers_.emplace_back(0);
            }
        }

        // 4. update the best solution
        if (num_inliers > (int)inliers01.size() * 0.8)
        {
            best_inliers = std::max(best_inliers, num_inliers);
            R01 = R01_;
            t01 = t01_;
            R02 = R02_;
            t02 = t02_;
            inliers = inliers_;
            //if (num_inliers > (int)inliers01.size() * 0.90)
            {
                // output
                std::cout << "R01 = " << R01 << std::endl;
                std::cout << "t01 = " << t01 << std::endl;
                std::cout << "R02 = " << R02 << std::endl;
                std::cout << "t02 = " << t02 << std::endl;
                std::cout<<"num_inliers: "<<num_inliers<<std::endl;
//                break;
            }
        }

    }
    return true;
}

bool LearningTriplePoses::solve(const std::vector<Eigen::Vector2d> &pts1, const std::vector<Eigen::Vector2d> &pts2,
                                const std::vector<Eigen::Vector2d> &pts3, std::vector<float> &depth1,
                                Eigen::Matrix3d &R01, Eigen::Vector3d &t01, Eigen::Matrix3d &R02, Eigen::Vector3d &t02) {
    // load the problem
    problem[0] = pts1[0].x(); problem[1] = pts1[1].x(); problem[2] = pts1[2].x(); problem[3] = pts1[3].x();
    problem[4] = pts1[0].y(); problem[5] = pts1[1].y(); problem[6] = pts1[2].y(); problem[7] = pts1[3].y();
    problem[8] = pts2[0].x(); problem[9] = pts2[1].x(); problem[10] = pts2[2].x(); problem[11] = pts2[3].x();
    problem[12] = pts2[0].y(); problem[13] = pts2[1].y(); problem[14] = pts2[2].y(); problem[15] = pts2[3].y();
    problem[16] = pts3[0].x(); problem[17] = pts3[1].x(); problem[18] = pts3[2].x(); problem[19] = pts3[3].x();
    problem[20] = pts3[0].y(); problem[21] = pts3[1].y(); problem[22] = pts3[2].y(); problem[23] = pts3[3].y();

    // mlp
    float input[24];
    for (int i = 0; i < 24; ++i) {
        input[i] = (float)problem[i];
    }
    Eigen::Map<Eigen::VectorXf> input_n2(input,24);
    Eigen::VectorXf input_ = input_n2;
    Eigen::VectorXf output_;

    int layers = (int)b_.size();
    for(int i=0;i<layers;++i)
    {
        float * ws_ = &ws[i][0];
        float * bs_ = &bs[i][0];
        float * ps_ = &ps[i][0];
        const Eigen::Map< const Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>, Eigen::Aligned > weights = Eigen::Map< const Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>, Eigen::Aligned >(ws_,a_[i],b_[i]);
        const Eigen::Map<const Eigen::VectorXf> bias = Eigen::Map<const Eigen::VectorXf>(bs_,a_[i]);

        output_ = weights*input_+bias;

        if(i==layers-1) break;

        const Eigen::Map<const Eigen::VectorXf> prelu = Eigen::Map<const Eigen::VectorXf>(ps_,a_[i]);
        input_ = output_.cwiseMax(output_.cwiseProduct(prelu));
    }

    // find the output with the highest score
    double best = -1000;
    int p = 0;

    for(int j=1;j<a_[layers-1];++j)
    {
        if(output_(j) > best)
        {
            best = output_(j);
            p = j;
        }
    }
    p = p-1;
    if(p==-1)
        return false;

    //copy the start problem
    for(int a=0;a<24;a++)
    {
        params[a] = anchors[p][a];
        params[a+24] = problem[a];
    }

    // copy the start solution
    double start[12];
    for(int a=0;a<12;++a)
        start[a] = start_a[p][a];

    // track the problem
    int num_steps;
    int status = track(settings, start, params, solution, &num_steps);

    // evaluate the solution
    if (status == 2)
    {
//        for (double i : solution)
//        {
//            std::cout << i << " ";
//        }
//        std::cout << std::endl;
        std::vector<Eigen::Vector3d> pt3d_1(4), pt3d_2(4), pt3d_3(4);


        pt3d_1[0] = Eigen::Vector3d(pts1[0].x(), pts1[0].y(), 1);
        pt3d_1[1] = Eigen::Vector3d(pts1[1].x() * solution[0], pts1[1].y() * solution[0], solution[0]);
        pt3d_1[2] = Eigen::Vector3d(pts1[2].x() * solution[1], pts1[2].y() * solution[1], solution[1]);
        pt3d_1[3] = Eigen::Vector3d(pts1[3].x() * solution[2], pts1[3].y() * solution[2], solution[2]);

        pt3d_2[0] = Eigen::Vector3d(pts2[0].x() * solution[3], pts2[0].y() * solution[3], solution[3]);
        pt3d_2[1] = Eigen::Vector3d(pts2[1].x() * solution[4], pts2[1].y() * solution[4], solution[4]);
        pt3d_2[2] = Eigen::Vector3d(pts2[2].x() * solution[5], pts2[2].y() * solution[5], solution[5]);
        pt3d_2[3] = Eigen::Vector3d(pts2[3].x() * solution[6], pts2[3].y() * solution[6], solution[6]);

        pt3d_3[0] = Eigen::Vector3d(pts3[0].x() * solution[7], pts3[0].y() * solution[7], solution[7]);
        pt3d_3[1] = Eigen::Vector3d(pts3[1].x() * solution[8], pts3[1].y() * solution[8], solution[8]);
        pt3d_3[2] = Eigen::Vector3d(pts3[2].x() * solution[9], pts3[2].y() * solution[9], solution[9]);
        pt3d_3[3] = Eigen::Vector3d(pts3[3].x() * solution[10], pts3[3].y() * solution[10], solution[10]);

        // solve the PnP problem
        pose_estimation_3d3d(pt3d_1, pt3d_2, R01, t01);
        pose_estimation_3d3d(pt3d_1, pt3d_3, R02, t02);

        // scale
        double scale = t01.norm();
        t01 = t01 / scale;
        t02 = t02 / scale;

        return true;
    }

    return false;
}

/**
 * @brief 3D-3D pose estimation using SVD
 * @param pts1 matches points in the first image
 * @param pts2 matches points in the second image
 * @param R    rotation matrix
 * @param t    translation vector
 */
void LearningTriplePoses::pose_estimation_3d3d(const std::vector<Eigen::Vector3d> &pts1,
                                               const std::vector<Eigen::Vector3d> &pts2, Eigen::Matrix3d &R,
                                               Eigen::Vector3d &t) {
    Eigen::Vector3d p1 = Eigen::Vector3d::Zero(), p2 = Eigen::Vector3d::Zero(); // center of mass
    int N = (int)pts1.size();
    for(int i = 0;i < N;i ++) {
        p1 += pts1[i];
        p2 += pts2[i];
    }
    p1 = p1 / N;
    p2 = p2 / N;
    std::vector<Eigen::Vector3d> q1(N), q2(N); // remove the center
    for(int i = 0;i < N;i ++) {
        q1[i] = pts1[i] - p1;
        q2[i] = pts2[i] - p2;
    }
    // compute q1*q2^T
    Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
    for(int i = 0;i < N;i ++) {
        W += q1[i] * q2[i].transpose();
    }
    // SVD on W
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    const Eigen::Matrix3d& V = svd.matrixV();
    // R = U*V^T
    if(U.determinant() * V.determinant() < 0) {
        for(int x = 0;x < 3;x ++) {
            U(x, 2) *= -1;
        }
    }
    R = U * V.transpose();
    t = p1 - R * p2;
}

/**
 * @brief projection error of 2d-2d matches
 * @param pts1 matches 2d feature in the first image
 * @param pts2 matches 2d feature in the second image
 * @param R    rotation matrix
 * @param t    translation vector
 */
int LearningTriplePoses::projection(const std::vector<Eigen::Vector2d> &pts1, const std::vector<Eigen::Vector2d> &pts2,
                                    const Eigen::Matrix3d &R, const Eigen::Vector3d &t, std::vector<char> &inliers) {
    int N = (int)pts1.size();
    inliers.resize(N);
    int num_inliers = 0;

    // 1. compute the essential matrix
    Eigen::Matrix3d t_hat;
    t_hat << 0, -t(2), t(1),
            t(2), 0, -t(0),
            -t(1), t(0), 0;
    Eigen::Matrix3d E = t_hat * R;

    // 2. compute the error
    for (int i = 0; i < N; i++) {
        Eigen::Vector3d x1(pts1[i].x(), pts1[i].y(), 1);
        Eigen::Vector3d x2(pts2[i].x(), pts2[i].y(), 1);
        double error = x2.transpose() * E * x1;
        if (error  < 0.05) {
            inliers[i] = 1;
            num_inliers++;
        } else {
            inliers[i] = 0;
        }
    }

    return num_inliers;
}

bool LearningTriplePoses::load_anchors(std::string data_file, std::vector<std::vector<double>> &problems,
                                       std::vector<std::vector<double>> &start,
                                       std::vector<std::vector<double>> &depths) {
    std::ifstream f;
    f.open(data_file);

    if(!f.good())
    {
        f.close();
        std::cout << "Anchor file not available\n";
        return 0;
    }

    int n;
    f >> n;
    std::cout << n << " anchors\n";

    problems = std::vector<std::vector<double>>(n);
    start = std::vector<std::vector<double>>(n);
    depths = std::vector<std::vector<double>>(n);

    //load the problems
    for(int i=0;i<n;i++)
    {
        std::vector<double> problem(24);
        std::vector<double> cst(12);
        std::vector<double> depth(13);

        //load the points
        for(int j=0;j<24;j++)
        {
            double u;
            f >> u;

            problem[j] = u;
        }
        problems[i] = problem;

        //load the depths and convert them to the solution
        double first_depth;
        f >> first_depth;
        depth[0] = first_depth;
        for(int j=0;j<11;j++)
        {
            double u;
            f >> u;

            cst[j] = u/first_depth;
            depth[j+1] = u;
        }
        double l;
        f >> l;
        cst[11] = l;
        depth[12] = l;

        start[i] = cst;
        depths[i] = depth;
    }
    f.close();
    return 1;
}

void LearningTriplePoses::load_NN(std::string model_dir, std::vector<std::vector<float>> &ws,
                                  std::vector<std::vector<float>> &bs, std::vector<std::vector<float>> &ps,
                                  std::vector<int> &a_, std::vector<int> &b_) {
    std::ifstream fnn;
    fnn.open(model_dir+"/nn.txt");
    int layers;
    fnn >> layers;
    ws = std::vector<std::vector<float>>(layers);
    bs = std::vector<std::vector<float>>(layers);
    ps = std::vector<std::vector<float>>(layers-1);
    a_ = std::vector<int>(layers);
    b_ = std::vector<int>(layers);
    for(int i=0;i<layers;++i)
    {
        int a;
        int b;
        fnn >> a;
        fnn >> b;
        a_[i] = a;
        b_[i] = b;

        std::cout << a << " " << b << "\n";

        std::vector<float> __attribute__((aligned(16))) cw(a*b);
        for(int j=0;j<a*b;++j)
        {
            float u;
            fnn >> u;
            cw[j] = u;
        }
        ws[i] = cw;

        fnn >> a;
        fnn >> b;
        std::vector<float> __attribute__((aligned(16))) cb(a);
        for(int j=0;j<a;++j)
        {
            float u;
            fnn >> u;
            cb[j] = u;
        }
        bs[i] = cb;

        if(i==layers-1)
            break;

        fnn >> a;
        fnn >> b;
        std::vector<float> __attribute__((aligned(16))) cp(a);
        for(int j=0;j<a;++j)
        {
            float u;
            fnn >> u;
            cp[j] = u;
        }
        ps[i] = cp;
    }
    fnn.close();
}








