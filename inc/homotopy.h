#ifndef LEARNING_TRIPLE_POSES_HOMOTOPY_H
#define LEARNING_TRIPLE_POSES_HOMOTOPY_H

#include <Eigen/SVD>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>


struct track_settings
{
    track_settings():
            init_dt_(0.05),   // m2 tStep, t_step, raw interface code initDt
            min_dt_(1e-4),        // m2 tStepMin, raw interface code minDt
            end_zone_factor_(0.05),
            epsilon_(4e-2), // m2 CorrectorTolerance
            epsilon2_(epsilon_ * epsilon_),
            dt_increase_factor_(3.),  // m2 stepIncreaseFactor
            dt_decrease_factor_(1./dt_increase_factor_),  // m2 stepDecreaseFactor not existent in DEFAULT, using what is in track.m2:77
            infinity_threshold_(1e7), // m2 InfinityThreshold
            infinity_threshold2_(infinity_threshold_ * infinity_threshold_),
            max_corr_steps_(9),  // m2 maxCorrSteps (track.m2 param of rawSetParametersPT corresp to max_corr_steps in NAG.cpp)
            num_successes_before_increase_(4), // m2 numberSuccessesBeforeIncrease
            corr_thresh_(0.00001),
            anch_num_(134)
    { }

    double init_dt_;   // m2 tStep, t_step, raw interface code initDt
    double min_dt_;        // m2 tStepMin, raw interface code minDt
    double end_zone_factor_;
    double epsilon_; // m2 CorrectorTolerance (chicago.m2, track.m2), raw interface code epsilon (interface2.d, NAG.cpp:rawSwetParametersPT)
    double epsilon2_;
    double dt_increase_factor_;  // m2 stepIncreaseFactor
    double dt_decrease_factor_;  // m2 stepDecreaseFactor not existent in DEFAULT, using what is in track.m2:77
    double infinity_threshold_; // m2 InfinityThreshold
    double infinity_threshold2_;
    unsigned max_corr_steps_;  // m2 maxCorrSteps (track.m2 param of rawSetParametersPT corresp to max_corr_steps in NAG.cpp)
    unsigned num_successes_before_increase_; // m2 numberSuccessesBeforeIncrease
    double corr_thresh_;
    unsigned anch_num_;
};


bool load_settings(std::string set_file, struct track_settings &settings);


template <unsigned N, typename F>
struct minus_array
{ // Speed critical -----------------------------------------
    static inline void
    multiply_scalar_to_self(F *__restrict__ a, F b)
    {
        for (unsigned i = 0; i < N; ++i, ++a) *a = *a * b;
    }

    static inline void
    negate_self(F * __restrict__ a)
    {
        for (unsigned i = 0; i < N; ++i, ++a) *a = -*a;
    }

    static inline void
    multiply_self(F * __restrict__ a, const F * __restrict__ b)
    {
        for (unsigned int i=0; i < N; ++i,++a,++b) *a *= *b;
    }

    static inline void
    add_to_self(F * __restrict__ a, const F * __restrict__ b)
    {
        for (unsigned int i=0; i < N; ++i,++a,++b) *a += *b;
    }

    static inline void
    add_scalar_to_self(F * __restrict__ a, F b)
    {
        for (unsigned int i=0; i < N; ++i,++a) *a += b;
    }

    static inline void
    copy(const F * __restrict__ a, F * __restrict__ b)
    {
        memcpy(b, a, N*sizeof(double));
    }

    static inline F
    norm2(const F *__restrict__ a)
    {
        F val = 0;
        F const* __restrict__ end = a+N;
        while (a != end) val += std::norm(*a++);
        return val;
    }
};

//Straight line program for evaluation of the Jacobian of the homotopy function, generated in Macaulay2
inline void evaluate_Hxt(const double * x, const double * params, double * y);


//Straight line program for evaluation of the Jacobian of the homotopy function, generated in Macaulay2
inline void evaluate_HxH(const double * x, const double * params, double * y);

//THE FUNCTION RESPONSIBLE FOR HOMOTOPY CONTINUATION TRACKING
int track(const struct track_settings s, const double s_sols[9], const double params[40], double solution[9], int * num_st);

#endif //LEARNING_TRIPLE_POSES_HOMOTOPY_H
