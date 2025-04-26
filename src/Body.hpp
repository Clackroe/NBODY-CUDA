#ifndef BODY_HPP
#define BODY_HPP
#include <Vec.hpp>

#ifdef USE_CUDA
#include <math.h>
#define VIS __host__ __device__
#else
#define VIS
#endif

struct Body {
    Vec2 position;
    Vec2 velocity;
    double mass;
};

VIS inline double Force(const Body& b1, const Body& b2, double G)
{
    static const double eps = 0.1;
    double distsq = distSqrd(b1.position, b2.position) + eps;

    return b1.mass * b2.mass * G / distsq;
};

#endif // BODY_HPP
