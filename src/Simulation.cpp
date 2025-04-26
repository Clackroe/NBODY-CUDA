#include "Body.hpp"
#include <Simulation.hpp>
#include <Vec.hpp>
#include <chrono>
#include <cstdio>
#include <omp.h>

double CalculateForcesMTReduction(std::vector<Body>& bodies, double G)
{

    const auto start(std::chrono::steady_clock::now());

    const int n = bodies.size();
    std::vector<Vec2> accelerations(n, { 0, 0 });

#pragma omp parallel
    {
        std::vector<Vec2> local_accel(n, { 0, 0 });

#pragma omp for schedule(dynamic)
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                const Body& b1 = bodies[i];
                const Body& b2 = bodies[j];
                double force = Force(b1, b2, G);
                Vec2 dir = Direction(b1.position, b2.position);

                double acc1 = force / b1.mass;

                Vec2 acc_vec1 = scale(dir, acc1);

                local_accel[i] = add(local_accel[i], acc_vec1);
            }
        }

#pragma omp critical
        {
            for (int i = 0; i < n; i++) {
                accelerations[i] = add(accelerations[i], local_accel[i]);
            }
        }
    }

#pragma omp parallel for
    for (int i = 0; i < n; i++) {
        bodies[i].velocity = add(bodies[i].velocity, accelerations[i]);
    }
    const auto end(std::chrono::steady_clock::now());
    std::chrono::duration<double> seconds { end - start };

    return seconds.count();
}

double UpdateMT(std::vector<Body>& bodies, double deltaTime, int width, int height)
{

    const auto start(std::chrono::steady_clock::now());

#pragma omp parallel for
    for (int i = 0; i < bodies.size(); i++) {
        Body& b = bodies[i];
        if (b.position.x < 0 || b.position.x > width) {
            b.velocity.x *= -1;
        }
        if (b.position.y < 0 || b.position.y > height) {
            b.velocity.y *= -1;
        }

        b.position = add(b.position, scale(b.velocity, deltaTime));
    }
    const auto end(std::chrono::steady_clock::now());
    std::chrono::duration<double> seconds { end - start };

    return seconds.count();
}
