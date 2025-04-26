#ifndef SIMULATION_HPP
#define SIMULATION_HPP

#include "Body.hpp"
#include <vector>

double CalculateForcesCuda(std::vector<Body>& bodies, double G);
double UpdateCuda(std::vector<Body>& bodies, double deltaTime, int width, int height);

double CalculateForcesMTReduction(std::vector<Body>& bodies, double G);
double UpdateMT(std::vector<Body>& bodies, double deltaTime, int width, int height);

void CudaInit(int maxBodies);
void CudaShutdown();

#endif // SIMULATION_HPP
