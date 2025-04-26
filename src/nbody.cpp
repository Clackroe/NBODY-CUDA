#include <algorithm>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <omp.h>

#include "Vec.hpp"
#include <Simulation.hpp>

#include "utils.hpp"
#include <Body.hpp>

#include <ostream>
#include <string>
#include <unordered_map>
#include <vector>

#define G 9.8
#define WIDTH 1920
#define HEIGHT 1080

const double DT = 0.07;

std::vector<Body>
CopyBodies(const std::vector<Body>& bodies)
{
    std::vector<Body> out(bodies.size());
    std::vector<Body> bodiesRender(bodies.size());
    for (int i = 0; i < out.size(); i++) {
        out[i] = bodies[i];
    }
    return out;
}

double BenchMark(std::function<double(std::vector<Body>&, double)> calc, std::function<double(std::vector<Body>&, double, int, int)> update, std::vector<Body>& bodies, const std::string& name, int frames)
{

    double seconds = 0;
    for (int i = 0; i < frames; i++) {
        seconds += calc(bodies, G);
        seconds += update(bodies, DT, WIDTH, HEIGHT);
    }
    std::cout << name << " took: " << seconds << " Seconds" << std::endl;
    return seconds;
}

double TotalAccelerationX(const std::vector<Body>& bodies)
{
    double sum = 0;
    for (const auto& b : bodies) {
        sum += b.position.x;
    }
    return sum;
}

std::vector<Body> GenerateBodiesMT(int size)
{
    std::vector<Body> bodies(size);
#pragma omp parallel for
    for (int i = 0; i < bodies.size(); i++) {
        Body b;
        b.position = { generate_random_double(100, WIDTH - 100), generate_random_double(100, HEIGHT - 100) };
        b.velocity = { 0, 0 };
        b.mass = generate_random_double(10, 100);
        bodies[i] = b;
    }
    return bodies;
}

int main()
{
    std::vector<int> bodyCounts = { 1000, 5000, 10000 };
    CudaInit(bodyCounts.back());

    std::string dataFile = "data.csv";
    if (std::filesystem::exists(dataFile)) {
        std::filesystem::remove(dataFile);
    }

    std::ofstream dataStream(dataFile);
    if (!dataStream.is_open()) {
        std::cout << "Failed to open file " << dataFile << std::endl;
    }

    dataStream << "Method";
    for (auto count : bodyCounts) {
        dataStream << "," << count;
    }
    dataStream << std::endl;

    std::unordered_map<std::string, std::vector<double>> times;

    for (auto count : bodyCounts) {
        std::cout << "=========" << std::endl;
        auto bs = GenerateBodiesMT(count);
        double mtTime = BenchMark(CalculateForcesMTReduction, UpdateMT, bs, "MT " + std::to_string(count), 1);
        times["OMP (CPU)"].push_back(mtTime);

        auto bsc = CopyBodies(bs);
        double cudaTime = BenchMark(CalculateForcesCuda, UpdateCuda, bsc, "Cuda " + std::to_string(count), 1);
        times["CUDA (GPU)"].push_back(cudaTime);

        double cudaSpeedUp = mtTime / cudaTime;
        times["CUDA Speedup"].push_back(cudaSpeedUp);

        std::cout << "Cuda Speedup: " << cudaSpeedUp << "x" << std::endl;
    }

    for (auto [method, data] : times) {

        dataStream << method;
        for (auto time : data) {
            dataStream << "," << time;
        }
        dataStream << std::endl;
    }

    dataStream.close();
    CudaShutdown();

    return 0;
}
