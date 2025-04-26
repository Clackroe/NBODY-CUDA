#include <vector>

#define USE_CUDA
#include <Body.hpp>
#include <Vec.hpp>

#include <stdio.h>

#define C_CHECK(ret)                                                                     \
    {                                                                                    \
        auto err = ret;                                                                  \
        if (err != cudaSuccess) {                                                        \
            printf("Cuda err in %s:%i %s", __FILE__, __LINE__, cudaGetErrorString(err)); \
        }                                                                                \
    }

__global__ void CalcForcesKernel(Body* bodies, Vec2* accels, int n, double G)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n)
        return;
    Vec2 acc = { 0, 0 };

    for (int j = 0; j < n; j++) {
        if (i == j) {
            continue;
        }
        Body b1 = bodies[i];
        Body b2 = bodies[j];
        double force = Force(b1, b2, G);
        Vec2 dir = Direction(b1.position, b2.position);
        double acc1 = force / b1.mass;
        acc = add(acc, scale(dir, acc1));
    }

    accels[i] = acc;
}

__global__ void UpdateKernel(Body* bodies, Vec2* accels, double deltaTime, int width, int height, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    Body& b = bodies[i];

    b.velocity = add(b.velocity, accels[i]);

    if (b.position.x < 0 || b.position.x > width) {
        b.velocity.x *= -1;
    }
    if (b.position.y < 0 || b.position.y > height) {
        b.velocity.y *= -1;
    }

    b.position = add(b.position, scale(b.velocity, deltaTime));
}

Body* g_DeviceBodies;
Vec2* g_DeviceAccels;

void CudaInit(int maxBodies)
{
    C_CHECK(cudaMalloc(&g_DeviceBodies, maxBodies * sizeof(Body)));
    C_CHECK(cudaMalloc(&g_DeviceAccels, maxBodies * sizeof(Vec2)));
}

void CudaShutdown()
{
    C_CHECK(cudaFree(g_DeviceBodies));
    C_CHECK(cudaFree(g_DeviceAccels));
}

double CalculateForcesCuda(std::vector<Body>& bodies, double G)
{

    cudaEvent_t start, end;
    int n = bodies.size();
    C_CHECK(cudaMemcpy(g_DeviceBodies, bodies.data(), n * sizeof(Body), cudaMemcpyHostToDevice));
    // Don't need to set accels since they are set to zero initially during the simulation

    // Dont want to include the MemCPY in the final time

    C_CHECK(cudaEventCreate(&start));
    C_CHECK(cudaEventCreate(&end));

    C_CHECK(cudaEventRecord(start));

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    CalcForcesKernel<<<numBlocks, blockSize>>>(g_DeviceBodies, g_DeviceAccels, n, G);

    C_CHECK(cudaEventRecord(end));
    C_CHECK(cudaDeviceSynchronize());

    float time;
    C_CHECK(cudaEventElapsedTime(&time, start, end));
    double seconds = time / 1000;

    C_CHECK(cudaEventDestroy(start));
    C_CHECK(cudaEventDestroy(end));

    return seconds;
}

double UpdateCuda(std::vector<Body>& bodies, double deltaTime, int width, int height)
{

    cudaEvent_t start, end;

    C_CHECK(cudaEventCreate(&start));
    C_CHECK(cudaEventCreate(&end));

    C_CHECK(cudaEventRecord(start));

    int n = bodies.size();

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    UpdateKernel<<<numBlocks, blockSize>>>(g_DeviceBodies, g_DeviceAccels, deltaTime, width, height, n);

    C_CHECK(cudaEventRecord(end));
    C_CHECK(cudaDeviceSynchronize());

    float time;
    C_CHECK(cudaEventElapsedTime(&time, start, end));
    double seconds = time / 1000;

    // Not including memcpy in the timings
    C_CHECK(cudaMemcpy(bodies.data(), g_DeviceBodies, n * sizeof(Body), cudaMemcpyDeviceToHost));

    C_CHECK(cudaEventDestroy(start));
    C_CHECK(cudaEventDestroy(end));

    return seconds;
}
