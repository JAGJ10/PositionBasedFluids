#ifndef CONSTANTS_H
#define CONSTANTS_H

#include "common.h"

//---------------------Fluid Constants----------------------
static const int PRESSURE_ITERATIONS = 4;
static const int numParticles = 10240;
static const int MAX_NEIGHBORS = 50;
static const int MAX_PARTICLES = 10;
static const int blockSize = 128;
static const dim3 dims = int(numParticles / blockSize);

static const int w = 40;
static const int h = 40;
static const int d = 40;
static const int gridSize = 64000;
static const dim3 gridDims = int(gridSize / blockSize);

#define NUM_PARTICLES_C numParticles
#define MAX_NEIGHBORS_C MAX_NEIGHBORS
#define MAX_PARTICLES_C MAX_PARTICLES
#define GRID_SIZE_C 64000

#define deltaT 0.0083f
#define PI 3.14159265358979323846f
#define GRAVITY glm::vec3(0, -9.8f, 0)
#define H 0.1f
#define KPOLY (315.0f / (64.0f * PI * glm::pow(H, 9)))
#define SPIKY (45.0f / (PI * glm::pow(H, 6)))
#define REST_DENSITY 6378.0f
#define EPSILON_LAMBDA 600.0f
#define EPSILON_VORTICITY 0.0001f
#define C 0.01f
#define K 0.00001f
#define deltaQMag 0.3f * H
#define wQH (KPOLY * glm::pow((H * H - deltaQMag * deltaQMag), 3))
#define lifetime 1.0f

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

#endif