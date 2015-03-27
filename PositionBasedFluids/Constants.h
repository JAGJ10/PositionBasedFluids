#ifndef CONSTANTS_H
#define CONSTANTS_H

#include "common.h"

//---------------------Fluid Constants----------------------
static const int PRESSURE_ITERATIONS = 4;
static const int SOLVER_ITERATIONS = 4;
static const int blockSize = 128;

#define gridWidth 50
#define gridHeight 50
#define gridDepth 30

static const int gridSize = gridWidth * gridHeight * gridDepth;

#define NUM_PARTICLES 1152
#define NUM_FOAM 2000000
#define MAX_NEIGHBORS 50
#define MAX_PARTICLES 50
#define MAX_CONTACTS 10
#define GRID_SIZE gridSize

static const dim3 dims = int(NUM_PARTICLES / blockSize);
static const dim3 gridDims = int(ceil(gridSize / blockSize));

#define deltaT 0.0083f
#define PI 3.14159265358979323846f
#define GRAVITY glm::vec3(0, -9.8f, 0)
#define H 0.1f
#define KPOLY 1566681471.06084471147494f //(315.0f / (64.0f * PI * glm::pow(H, 9)))
#define SPIKY 14323944.878270580219199538f //(45.0f / (PI * glm::pow(H, 6)))
#define REST_DENSITY 6378.0f
#define EPSILON_LAMBDA 600.0f
#define EPSILON_VORTICITY 0.0001f
#define C 0.01f
#define K 0.00001f
#define deltaQMag 0.3f * H
#define wQH 1180.60572282879181f //(KPOLY * glm::pow((H * H - deltaQMag * deltaQMag), 3))
#define lifetime 1.0f

#endif