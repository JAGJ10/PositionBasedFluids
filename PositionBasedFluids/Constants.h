#ifndef CONSTANTS_H
#define CONSTANTS_H

#include "common.h"

//---------------------Fluid Constants----------------------
#define gridWidth 50
#define gridHeight 50
#define gridDepth 30

static const int gridSize = gridWidth * gridHeight * gridDepth;

#define GRID_SIZE gridSize

//static const dim3 dims = int(NUM_PARTICLES / blockSize);
//static const dim3 gridDims = int(ceil(gridSize / blockSize));

#define PI 3.14159265358979323846f
//#define GRAVITY glm::vec3(0, -9.8f, 0)
//#define H 0.1f
//#define KPOLY (315.0f / (64.0f * PI * glm::pow(H, 9)))
//#define SPIKY (45.0f / (PI * glm::pow(H, 6)))
//#define REST_DENSITY 6378.0f
//#define EPSILON_LAMBDA 600.0f
//#define EPSILON_VORTICITY 0.0001f
//#define C 0.01f
//#define K 0.00001f
//#define deltaQMag 0.3f * H
//#define wQH (KPOLY * glm::pow((H * H - deltaQMag * deltaQMag), 3))
#define lifetime 1.0f

#endif