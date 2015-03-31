#ifndef PARTICLE_SYSTEM_CUH
#define PARTICLE_SYSTEM_CUH

#include "common.h"

void update(solver *s);
void getPositions(float4* oldPos, float* positionsPtr);
void setParams(solverParams *tempParams);
#endif