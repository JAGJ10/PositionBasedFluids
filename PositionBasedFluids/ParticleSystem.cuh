#ifndef PARTICLE_SYSTEM_CUH
#define PARTICLE_SYSTEM_CUH

#include "common.h"

void update(solver *s, solverParams *sp);
void getPositions(float4* oldPos, float* positionsPtr);
void getDiffuse(float4* diffusePos, float3* diffuseVelocities, float* diffusePosPtr, float* diffuseVelPtr);
void setParams(solverParams *tempParams);
#endif