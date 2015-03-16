#ifndef PARTICLE_SYSTEM_CUH
#define PARTICLE_SYSTEM_CUH

#include "common.h"
#include "Particle.hpp"

__constant__ float width = 1;
__constant__ float height = 1;
__constant__ float depth = 1;

__device__ float WPoly6(glm::vec3 &pi, glm::vec3 &pj);
__device__ glm::vec3 gradWPoly6(glm::vec3 &pi, glm::vec3 &pj);
__device__ glm::vec3 WSpiky(glm::vec3 &pi, glm::vec3 &pj);
__device__ float WAirPotential(glm::vec3 &pi, glm::vec3 &pj);
__device__ float calcDensityConstraint(Particle* particles, int* neighbors, int* numNeighbors, int index);
__device__ glm::vec3 eta(Particle* particles, int* neighbors, int* numNeighbors, int index, float &vorticityMag);
__device__ glm::vec3 vorticityForce(Particle* particles, int* neighbors, int* numNeighbors, int index);
__device__ void confineToBox(Particle &p);
__device__ float sCorrCalc(Particle &pi, Particle &pj);
__device__ glm::vec3 xsphViscosity(Particle* particles, int* neighbors, int* numNeighbors, int index);

__global__ void predictPositions(Particle* particles);
__global__ void clearNeighbors(int* neighbors, int* numNeighbors);
__global__ void updateNeighbors(Particle* particles, int* neighbors, int* numNeighbors);
__global__ void calcLambda(Particle* particles, int* neighbors, int* numNeighbors, float* buffer2);
__global__ void calcDeltaP(Particle* particles, int* neighbors, int* numNeighbors, glm::vec3* buffer1, float* buffer2);
__global__ void applyDeltaP(Particle* particles, glm::vec3* buffer1);
__global__ void updateVelocities(Particle* particles, int* neighbors, int* numNeighbors);
__global__ void updateXSPHVelocities(Particle* particles, glm::vec3* buffer1);

void update(Particle* particles, int* neighbors, int* numNeighbors, glm::vec3* buffer1, float* buffer2);
void setVBO(Particle* particles, float* vboPtr);

#endif