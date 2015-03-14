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
__device__ float calcDensityConstraint(Particle &p, std::vector<Particle*> &neighbors);
__device__ glm::vec3 eta(Particle &p, float &vorticityMag);
__device__ glm::vec3 vorticityForce(Particle &p);
__device__ void confineToBox(Particle &p);
__device__ float sCorrCalc(Particle &pi, Particle* &pj);
__device__ glm::vec3 xsphViscosity(Particle &p);

__global__ void predictPositions();
__global__ void calcLambda();
__global__ void calcDeltaP();
__global__ void updatePositions();
__global__ void updateVelocities();
__global__ void updateXSPHVelocities();

#endif