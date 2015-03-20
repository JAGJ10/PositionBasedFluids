#ifndef PARTICLE_SYSTEM_CUH
#define PARTICLE_SYSTEM_CUH

#include "common.h"
#include "Particle.hpp"
#include "FoamParticle.hpp"

void update(Particle* particles, FoamParticle* foamParticles, int* neighbors, int* numNeighbors, int* gridCells, int* gridCounters, glm::vec3* buffer0, glm::vec3* buffer1, float* buffer2, float* buffer3);
void setVBO(Particle* particles, float* vboPtr);

#endif