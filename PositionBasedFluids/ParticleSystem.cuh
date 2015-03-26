#ifndef PARTICLE_SYSTEM_CUH
#define PARTICLE_SYSTEM_CUH

#include "common.h"
#include "ParticleSystem.h"

void update(Buffers *p);
void setVBO(Particle* particles, float* vboPtr);

#endif