#ifndef PARTICLE_SYSTEM_CUH
#define PARTICLE_SYSTEM_CUH

#include "common.h"
#include "ParticleSystem.h"

void update(solver *s);
void setVBO(Particle* particles, float* fluidPositions, float* clothPositions);
void initParams(solver* s, solverParams* sp);
void freeParams(solver* s, solverParams* sp);
#endif