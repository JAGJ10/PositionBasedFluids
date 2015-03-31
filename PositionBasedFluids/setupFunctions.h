#ifndef SETUP_FUNCTIONS_H
#define SETUP_FUNCTIONS_H

#include "common.h"
#include "parameters.h"

void createParticleGrid(tempSolver* s, solverParams* sp, float3 lower, int3 dims, float radius) {
	for (int x = 0; x < dims.x; x++) {
		for (int y = 0; y < dims.y; y++) {
			for (int z = 0; z < dims.z; z++) {
				float3 pos = lower + make_float3(float(x), float(y), float(z)) * radius;
				s->positions.push_back(make_float4(pos, 1.0f));
				s->velocities.push_back(make_float3(0));
				s->phases.push_back(0);
			}
		}
	}
}

#endif