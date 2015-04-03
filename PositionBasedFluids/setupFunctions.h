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

int getIndex(int x, int y, int dx) {
	return y * dx + x;
}

void addConstraint(tempSolver* s, solverParams* sp, int p1, int p2, float stiffness) {
	s->clothIndices.push_back(p1);
	s->clothIndices.push_back(p2);
	s->restLengths.push_back(length(make_float3(s->positions[p1] - s->positions[p2])));
	s->stiffness.push_back(stiffness);
}

void createCloth(tempSolver* s, solverParams* sp, float3 lower, int3 dims, float radius, int phase, float stretch, float bend, float shear, float invMass) {
	//Create grid of particles and add triangles
	for (int z = 0; z < dims.z; z++) {
		for (int y = 0; y < dims.y; y++) {
			for (int x = 0; x < dims.x; x++) {
				float3 pos = lower + make_float3(float(x), float(y), float(z)) * radius;
				s->positions.push_back(make_float4(pos, invMass));
				s->velocities.push_back(make_float3(0));
				s->phases.push_back(phase);

				if (x > 0 && z > 0) {
					s->triangles.push_back(getIndex(x - 1, z - 1, dims.x));
					s->triangles.push_back(getIndex(x, z - 1, dims.x));
					s->triangles.push_back(getIndex(x, z, dims.x));

					s->triangles.push_back(getIndex(x - 1, z - 1, dims.x));
					s->triangles.push_back(getIndex(x, z, dims.x));
					s->triangles.push_back(getIndex(x - 1, z, dims.x));
				}
			}
		}
	}

	//Horizontal constraints
	for (int j = 0; j < dims.z; j++) {
		for (int i = 0; i < dims.x; i++) {
			int i0 = getIndex(i, j, dims.x);
			if (i > 0) {
				int i1 = j * dims.x + i - 1;
				addConstraint(s, sp, i0, i1, stretch);
			}

			if (i > 1) {
				int i2 = j * dims.x + i - 2;
				addConstraint(s, sp, i0, i2, bend);
			}

			if (j > 0 && i < dims.x - 1) {
				int iDiag = (j - 1) * dims.x + i + 1;
				addConstraint(s, sp, i0, iDiag, shear);
			}

			if (j > 0 && i > 0) {
				int iDiag = (j - 1) * dims.x + i - 1;
				addConstraint(s, sp, i0, iDiag, shear);
			}
		}
	}

	//Vertical constraints
	for (int i = 0; i < dims.x; i++) {
		for (int j = 0; j < dims.z; j++) {
			int i0 = getIndex(i, j, dims.x);
			if (j > 0) {
				int i1 = (j - 1) * dims.x + i;
				addConstraint(s, sp, i0, i1, stretch);
			}

			if (j > 1) {
				int i2 = (j - 2) * dims.x + i;
				addConstraint(s, sp, i0, i2, bend);
			}
		}
	}
}
#endif