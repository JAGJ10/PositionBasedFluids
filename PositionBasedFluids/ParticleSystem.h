#ifndef PARTICLE_SYSTEM_H
#define PARTICLE_SYSTEM_H

#include "common.h"
#include "Constants.h"
#include "FoamParticle.hpp"
#include "Particle.hpp"

class ParticleSystem {
public:
	Particle* particles;
	Particle* tempParticles;
	FoamParticle* foamParticles;
	//int* freeList;
	int* neighbors;
	int* numNeighbors;
	int* gridCells;
	int* gridCounters;
	glm::vec3* buffer0;
	glm::vec3* buffer1;
	float* densities;
	float* buffer3;

	ParticleSystem();
	~ParticleSystem();

	void updateWrapper();
	void setVBOWrapper(float* vboPtr);

private:
	void updatePositions2();
	void updateFoam();
	float easeInOutQuad(float t, float b, float c, float d);

	#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
	inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
		if (code != cudaSuccess) {
			fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
			if (abort) exit(code);
		}
	}
};

#endif
