#ifndef PARTICLE_SYSTEM_H
#define PARTICLE_SYSTEM_H

#include "common.h"
#include "Constants.h"
#include "FoamParticle.hpp"
#include "Particle.hpp"
#include "DistanceConstraint.hpp"

struct Buffers {
	Particle* particles;
	DistanceConstraint* dConstraints;
	int* neighbors;
	int* numNeighbors;
	int* gridCells;
	int* gridCounters;
	glm::vec3* deltaPs;
	glm::vec3* buffer1;
	float* densities;
	float* buffer3;
	int numConstraints;
};

class ParticleSystem {
public:
	Buffers* p;
	Particle* tempParticles;
	std::vector<DistanceConstraint> tempdConstraints;
	FoamParticle* foamParticles;

	ParticleSystem();
	~ParticleSystem();

	void updateWrapper();
	void setVBOWrapper(float* vboPtr);

private:
	void updatePositions2();
	int getIndex(float i, float j);
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
