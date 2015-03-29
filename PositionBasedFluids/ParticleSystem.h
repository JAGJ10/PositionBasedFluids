#ifndef PARTICLE_SYSTEM_H
#define PARTICLE_SYSTEM_H

#include "common.h"
#include "parameters.h"
#include "FoamParticle.hpp"
#include "Particle.hpp"
#include "DistanceConstraint.hpp"

class ParticleSystem {
public:
	solver* s;
	solverParams* sp;

	std::vector<DistanceConstraint> tempdConstraints;

	ParticleSystem();
	~ParticleSystem();

	void updateWrapper();
	void setVBOWrapper(float* fluidPositions, float* clothPositions);

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

	#define cudaCheck(x) { cudaError_t err = x; if (err != cudaSuccess) { printf("Cuda error: %d in %s at %s:%d\n", err, #x, __FILE__, __LINE__); assert(0); } }
};

#endif
