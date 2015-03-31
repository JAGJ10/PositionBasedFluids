#ifndef PARTICLE_SYSTEM_H
#define PARTICLE_SYSTEM_H

#include "common.h"
#include "parameters.h"
#include "DistanceConstraint.hpp"

class ParticleSystem {
public:
	bool running;
	solver* s;

	std::vector<DistanceConstraint> tempdConstraints;

	ParticleSystem();
	~ParticleSystem();

	void initialize(tempSolver &tp, solverParams &tempParams);
	void updateWrapper();
	void getPositionsWrapper(float* positionsPtr);

private:
	void updatePositions2();
	int getIndex(float i, float j);
	void updateFoam();
	float easeInOutQuad(float t, float b, float c, float d);

	#define cudaCheck(x) { cudaError_t err = x; if (err != cudaSuccess) { printf("Cuda error: %d in %s at %s:%d\n", err, #x, __FILE__, __LINE__); assert(0); } }
};

#endif
