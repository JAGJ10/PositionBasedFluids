#ifndef PARTICLE_SYSTEM_H
#define PARTICLE_SYSTEM_H

#include "common.h"
#include "parameters.h"

class ParticleSystem {
public:
	bool running;
	bool moveWall;
	solver* s;

	ParticleSystem();
	~ParticleSystem();

	void initialize(tempSolver &tp, solverParams &tempParams);
	void updateWrapper(solverParams &tempParams);
	void getPositions(float* positionsPtr, int numParticles);
	void getDiffuse(float* diffusePosPtr, float* diffuseVelPtr, int numDiffuse);

private:
	int getIndex(float i, float j);
	float easeInOutQuad(float t, float b, float c, float d);

	#define cudaCheck(x) { cudaError_t err = x; if (err != cudaSuccess) { printf("Cuda error: %d in %s at %s:%d\n", err, #x, __FILE__, __LINE__); assert(0); } }
};

#endif
