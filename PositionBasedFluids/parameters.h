#ifndef PARAMETERS_H
#define PARAMETERS_H

#include "common.h"

struct tempSolver {
	std::vector<float4> positions;
	std::vector<float3> velocities;
	std::vector<int> phases;
};

struct solver {
	float4* oldPos;
	float4* newPos;
	float3* velocities;
	float* densities;
	int* phases;
	//int* indices;

	//float4* diffusePos;
	//float3* diffuseVelocities;

	int* neighbors;
	int* numNeighbors;
	int* gridCells;
	int* gridCounters;
	int* contacts;
	int* numContacts;

	float3* deltaPs;

	float* buffer0;

	//DistanceConstraint* dConstraints;
	//int numConstraints;
};

struct solverParams {
public:
	int MAX_NEIGHBORS;
	int MAX_PARTICLES;
	int MAX_CONTACTS;
	int gridWidth, gridHeight, gridDepth;

	int numParticles;
	int numDiffuse;
	float3 gravity;
	float3 bounds;
	int numIterations;
	float radius;
	float restDistance;
	//float sor;
	//float vorticity;
	int gridSize;

	float KPOLY;
	float SPIKY;
	float restDensity;
	float lambdaEps;
	float vorticityEps;
	float C;
	float K;
	float dqMag;
	float wQH;
};

#endif