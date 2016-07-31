#ifndef PARAMETERS_H
#define PARAMETERS_H

#include "common.h"
#include "Mesh.h"

struct tempSolver {
	std::vector<float4> positions;
	std::vector<float3> velocities;
	std::vector<float> boundaryPsi;
	std::vector<int> phases;

	std::vector<float4> diffusePos;
	std::vector<float3> diffuseVelocities;

	std::vector<int> clothIndices;
	std::vector<float> restLengths;
	std::vector<float> stiffness;
	std::vector<int> triangles;

	std::vector<Mesh> meshes;
};

struct solver {
	float4* oldPos;
	float4* newPos;
	float3* velocities;
	float3* normals;
	float* densities;
	float* boundaryPsi;
	int* phases;

	float4* diffusePos;
	float3* diffuseVelocities;

	int* clothIndices;
	float* restLengths;
	float* stiffness;

	int* neighbors;
	int* numNeighbors;
	int* gridCells;
	int* gridCounters;
	int* contacts;
	int* numContacts;

	float3* deltaPs;

	float* buffer0;
};

struct solverParams {
public:
	int maxNeighbors;
	int maxParticles;
	int maxContacts;
	int gridWidth, gridHeight, gridDepth;
	int gridSize;

	int fluidOffset;
	int numParticles;
	int numDiffuse;

	float3 gravity;
	float3 bounds;

	int numIterations;
	float radius;
	float restDistance;
	float samplingDensity;
	float surfaceTension;
	//float sor;
	//float vorticity;

	float KPOLY;
	float SPIKY;
	float tension;
	float restDensity;
	float lambdaEps;
	float vorticityEps;
	float C;
	float K;
	float dqMag;
	float wQH;
};

#endif