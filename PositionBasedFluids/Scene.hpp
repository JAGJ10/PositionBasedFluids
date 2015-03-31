#ifndef SCENE_H
#define SCENE_H

#include "common.h"
#include "parameters.h"
#include "setupFunctions.h"

class Scene {
public:
	Scene(std::string name) : name(name) {}
	virtual void init(tempSolver* tp, solverParams* sp) = 0;
	
private:
	std::string name;

};

class DamBreak : public Scene {
public:
	DamBreak(std::string name) : Scene(name) {}

	virtual void init(tempSolver* tp, solverParams* sp) {
		const float radius = 0.1f;
		const float restDistance = radius * 0.5f;
		float3 lower = make_float3(0.0f, 0.1f, 0.0f);
		int3 dims = make_int3(40, 40, 40);
		createParticleGrid(tp, sp, lower, dims, restDistance);
		
		sp->radius = radius;
		sp->restDistance = restDistance;
		sp->numIterations = 4;
		sp->numDiffuse = 1024 * 1024;
		sp->numParticles = int(tp->positions.size());
		sp->gravity = make_float3(0, -9.8f, 0);
		sp->bounds = make_float3(dims) * radius;
		sp->gridWidth = int(sp->bounds.x / radius);
		sp->gridHeight = int(sp->bounds.y / radius);
		sp->gridDepth = int(sp->bounds.z / radius);
		sp->gridSize = sp->gridWidth * sp->gridHeight * sp->gridDepth;
		sp->MAX_CONTACTS = 10;
		sp->MAX_NEIGHBORS = 50;
		sp->MAX_PARTICLES = 50;
		sp->restDensity = 6378.0f;
		sp->lambdaEps = 600.0f;
		sp->vorticityEps = 0.0001f;
		sp->C = 0.01f;
		sp->K = 0.00001f;
		sp->KPOLY = 315.0f / (64.0f * PI * pow(radius, 9));
		sp->SPIKY = 45.0f / (PI * pow(radius, 6));
		sp->dqMag = 0.3f * radius;
		sp->wQH = sp->KPOLY * pow((radius * radius - sp->dqMag * sp->dqMag), 3);
	}
};

class FluidCloth : public Scene {
public:

};

class Lighthouse : public Scene {
public:

};

#endif