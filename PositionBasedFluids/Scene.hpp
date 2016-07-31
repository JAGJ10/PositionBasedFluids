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
		int3 dims = make_int3(68, 48, 88);
		createParticleGrid(tp, sp, lower, dims, restDistance);
		
		sp->radius = radius;
		sp->restDistance = restDistance;
		sp->numIterations = 4;
		sp->numDiffuse = 1024 * 2048;
		sp->numParticles = int(tp->positions.size());
		sp->fluidOffset = 0;
		sp->gravity = make_float3(0, -9.8f, 0);
		sp->bounds = make_float3(dims) * radius;
		sp->gridWidth = int(sp->bounds.x / radius);
		sp->gridHeight = int(sp->bounds.y / radius);
		sp->gridDepth = int(sp->bounds.z / radius);
		sp->gridSize = sp->gridWidth * sp->gridHeight * sp->gridDepth;
		sp->maxContacts = 10;
		sp->maxNeighbors = 50;
		sp->maxParticles = 50;
		sp->restDensity = 6378.0f;
		sp->lambdaEps = 600.0f;
		sp->vorticityEps = 0.0001f;
		sp->C = 0.01f;
		sp->K = 0.00001f;
		sp->KPOLY = 315.0f / (64.0f * PI * pow(radius, 9));
		sp->SPIKY = 45.0f / (PI * pow(radius, 6));

		tp->diffusePos.resize(sp->numDiffuse);
		tp->diffuseVelocities.resize(sp->numDiffuse);
	}
};

class BunnyBath : public Scene {
public:
	BunnyBath(std::string name) : Scene(name) {}

	virtual void init(tempSolver* tp, solverParams* sp) {
		const float radius = 0.1f;
		const float restDistance = radius * 0.5f;
		float3 lower = make_float3(0, 0.f, 1);
		float3 lower1 = make_float3(4.0f, -0.5f, 1.0f);
		int3 dims = make_int3(50, 60, 40);
		tp->meshes.resize(2);
		loadMeshes("meshes/bunny_closed_fixed.cobj", tp->meshes, lower1, 1.0f, 0);
		//loadMeshes("meshes/cube.cobj", tp->meshes, lower1, 2.0f, 0);
		loadMeshes("meshes/plane.cobj", tp->meshes, make_float3(0.0f), 50.0f, 1);
		createParticleShape("meshes/bunny_closed_fixed.obj", tp, lower1, true);
		createParticleShape("meshes/cube.obj", tp, lower1, true);
		/*for (int x = 0; x < 5; x++) {
			for (int y = 0; y < 60; y++) {
				for (int z = 0; z < 25; z++) {
					float3 pos = lower1 + make_float3(float(x), float(y), float(z)) * restDistance;
					tp->positions.push_back(make_float4(pos, 1.0f));
					tp->velocities.push_back(make_float3(0));
					tp->phases.push_back(2);
				}
			}
		}*/
		sp->fluidOffset = int(tp->positions.size());
		std::cout << "Num boundary particles: " << (int)sp->fluidOffset << std::endl;
		createParticleGrid(tp, sp, lower, dims, restDistance);
		std::cout << "Num fluid particles: " << (int)tp->positions.size() - sp->fluidOffset << std::endl;
		sp->radius = radius;
		sp->restDistance = restDistance;
		sp->samplingDensity = 2.0f;
		sp->numIterations = 4;
		sp->numDiffuse = 1024 * 500;
		sp->numParticles = int(tp->positions.size());

		sp->gravity = make_float3(0, -4.9f, 0);
		sp->bounds = make_float3(dims) * radius;
		sp->bounds.x *= 2;
		sp->gridWidth = int(sp->bounds.x / radius);
		sp->gridHeight = int(sp->bounds.y / radius);
		sp->gridDepth = int(sp->bounds.z / radius);
		sp->gridSize = sp->gridWidth * sp->gridHeight * sp->gridDepth;
		sp->maxContacts = 10;
		sp->maxNeighbors = 50;
		sp->maxParticles = 50;
		sp->restDensity = 6378.0f;
		sp->lambdaEps = 600.0f;
		sp->vorticityEps = 0.0001f;
		sp->C = 0.01f;
		sp->KPOLY = 315.0f / (64.0f * PI * pow(radius, 9));
		sp->SPIKY = 45.0f / (PI * pow(radius, 6));
		sp->tension = 32.0f / (PI * pow(radius, 9));
		sp->surfaceTension = 0.005f;

		tp->diffusePos.resize(sp->numDiffuse);
		tp->diffuseVelocities.resize(sp->numDiffuse);
	}
};

#endif