#ifndef PARTICLE_SYSTEM_H
#define PARTICLE_SYSTEM_H

#include "common.h"
#include "Constants.h"
#include "FoamParticle.hpp"
#include "Particle.hpp"

class ParticleSystem {
public:
	Particle* particles;
	int* neighbors;
	int* numNeighbors;
	glm::vec3* buffer1;
	float* buffer2;

	ParticleSystem();
	~ParticleSystem();

	void updateWrapper();
	std::vector<glm::vec3>& getFluidPositions();
	std::vector<glm::vec4>& getFoamPositions();

private:
	std::vector<FoamParticle> foam;

	std::vector<glm::vec3> fluidPositions;
	std::vector<glm::vec4> foamPositions;

	void confineToBox(FoamParticle &p);
	void updatePositions2();
	void setNeighbors();
	void calcDensities();
	void updateFoam();
	void generateFoam();
	float easeInOutQuad(float t, float b, float c, float d);
};

#endif
