#ifndef PARTICLE_SYSTEM_H
#define PARTICLE_SYSTEM_H

#include "common.h"
#include "Cell.hpp"
#include "CellGrid.h"
#include "FoamParticle.hpp"

class ParticleSystem {
public:
	ParticleSystem();
	~ParticleSystem();

	void update();
	std::vector<glm::vec3>& getFluidPositions();
	std::vector<glm::vec4>& getFoamPositions();

private:
	std::vector<Particle> particles;
	std::vector<FoamParticle> foam;

	std::vector<glm::vec3> fluidPositions;
	std::vector<glm::vec4> foamPositions;

	CellGrid grid;

	void confineToBox(FoamParticle &p);
	void updatePositions2();
	void setNeighbors();
	void calcDensities();
	void updateFoam();
	void generateFoam();
	float easeInOutQuad(float t, float b, float c, float d);
};

#endif
