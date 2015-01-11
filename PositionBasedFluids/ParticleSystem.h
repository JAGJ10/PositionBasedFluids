#ifndef PARTICLE_SYSTEM_H
#define PARTICLE_SYSTEM_H

#include "common.h"
#include "Cell.hpp"
#include "CellGrid.h"

class ParticleSystem {
public:
	ParticleSystem();
	~ParticleSystem();

	std::vector<glm::vec3> getPositions();
	void update();

private:
	std::vector<Particle> particles;
	CellGrid grid;

	void applyGravity(Particle &p);
	float WPoly6(glm::vec3 &pi, glm::vec3 &pj);
	glm::vec3 WSpiky(glm::vec3 &pi, glm::vec3 &pj);
	glm::vec3 WViscosity(glm::vec3 &pi, glm::vec3 &pj);
	float lambda(Particle &p, std::vector<Particle*> &neighbors);
	float calcDensityConstraint(Particle &p, std::vector<Particle*> &neighbors);
	glm::vec3 vorticity(Particle &p);
	glm::vec3 eta(Particle &p, float vorticityMag);
	glm::vec3 vorticityForce(Particle &p);
	void imposeConstraints(Particle &p);
	float clampedConstraint(float x, float max);
	float sCorrCalc(Particle &pi, Particle* &pj);
	glm::vec3 xsphViscosity(Particle &p);
	bool outOfRange(float x, float y, float z);
	void initializeGrid();
};

#endif
