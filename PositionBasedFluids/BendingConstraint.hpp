#ifndef BENDING_CONSTRAINT_H
#define BENDING_CONSTRAINT_H

#include "common.h"
#include "Particle.hpp"

class BendingConstraint {
public:
	Particle *p1;
	Particle *p2;
	Particle *p3;
	float restLength;
	float w;

	BendingConstraint(Particle *p1, Particle *p2, Particle *p3) : p1(p1), p2(p2), p3(p3) {
		glm::vec3 center = (1.0f / 3.0f) * (p1->oldPos + p2->oldPos + p3->oldPos);
		restLength = glm::length(p2->oldPos - center);
		w = p1->invMass + p3->invMass + (2 * p2->invMass);
	}
};

#endif