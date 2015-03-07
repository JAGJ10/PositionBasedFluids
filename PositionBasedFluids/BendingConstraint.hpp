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

	BendingConstraint(Particle *p1, Particle *p2, Particle *p3, Particle *p4) : p1(p1), p2(p2), p3(p3) {
		glm::vec3 center = (1 / 3) * (p1->oldPos + p2->oldPos + p3->oldPos);
		restLength = glm::length(p3->oldPos - center);
		w = p1->invMass + p2->invMass + (2 * p3->invMass);
	}
};

#endif