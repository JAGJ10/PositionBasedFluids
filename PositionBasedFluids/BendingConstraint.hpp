#ifndef BENDING_CONSTRAINT_H
#define BENDING_CONSTRAINT_H

#include "common.h"
#include "Particle.hpp"

class BendingConstraint {
public:
	int p1;
	int p2;
	int p3;
	float restLength;
	float w;

	BendingConstraint(int p1, int p2, int p3, float restLength, float w) : p1(p1), p2(p2), p3(p3), restLength(restLength), w(w) {
//		glm::vec3 center = (1.0f / 3.0f) * (p1->oldPos + p2->oldPos + p3->oldPos);
//		restLength = glm::length(p2->oldPos - center);
//		w = p1->invMass + p3->invMass + (2 * p2->invMass);
	}
};

#endif