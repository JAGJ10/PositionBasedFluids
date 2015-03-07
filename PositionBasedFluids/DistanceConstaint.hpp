#ifndef DISTANCE_CONSTRAINT_H
#define DISTANCE_CONSTRAINT_H

#include "common.h"
#include "Particle.hpp"

class DistanceConstraint {
public:
	Particle *p1;
	Particle *p2;
	float restLength;
	float k;

	DistanceConstraint(Particle *p1, Particle *p2, float k) : p1(p1), p2(p2), k(k) {
		restLength = glm::length(p1->oldPos - p2->oldPos);
	}
};

#endif