#ifndef DISTANCE_CONSTRAINT_H
#define DISTANCE_CONSTRAINT_H

#include "common.h"
#include "Particle.hpp"

class DistanceConstraint {
public:
	int p1;
	int p2;
	float restLength;

	DistanceConstraint(int p1, int p2, float restLength) : p1(p1), p2(p2), restLength(restLength) {}
};

#endif