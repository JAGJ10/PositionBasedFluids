#ifndef DISTANCE_CONSTRAINT_H
#define DISTANCE_CONSTRAINT_H

#include "common.h"

class DistanceConstraint {
public:
	int p1;
	int p2;
	float restLength;
	float stiffness;

	DistanceConstraint();

	DistanceConstraint(int p1, int p2, float restLength, float stiffness) :
		p1(p1), p2(p2), restLength(restLength), stiffness(stiffness) {}
};

#endif