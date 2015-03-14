#ifndef PARTICLE_SYSTEM_CU
#define PARTICLE_SYSTEM_CU

#include "common.h"
#include "Constants.h"
#include "Particle.hpp"

__constant__ float width = 1;
__constant__ float height = 1;
__constant__ float depth = 1;

__device__ float WPoly6(glm::vec3 &pi, glm::vec3 &pj) {
	glm::vec3 r = pi - pj;
	float rLen = glm::length(r);
	if (rLen > H || rLen == 0) {
		return 0;
	}

	return KPOLY * glm::pow((H * H - glm::length2(r)), 3);
}

__device__ glm::vec3 gradWPoly6(glm::vec3 &pi, glm::vec3 &pj) {
	glm::vec3 r = pi - pj;
	float rLen = glm::length(r);
	if (rLen > H || rLen == 0) {
		return glm::vec3(0.0f);
	}

	float coeff = glm::pow((H * H) - (rLen * rLen), 2);
	coeff *= -6 * KPOLY;
	return r * coeff;
}

__device__ glm::vec3 WSpiky(glm::vec3 &pi, glm::vec3 &pj) {
	glm::vec3 r = pi - pj;
	float rLen = glm::length(r);
	if (rLen > H || rLen == 0) {
		return glm::vec3(0.0f);
	}

	float coeff = (H - rLen) * (H - rLen);
	coeff *= SPIKY;
	coeff /= rLen;
	return r * -coeff;
}

__device__ float WAirPotential(glm::vec3 &pi, glm::vec3 &pj) {
	glm::vec3 r = pi - pj;
	float rLen = glm::length(r);
	if (rLen > H || rLen == 0) {
		return 0.0f;
	}

	return 1 - (rLen / H);
}

//Returns density constraint of a particle
__device__ float calcDensityConstraint(Particle &p) {
	float sum = 0.0f;
	/*for (auto &n : neighbors) {
		sum += WPoly6(p.newPos, n->newPos);
	}*/

	return (sum / REST_DENSITY) - 1;
}

//Returns the eta vector that points in the direction of the corrective force
__device__ glm::vec3 eta(Particle &p, float &vorticityMag) {
	glm::vec3 eta = glm::vec3(0.0f);
	/*for (auto &n : p.neighbors) {
		eta += WSpiky(p.newPos, n->newPos) * vorticityMag;
	}*/

	return eta;
}

//Calculates the vorticity force for a particle
__device__ glm::vec3 vorticityForce(Particle &p) {
	//Calculate omega_i
	glm::vec3 omega = glm::vec3(0.0f);
	glm::vec3 velocityDiff;
	glm::vec3 gradient;

	/*for (auto &n : p.neighbors) {
		velocityDiff = n->velocity - p.velocity;
		gradient = WSpiky(p.newPos, n->newPos);
		omega += glm::cross(velocityDiff, gradient);
	}*/

	float omegaLength = glm::length(omega);
	if (omegaLength == 0.0f) {
		//No direction for eta
		return glm::vec3(0.0f);
	}

	glm::vec3 etaVal = eta(p, omegaLength);
	if (etaVal == glm::vec3(0.0f)) {
		//Particle is isolated or net force is 0
		return glm::vec3(0.0f);
	}

	glm::vec3 n = glm::normalize(etaVal);
	if (glm::isinf(n.x) || glm::isinf(n.y) || glm::isinf(n.z)) {
		return glm::vec3(0.0f);
	}

	return (glm::cross(n, omega) * EPSILON_VORTICITY);
}

__device__ float sCorrCalc(Particle &pi, Particle* &pj) {
	//Get Density from WPoly6
	float corr = WPoly6(pi.newPos, pj->newPos) / wQH;
	corr *= corr * corr * corr;
	return -K * corr;
}

__device__ glm::vec3 xsphViscosity(Particle &p) {
	glm::vec3 visc = glm::vec3(0.0f);
	/*for (auto &n : p.neighbors) {
		glm::vec3 velocityDiff = n->velocity - p.velocity;
		velocityDiff *= WPoly6(p.newPos, n->newPos);
		visc += velocityDiff;
	}*/

	return visc * C;
}

__device__ void confineToBox(Particle &p) {
	if (p.newPos.x < 0 || p.newPos.x > width) {
		p.velocity.x = 0;
		if (p.newPos.x < 0) p.newPos.x = 0.001f;
		else p.newPos.x = width - 0.001f;
	}

	if (p.newPos.y < 0 || p.newPos.y > height) {
		p.velocity.y = 0;
		if (p.newPos.y < 0) p.newPos.y = 0.001f;
		else p.newPos.y = height - 0.001f;
	}

	if (p.newPos.z < 0 || p.newPos.z > depth) {
		p.velocity.z = 0;
		if (p.newPos.z < 0) p.newPos.z = 0.001f;
		else p.newPos.z = depth - 0.001f;
	}
}

__global__ void predictPositions() {
	/*for (int i = 0; i < particles.size(); i++) {
		Particle &p = particles.at(i);

		//update velocity vi = vi + dt * fExt
		p.velocity += GRAVITY * deltaT;

		//predict position x* = xi + dt * vi
		p.newPos += p.velocity * deltaT;

		confineToBox(p);
	}*/
}

__global__ void calcLambda() {
	/*for (int i = 0; i < particles.size(); i++) {
		Particle &p = particles.at(i);
		buffer2[i] = lambda(p, p.neighbors);
	}
	float densityConstraint = calcDensityConstraint(p, neighbors);
	glm::vec3 gradientI = glm::vec3(0.0f);
	float sumGradients = 0.0f;
	for (auto &n : neighbors) {
		//Calculate gradient with respect to j
		glm::vec3 gradientJ = WSpiky(p.newPos, n->newPos) / REST_DENSITY;

		//Add magnitude squared to sum
		sumGradients += glm::length2(gradientJ);
		gradientI += gradientJ;
	}

	//Add the particle i gradient magnitude squared to sum
	sumGradients += glm::length2(gradientI);
	return ((-1) * densityConstraint) / (sumGradients + EPSILON_LAMBDA);*/
}

__global__ void calcDeltaP() {
	/*for (int i = 0; i < particles.size(); i++) {
		Particle &p = particles.at(i);
		glm::vec3 deltaP = glm::vec3(0.0f);
		for (auto &n : p.neighbors) {
			float lambdaSum = buffer2[i] + buffer2[n->index];
			float sCorr = sCorrCalc(p, n);
			deltaP += WSpiky(p.newPos, n->newPos) * (lambdaSum + sCorr);
		}

		buffer1[i] = deltaP / REST_DENSITY;
	}*/
}

__global__ void updatePositions() {
	/*for (int i = 0; i < particles.size(); i++) {
		Particle &p = particles.at(i);
		p.newPos += buffer1[i];
	}*/
}

__global__ void updateVelocities() {
	/*for (int i = 0; i < particles.size(); i++) {
		Particle &p = particles.at(i);
		confineToBox(p);

		//set new velocity vi = (x*i - xi) / dt
		p.velocity = (p.newPos - p.oldPos) / deltaT;

		//apply vorticity confinement
		p.velocity += vorticityForce(p) * deltaT;

		//apply XSPH viscosity
		buffer1[i] = xsphViscosity(p);

		//update position xi = x*i
		p.oldPos = p.newPos;
	}*/
}

__global__ void updateXSPHVelocities() {
	/*for (int i = 0; i < particles.size(); i++) {
		Particle &p = particles.at(i);
		p.velocity += buffer1[i] * deltaT;
	}*/
}

void update() {
	//Move wall
	/*if (frameCounter >= 500) {
	//width = (1 - abs(sin((frameCounter - 400) * (deltaT / 1.25f)  * 0.5f * PI)) * 1) + 4;
	t += flag * deltaT / 1.5f;
	if (t >= 1) {
	t = 1;
	flag *= -1;
	} else if (t <= 0) {
	t = 0;
	flag *= -1;
	}

	width = easeInOutQuad(t, 8, -3.0f, 1.5f);
	}
	frameCounter++;*/

	//------------------WATER-----------------
	//Predict positions and update velocity
	predictPositions<<<threads, blockSize>>>();

	//Update neighbors
	//grid.updateCells(particles);
	//setNeighbors();

	//Needs to be after neighbor finding for weighted positions
	updatePositions<<<threads, blockSize>>>();

	for (int pi = 0; pi < PRESSURE_ITERATIONS; pi++) {
		//set lambda
		calcLambda<<<threads, blockSize>>>();

		//calculate deltaP
		calcDeltaP<<<threads, blockSize>>>();

		//update position x*i = x*i + deltaPi
		updatePositions<<<threads, blockSize>>>();
	}

	//Update velocity, apply vorticity confinement, apply xsph viscosity, update position
	updateVelocities<<<threads, blockSize>>>();

	//Set new velocity
	updateXSPHVelocities<<<threads, blockSize>>>();
}

#endif