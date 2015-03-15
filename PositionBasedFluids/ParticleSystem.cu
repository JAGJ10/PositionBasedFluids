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
__device__ float calcDensityConstraint(Particle* particles, int* neighbors, int* numNeighbors, int index) {
	float sum = 0.0f;
	for (int i = 0; i < numNeighbors[index]; i++) {
		sum += WPoly6(particles[index].newPos, particles[neighbors[index + i]].newPos);
	}

	return (sum / REST_DENSITY) - 1;
}

//Returns the eta vector that points in the direction of the corrective force
__device__ glm::vec3 eta(Particle* particles, int* neighbors, int* numNeighbors, int index, float &vorticityMag) {
	glm::vec3 eta = glm::vec3(0.0f);
	for (int i = 0; i < numNeighbors[index]; i++) {
		eta += WSpiky(particles[index].newPos, particles[neighbors[index + i]].newPos) * vorticityMag;
	}

	return eta;
}

//Calculates the vorticity force for a particle
__device__ glm::vec3 vorticityForce(Particle* particles, int* neighbors, int* numNeighbors, int index) {
	//Calculate omega_i
	glm::vec3 omega = glm::vec3(0.0f);
	glm::vec3 velocityDiff;
	glm::vec3 gradient;

	for (int i = 0; i < numNeighbors[index]; i++) {
		velocityDiff = particles[neighbors[index + i]].velocity - particles[index].velocity;
		gradient = WSpiky(particles[index].newPos, particles[neighbors[index + i]].newPos);
		omega += glm::cross(velocityDiff, gradient);
	}

	float omegaLength = glm::length(omega);
	if (omegaLength == 0.0f) {
		//No direction for eta
		return glm::vec3(0.0f);
	}

	glm::vec3 etaVal = eta(particles, neighbors, numNeighbors, index, omegaLength);
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

__device__ float sCorrCalc(Particle &pi, Particle &pj) {
	//Get Density from WPoly6
	float corr = WPoly6(pi.newPos, pj.newPos) / wQH;
	corr *= corr * corr * corr;
	return -K * corr;
}

__device__ glm::vec3 xsphViscosity(Particle* particles, int* neighbors, int* numNeighbors, int index) {
	glm::vec3 visc = glm::vec3(0.0f);
	for (int i = 0; i < numNeighbors[index]; i++) {
		glm::vec3 velocityDiff = particles[neighbors[index + i]].velocity - particles[index].velocity;
		velocityDiff *= WPoly6(particles[index].newPos, particles[neighbors[index + i]].newPos);
		visc += velocityDiff;
	}

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

__global__ void predictPositions(Particle* particles) {
	int i = threadIdx.x + (blockIdx.x * blockDim.x);

	//update velocity vi = vi + dt * fExt
	particles[i].velocity += GRAVITY * deltaT;

	//predict position x* = xi + dt * vi
	particles[i].newPos += particles[i].velocity * deltaT;

	confineToBox(particles[i]);
}

__global__ void clearNeighbors(int* neighbors, int* numNeighbors) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index > NUM_PARTICLES_C) return;
	/*for (int i = 0; i < numNeighbors[index]; i++) {
		neighbors[index + i] = -1;
	}*/

	numNeighbors[index] = 0;
}

__global__ void updateNeighbors(Particle* particles, int* neighbors, int* numNeighbors) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);

	//Naive method for now
	for (int i = 0; i < NUM_PARTICLES_C; i++) {
		if (numNeighbors[index] >= MAX_NEIGHBORS_C) return;
		if (glm::distance(particles[index].newPos, particles[i].newPos) <= H) {
			neighbors[numNeighbors[index]] = i;
			numNeighbors[index]++;
		}
	}
}

__global__ void calcLambda(Particle* particles, int* neighbors, int* numNeighbors, float* buffer2) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);

	float densityConstraint = calcDensityConstraint(particles, neighbors, numNeighbors, index);
	glm::vec3 gradientI = glm::vec3(0.0f);
	float sumGradients = 0.0f;
	for (int i = 0; i < numNeighbors[index]; i++) {
		//Calculate gradient with respect to j
		glm::vec3 gradientJ = WSpiky(particles[index].newPos, particles[neighbors[index + i]].newPos) / REST_DENSITY;

		//Add magnitude squared to sum
		sumGradients += glm::length2(gradientJ);
		gradientI += gradientJ;
	}

	//Add the particle i gradient magnitude squared to sum
	sumGradients += glm::length2(gradientI);
	buffer2[index] = ((-1) * densityConstraint) / (sumGradients + EPSILON_LAMBDA);
}

__global__ void calcDeltaP(Particle* particles, int* neighbors, int* numNeighbors, glm::vec3* buffer1, float* buffer2) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);

	glm::vec3 deltaP = glm::vec3(0.0f);
	for (int i = 0; i < numNeighbors[index]; i++) {
		float lambdaSum = buffer2[index] + buffer2[neighbors[index + i]];
		float sCorr = sCorrCalc(particles[index], particles[neighbors[index + i]]);
		deltaP += WSpiky(particles[index].newPos, particles[neighbors[index + i]].newPos) * (lambdaSum + sCorr);
	}

	buffer1[index] = deltaP / REST_DENSITY;
}

__global__ void updatePositions(Particle* particles, glm::vec3* buffer1) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);

	particles[index].newPos += buffer1[index];
}

__global__ void updateVelocities(Particle* particles, int* neighbors, int* numNeighbors, glm::vec3* buffer1) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);

	confineToBox(particles[index]);

	//set new velocity vi = (x*i - xi) / dt
	particles[index].velocity = (particles[index].newPos - particles[index].oldPos) / deltaT;

	//apply vorticity confinement
	particles[index].velocity += vorticityForce(particles, neighbors, numNeighbors, index) * deltaT;

	//apply XSPH viscosity
	buffer1[index] = xsphViscosity(particles, neighbors, numNeighbors, index);

	//update position xi = x*i
	particles[index].oldPos = particles[index].newPos;

}

__global__ void updateXSPHVelocities(Particle* particles, glm::vec3* buffer1) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	
	particles[index].velocity += buffer1[index] * deltaT;
}

void update(Particle* particles, int* neighbors, int* numNeighbors, glm::vec3* buffer1, float* buffer2) {
	//------------------WATER-----------------
	//Predict positions and update velocity
	predictPositions<<<dims, blockSize>>>(particles);

	//Update neighbors
	clearNeighbors<<<dims, blockSize>>>(neighbors, numNeighbors);
	updateNeighbors<<<dims, blockSize>>>(particles, neighbors, numNeighbors);

	//Needs to be after neighbor finding for weighted positions
	//updatePositions<<<dims, blockSize>>>();

	for (int pi = 0; pi < PRESSURE_ITERATIONS; pi++) {
		//set lambda
		calcLambda<<<dims, blockSize>>>(particles, neighbors, numNeighbors, buffer2);

		//calculate deltaP
		calcDeltaP<<<dims, blockSize>>>(particles, neighbors, numNeighbors, buffer1, buffer2);

		//update position x*i = x*i + deltaPi
		updatePositions<<<dims, blockSize>>>(particles, buffer1);
	}

	//Update velocity, apply vorticity confinement, apply xsph viscosity, update position
	updateVelocities<<<dims, blockSize>>>(particles, neighbors, numNeighbors, buffer1);

	//Set new velocity
	updateXSPHVelocities<<<dims, blockSize>>>(particles, buffer1);
}

#endif