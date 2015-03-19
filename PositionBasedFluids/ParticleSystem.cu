#ifndef PARTICLE_SYSTEM_CU
#define PARTICLE_SYSTEM_CU

#include "common.h"
#include "Constants.h"
#include "Particle.hpp"

__constant__ float width = w * H;
__constant__ float height = h * H;
__constant__ float depth = d * H;

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
		sum += WPoly6(particles[index].newPos, particles[neighbors[(index * MAX_NEIGHBORS_C) + i]].newPos);
	}

	return (sum / REST_DENSITY) - 1;
}

//Returns the eta vector that points in the direction of the corrective force
__device__ glm::vec3 eta(Particle* particles, int* neighbors, int* numNeighbors, int index, float &vorticityMag) {
	glm::vec3 eta = glm::vec3(0.0f);
	for (int i = 0; i < numNeighbors[index]; i++) {
		eta += WSpiky(particles[index].newPos, particles[neighbors[(index * MAX_NEIGHBORS_C) + i]].newPos) * vorticityMag;
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
		velocityDiff = particles[neighbors[(index * MAX_NEIGHBORS_C) + i]].velocity - particles[index].velocity;
		gradient = WSpiky(particles[index].newPos, particles[neighbors[(index * MAX_NEIGHBORS_C) + i]].newPos);
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
	//if (glm::isinf(n.x) || glm::isinf(n.y) || glm::isinf(n.z)) {
		//return glm::vec3(0.0f);
	//}

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
		glm::vec3 velocityDiff = particles[neighbors[(index * MAX_NEIGHBORS_C) + i]].velocity - particles[index].velocity;
		velocityDiff *= WPoly6(particles[index].newPos, particles[neighbors[(index * MAX_NEIGHBORS_C) + i]].newPos);
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

__device__ int getGridIndex(glm::ivec3 pos) {
	//return int(pos.x + w * (pos.y + d * pos.z));
	//return int((pos.x * h + pos.y) * d + pos.z);
	//return int(pos.x*w*h + pos.y*w + pos.z);
	return int((pos.z * h * w) + (pos.y * w) + pos.x);
}

__global__ void predictPositions(Particle* particles) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= NUM_PARTICLES_C) return;

	//update velocity vi = vi + dt * fExt
	particles[index].velocity += GRAVITY * deltaT;

	//predict position x* = xi + dt * vi
	particles[index].newPos += particles[index].velocity * deltaT;

	confineToBox(particles[index]);
}

__global__ void clearNeighbors(int* neighbors, int* numNeighbors) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= NUM_PARTICLES_C) return;

	numNeighbors[index] = 0;
}

__global__ void clearGrid(int* gridCounters) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= GRID_SIZE_C) return;

	gridCounters[index] = 0;
}

__global__ void updateGrid(Particle* particles, int* gridCells, int* gridCounters) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= NUM_PARTICLES_C) return;

	Particle &p = particles[index];
	int gIndex = getGridIndex(glm::ivec3(int(p.newPos.x / H) % w, int(p.newPos.y / H) % h, int(p.newPos.z / H) % d));

	int i = atomicAdd(&gridCounters[gIndex], 1);
	i = min(i, MAX_PARTICLES_C - 1);
	gridCells[gIndex * MAX_PARTICLES_C + i] = index;
}

__global__ void updateNeighbors(Particle* particles, int* gridCells, int* gridCounters, int* neighbors, int* numNeighbors) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= NUM_PARTICLES_C) return;
	
	Particle &p = particles[index];
	glm::ivec3 pos = glm::ivec3(int(p.newPos.x / H) % w, int(p.newPos.y / H) % h, int(p.newPos.z / H) % d);
	int pIndex;

	for (int z = -1; z < 2; z++) {
		for (int y = -1; y < 2; y++) {
			for (int x = -1; x < 2; x++) {
				glm::ivec3 n = glm::ivec3(pos.x + x, pos.y + y, pos.z + z);
				if (n.x >= 0 && n.x < w && n.y >= 0 && n.y < h && n.z >= 0 && n.z < d) {
					int gIndex = getGridIndex(n);
					int cellParticles = min(gridCounters[gIndex], MAX_PARTICLES_C - 1);
					for (int i = 0; i < cellParticles; i++) {
						if (numNeighbors[index] >= MAX_NEIGHBORS_C) return;

						pIndex = gridCells[gIndex * MAX_PARTICLES_C + i];
						if (glm::distance(particles[index].newPos, particles[pIndex].newPos) <= H) {
							neighbors[(index * MAX_NEIGHBORS_C) + numNeighbors[index]] = pIndex;
							numNeighbors[index]++;
						}
					}
				}
			}
		}
	}
}

__global__ void calcLambda(Particle* particles, int* neighbors, int* numNeighbors, float* buffer2) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= NUM_PARTICLES_C) return;

	float densityConstraint = calcDensityConstraint(particles, neighbors, numNeighbors, index);
	glm::vec3 gradientI = glm::vec3(0.0f);
	float sumGradients = 0.0f;
	for (int i = 0; i < numNeighbors[index]; i++) {
		//Calculate gradient with respect to j
		glm::vec3 gradientJ = WSpiky(particles[index].newPos, particles[neighbors[(index * MAX_NEIGHBORS_C) + i]].newPos) / REST_DENSITY;

		//Add magnitude squared to sum
		sumGradients += glm::length2(gradientJ);
		gradientI += gradientJ;
	}

	//Add the particle i gradient magnitude squared to sum
	sumGradients += glm::length2(gradientI);
	buffer2[index] = (-1 * densityConstraint) / (sumGradients + EPSILON_LAMBDA);
}

__global__ void calcDeltaP(Particle* particles, int* neighbors, int* numNeighbors, glm::vec3* buffer1, float* buffer2) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= NUM_PARTICLES_C) return;

	glm::vec3 deltaP = glm::vec3(0.0f);
	for (int i = 0; i < numNeighbors[index]; i++) {
		float lambdaSum = buffer2[index] + buffer2[neighbors[(index * MAX_NEIGHBORS_C) + i]];
		float sCorr = sCorrCalc(particles[index], particles[neighbors[(index * MAX_NEIGHBORS_C) + i]]);
		deltaP += WSpiky(particles[index].newPos, particles[neighbors[(index * MAX_NEIGHBORS_C) + i]].newPos) * (lambdaSum + sCorr);
	}

	buffer1[index] = deltaP / REST_DENSITY;
}

__global__ void applyDeltaP(Particle* particles, glm::vec3* buffer1) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= NUM_PARTICLES_C) return;

	particles[index].newPos += buffer1[index];
}

__global__ void updateVelocities(Particle* particles, int* neighbors, int* numNeighbors, glm::vec3* buffer1) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= NUM_PARTICLES_C) return;

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
	if (index >= NUM_PARTICLES_C) return;

	particles[index].velocity += buffer1[index] * deltaT;
}

__global__ void updateVBO(Particle* particles, float* vboPtr) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= NUM_PARTICLES_C) return;

	vboPtr[3 * index] = particles[index].oldPos.x;
	vboPtr[3 * index + 1] = particles[index].oldPos.y;
	vboPtr[3 * index + 2] = particles[index].oldPos.z;
}

void update(Particle* particles, int* gridCells, int* gridCounters, int* neighbors, int* numNeighbors, glm::vec3* buffer1, float* buffer2) {
	//------------------WATER-----------------
	//Predict positions and update velocity
	predictPositions<<<dims, blockSize>>>(particles);

	//Update neighbors
	clearNeighbors<<<dims, blockSize>>>(neighbors, numNeighbors);
	clearGrid<<<gridDims, blockSize>>>(gridCounters);
	updateGrid<<<dims, blockSize>>>(particles, gridCells, gridCounters);
	updateNeighbors<<<dims, blockSize>>>(particles, gridCells, gridCounters, neighbors, numNeighbors);

	for (int i = 0; i < PRESSURE_ITERATIONS; i++) {
		//set lambda
		calcLambda<<<dims, blockSize>>>(particles, neighbors, numNeighbors, buffer2);

		//calculate deltaP
		calcDeltaP<<<dims, blockSize>>>(particles, neighbors, numNeighbors, buffer1, buffer2);

		//update position x*i = x*i + deltaPi
		applyDeltaP<<<dims, blockSize>>>(particles, buffer1);
	}

	//Update velocity, apply vorticity confinement, apply xsph viscosity, update position
	updateVelocities<<<dims, blockSize>>>(particles, neighbors, numNeighbors, buffer1);

	//Set new velocity
	updateXSPHVelocities<<<dims, blockSize>>>(particles, buffer1);
}

void setVBO(Particle* particles, float* vboPtr) {
	updateVBO<<<dims, blockSize>>>(particles, vboPtr);
}

#endif