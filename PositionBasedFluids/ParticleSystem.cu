#ifndef PARTICLE_SYSTEM_CU
#define PARTICLE_SYSTEM_CU

#include "common.h"
#include "Constants.h"
#include "ParticleSystem.h"

__constant__ int width = gridWidth * H;
__constant__ int height = gridHeight * H;
__constant__ int depth = gridDepth * H;
__constant__ float distr[] = 
{ 
	-0.34828757091811f, -0.64246175794046f, -0.15712936555833f, -0.28922267225069f, 0.70090742209037f,
	0.54293139350737f, 0.86755128105523f, 0.68346917800767f, -0.74589352018474f, 0.39762042062246f,
	-0.70243115988673f, -0.85088539675385f, -0.25780126697281f, 0.61167922970451f, -0.8751634423971f,
	-0.12334015086449f, 0.10898816916579f, -0.97167591190509f, 0.89839695948101f, -0.71134930649369f,
	-0.33928178406287f, -0.27579196788175f, -0.5057460942798f, 0.2341509513716f, 0.97802030852904f,
	0.49743173248015f, -0.92212845381448f, 0.088328595779989f, -0.70214782175708f, -0.67050553191011f
};
__device__ int foamCount = 0;

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

//Returns the eta vector that points in the direction of the corrective force
__device__ glm::vec3 eta(Particle* particles, int* neighbors, int* numNeighbors, int index, float &vorticityMag) {
	glm::vec3 eta = glm::vec3(0.0f);
	for (int i = 0; i < numNeighbors[index]; i++) {
		if (particles[neighbors[(index * MAX_NEIGHBORS) + i]].phase == 0)
			eta += WSpiky(particles[index].newPos, particles[neighbors[(index * MAX_NEIGHBORS) + i]].newPos) * vorticityMag;
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
		if (particles[neighbors[(index * MAX_NEIGHBORS) + i]].phase == 0) {
			velocityDiff = particles[neighbors[(index * MAX_NEIGHBORS) + i]].velocity - particles[index].velocity;
			gradient = WSpiky(particles[index].newPos, particles[neighbors[(index * MAX_NEIGHBORS) + i]].newPos);
			omega += glm::cross(velocityDiff, gradient);
		}
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
		if (particles[neighbors[(index * MAX_NEIGHBORS) + i]].phase == 0) {
			glm::vec3 velocityDiff = particles[neighbors[(index * MAX_NEIGHBORS) + i]].velocity - particles[index].velocity;
			velocityDiff *= WPoly6(particles[index].newPos, particles[neighbors[(index * MAX_NEIGHBORS) + i]].newPos);
			visc += velocityDiff;
		}
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

__device__ void confineToBox(FoamParticle &p) {
	if (p.pos.x < 0 || p.pos.x > width) {
		p.velocity.x = 0;
		if (p.pos.x < 0) p.pos.x = 0.001f;
		else p.pos.x = width - 0.001f;
	}

	if (p.pos.y < 0 || p.pos.y > height) {
		p.velocity.y = 0;
		if (p.pos.y < 0) p.pos.y = 0.001f;
		else p.pos.y = height - 0.001f;
	}

	if (p.pos.z < 0 || p.pos.z > depth) {
		p.velocity.z = 0;
		if (p.pos.z < 0) p.pos.z = 0.001f;
		else p.pos.z = depth - 0.001f;
	}
}

__device__ glm::ivec3 getGridPos(glm::vec3 pos) {
	return glm::ivec3(int(pos.x / H) % gridWidth, int(pos.y / H) % gridHeight, int(pos.z / H) % gridDepth);
}

__device__ int getGridIndex(glm::ivec3 pos) {
	return (pos.z * gridHeight * gridWidth) + (pos.y * gridWidth) + pos.x;
}

__global__ void predictPositions(Particle* particles) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= NUM_PARTICLES) return;

	//update velocity vi = vi + dt * fExt
	particles[index].velocity += particles[index].invMass * GRAVITY * deltaT;

	//predict position x* = xi + dt * vi
	particles[index].newPos += particles[index].velocity * deltaT;

	confineToBox(particles[index]);
}

__global__ void clearNeighbors(int* numNeighbors, int* numContacts) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= NUM_PARTICLES) return;

	numNeighbors[index] = 0;
	numContacts[index] = 0;
}

__global__ void clearGrid(int* gridCounters) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= GRID_SIZE) return;

	gridCounters[index] = 0;
}

__global__ void updateGrid(Particle* particles, int* gridCells, int* gridCounters) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= NUM_PARTICLES) return;

	glm::ivec3 pos = getGridPos(particles[index].newPos);
	int gIndex = getGridIndex(pos);

	int i = atomicAdd(&gridCounters[gIndex], 1);
	i = min(i, MAX_PARTICLES - 1);
	gridCells[gIndex * MAX_PARTICLES + i] = index;
}

__global__ void updateNeighbors(Particle* particles, int* gridCells, int* gridCounters, int* neighbors, int* numNeighbors, int* contacts, int* numContacts) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= NUM_PARTICLES) return;
	
	glm::ivec3 pos = getGridPos(particles[index].newPos);
	int pIndex;

	for (int z = -1; z < 2; z++) {
		for (int y = -1; y < 2; y++) {
			for (int x = -1; x < 2; x++) {
				glm::ivec3 n = glm::ivec3(pos.x + x, pos.y + y, pos.z + z);
				if (n.x >= 0 && n.x < gridWidth && n.y >= 0 && n.y < gridHeight && n.z >= 0 && n.z < gridDepth) {
					int gIndex = getGridIndex(n);
					int cellParticles = min(gridCounters[gIndex], MAX_PARTICLES - 1);
					for (int i = 0; i < cellParticles; i++) {
						if (numNeighbors[index] >= MAX_NEIGHBORS) return;

						pIndex = gridCells[gIndex * MAX_PARTICLES + i];
						if (glm::distance(particles[index].newPos, particles[pIndex].newPos) <= H) {
							neighbors[(index * MAX_NEIGHBORS) + numNeighbors[index]] = pIndex;
							numNeighbors[index]++;
							if (particles[index].phase == 0 && particles[pIndex].phase == 1 && numContacts[index] < MAX_CONTACTS) {
								contacts[index * MAX_CONTACTS + numContacts[index]] = pIndex;
								numContacts[index]++;
							}
						}
					}
				}
			}
		}
	}
}

__global__ void particleCollisions(Particle* particles, int* contacts, int* numContacts, glm::vec3* deltaPs, float* buffer3) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= NUM_PARTICLES) return;

	for (int i = 0; i < numContacts[index]; i++) {
		int nIndex = contacts[index * MAX_CONTACTS + i];
		if (particles[nIndex].invMass == 0) continue;
		glm::vec3 dir = particles[index].newPos - particles[nIndex].newPos;
		float length = glm::length(dir);
		float invMass = particles[index].invMass + particles[nIndex].invMass;
		glm::vec3 dp;
		if (length == 0.0f || invMass == 0.0f) dp = glm::vec3(0);
		else dp = (1 / invMass) * (length - H) * (dir / length);
		deltaPs[index] -= dp;
		buffer3[index]++;

		atomicAdd(&deltaPs[nIndex].x, dp.x);
		atomicAdd(&deltaPs[nIndex].y, dp.y);
		atomicAdd(&deltaPs[nIndex].z, dp.z);
		atomicAdd(&buffer3[nIndex], 1);
	}
}

__global__ void calcDensities(Particle* particles, int* neighbors, int* numNeighbors, float* densities) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= NUM_PARTICLES || particles[index].phase != 0) return;

	float rhoSum = 0.0f;
	for (int i = 0; i < numNeighbors[index]; i++) {
		if (particles[neighbors[(index * MAX_NEIGHBORS) + i]].phase == 0)
			rhoSum += WPoly6(particles[index].newPos, particles[neighbors[(index * MAX_NEIGHBORS) + i]].newPos);
	}

	densities[index] = rhoSum;
}

__global__ void calcLambda(Particle* particles, int* neighbors, int* numNeighbors, float* densities, float* buffer3) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= NUM_PARTICLES || particles[index].phase != 0) return;

	float densityConstraint = (densities[index] / REST_DENSITY) - 1;
	glm::vec3 gradientI = glm::vec3(0.0f);
	float sumGradients = 0.0f;
	for (int i = 0; i < numNeighbors[index]; i++) {
		if (particles[neighbors[(index * MAX_NEIGHBORS) + i]].phase == 0) {
			//Calculate gradient with respect to j
			glm::vec3 gradientJ = WSpiky(particles[index].newPos, particles[neighbors[(index * MAX_NEIGHBORS) + i]].newPos) / REST_DENSITY;

			//Add magnitude squared to sum
			sumGradients += glm::length2(gradientJ);
			gradientI += gradientJ;
		}
	}

	//Add the particle i gradient magnitude squared to sum
	sumGradients += glm::length2(gradientI);
	buffer3[index] = (-1 * densityConstraint) / (sumGradients + EPSILON_LAMBDA);
}

__global__ void calcDeltaP(Particle* particles, int* neighbors, int* numNeighbors, glm::vec3* deltaPs, float* buffer3) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= NUM_PARTICLES || particles[index].phase != 0) return;
	deltaPs[index] = glm::vec3(0);

	glm::vec3 deltaP = glm::vec3(0.0f);
	for (int i = 0; i < numNeighbors[index]; i++) {
		if (particles[neighbors[(index * MAX_NEIGHBORS) + i]].phase == 0) {
			float lambdaSum = buffer3[index] + buffer3[neighbors[(index * MAX_NEIGHBORS) + i]];
			float sCorr = sCorrCalc(particles[index], particles[neighbors[(index * MAX_NEIGHBORS) + i]]);
			deltaP += WSpiky(particles[index].newPos, particles[neighbors[(index * MAX_NEIGHBORS) + i]].newPos) * (lambdaSum + sCorr);

		}
	}

	deltaPs[index] = deltaP / REST_DENSITY;
}

__global__ void applyDeltaP(Particle* particles, glm::vec3* deltaPs, float* buffer3, int flag) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= NUM_PARTICLES) return;

	if (buffer3[index] > 0 && flag == 1) particles[index].newPos += deltaPs[index] / buffer3[index];
	else if (flag == 0) particles[index].newPos += deltaPs[index];
}

__global__ void updateVelocities(Particle* particles, int* neighbors, int* numNeighbors, glm::vec3* deltaPs) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= NUM_PARTICLES || particles[index].phase != 0) return;

	confineToBox(particles[index]);

	//set new velocity vi = (x*i - xi) / dt
	particles[index].velocity = (particles[index].newPos - particles[index].oldPos) / deltaT;

	//apply vorticity confinement
	particles[index].velocity += vorticityForce(particles, neighbors, numNeighbors, index) * deltaT;

	//apply XSPH viscosity
	deltaPs[index] = xsphViscosity(particles, neighbors, numNeighbors, index);

	//update position xi = x*i
	particles[index].oldPos = particles[index].newPos;
}

__global__ void updateXSPHVelocities(Particle* particles, glm::vec3* deltaPs) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= NUM_PARTICLES || particles[index].phase != 0) return;

	particles[index].velocity += deltaPs[index] * deltaT;
}

__global__ void generateFoam(Particle* particles, FoamParticle* foamParticles, int* neighbors, int* numNeighbors, float* densities) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= NUM_PARTICLES || foamCount >= NUM_FOAM) return;

	float velocityDiff = 0.0f;
	for (int i = 0; i < numNeighbors[index]; i++) {
		int nIndex = neighbors[(index * MAX_NEIGHBORS) + i];
		if (index != nIndex) {
			float wAir = WAirPotential(particles[index].newPos, particles[nIndex].newPos);
			glm::vec3 xij = glm::normalize(particles[index].newPos - particles[nIndex].newPos);
			glm::vec3 vijHat = glm::normalize(particles[index].velocity - particles[nIndex].velocity);
			velocityDiff += glm::length(particles[index].velocity - particles[nIndex].velocity) * (1 - glm::dot(vijHat, xij)) * wAir;
		}
	}

	float ek = 0.5f * glm::length2(particles[index].velocity);
	float potential = velocityDiff * ek * max(1.0f - (1.0f * densities[index] / REST_DENSITY), 0.0f);
	int nd = 0;
	if (potential > 0.7f) nd = min(20, (NUM_FOAM - 1 - foamCount));
	nd = atomicAdd(&foamCount, nd);
	for (int i = 0; i < nd; i++) {
		float rx = distr[i % 30] * H;
		float ry = distr[(i + 1) % 30] * H;
		float rz = distr[(i + 2) % 30] * H;
		int rd = distr[index % 30] > 0.5f ? 1 : -1;

		glm::vec3 xd = particles[index].newPos + glm::vec3(rx * rd, ry * rd, rz * rd);
		int type;
		if (numNeighbors[index] + 1 < 8) type = 1;
		else type = 2;
		foamParticles[foamCount + i].pos = xd;
		foamParticles[foamCount + i].velocity = particles[index].velocity;
		foamParticles[foamCount + i].ttl = 1.0f;
		foamParticles[foamCount + i].type = type;
		confineToBox(foamParticles[foamCount + i]);
	}
}

__global__ void updateFoam(FoamParticle* foamParticles) {

}

__global__ void clearDeltaP(Particle* particles, glm::vec3* deltaPs, float* buffer3) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= NUM_PARTICLES) return;

	deltaPs[index] = glm::vec3(0);
	buffer3[index] = 0;
}

__global__ void solveDistance(Particle* particles, DistanceConstraint* dConstraints, int numConstraints, glm::vec3* deltaPs, float* buffer3) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= numConstraints) return;

	DistanceConstraint &c = dConstraints[index];
	glm::vec3 dir = particles[c.p1].newPos - particles[c.p2].newPos;
	float length = glm::length(dir);
	float invMass = particles[c.p1].invMass + particles[c.p2].invMass;
	glm::vec3 dp;
	if (length == 0.0f || invMass == 0.0f) dp = glm::vec3(0);
	else {
		if (c.stiffness > 0) dp = (1 / invMass) * (length - c.restLength) * (dir / length) * (1.0f - glm::pow(1.0f - c.stiffness, 1.0f / 4));
		else if (length > c.restLength) {
			dp += (1 / invMass) * (length - c.restLength) * (dir / length) * (1.0f - glm::pow(1.0f + c.stiffness, 1.0f / 4));
		}
	}
	if (particles[c.p1].invMass > 0) {
		atomicAdd(&deltaPs[c.p1].x, -dp.x);
		atomicAdd(&deltaPs[c.p1].y, -dp.y);
		atomicAdd(&deltaPs[c.p1].z, -dp.z);
		atomicAdd(&buffer3[c.p1], 1);
	}

	if (particles[c.p2].invMass > 0) {
		atomicAdd(&deltaPs[c.p2].x, dp.x);
		atomicAdd(&deltaPs[c.p2].y, dp.y);
		atomicAdd(&deltaPs[c.p2].z, dp.z);
		atomicAdd(&buffer3[c.p2], 1);
	}
}

__global__ void updateClothVelocity(Particle* particles) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= NUM_PARTICLES || particles[index].phase != 1) return;

	particles[index].velocity = (particles[index].newPos - particles[index].oldPos) / deltaT;
	particles[index].oldPos = particles[index].newPos;
}

void updateWater(Buffers* p) {
	//------------------WATER-----------------
	for (int i = 0; i < PRESSURE_ITERATIONS; i++) {
		//Calculate fluid densities and store in densities
		calcDensities<<<dims, blockSize>>>(p->particles, p->neighbors, p->numNeighbors, p->densities);

		//Calculate all lambdas and store in buffer3
		calcLambda<<<dims, blockSize>>>(p->particles, p->neighbors, p->numNeighbors, p->densities, p->buffer3);

		//calculate deltaP
		calcDeltaP<<<dims, blockSize>>>(p->particles, p->neighbors, p->numNeighbors, p->deltaPs, p->buffer3);

		//update position x*i = x*i + deltaPi
		applyDeltaP<<<dims, blockSize>>>(p->particles, p->deltaPs, p->buffer3, 0);
	}

	//Update velocity, apply vorticity confinement, apply xsph viscosity, update position
	updateVelocities<<<dims, blockSize>>>(p->particles, p->neighbors, p->numNeighbors, p->deltaPs);

	//Set new velocity
	updateXSPHVelocities<<<dims, blockSize>>>(p->particles, p->deltaPs);
}

void updateCloth(Buffers* p) {
	static const dim3 constraintDims = int(ceil(p->numConstraints / blockSize));
	clearDeltaP<<<dims, blockSize>>>(p->particles, p->deltaPs, p->buffer3);

	for (int i = 0; i < SOLVER_ITERATIONS; i++) {
		solveDistance<<<constraintDims, blockSize>>>(p->particles, p->dConstraints, p->numConstraints, p->deltaPs, p->buffer3);
		applyDeltaP<<<dims, blockSize>>>(p->particles, p->deltaPs, p->buffer3, 1);
	}

	updateClothVelocity<<<dims, blockSize>>>(p->particles);
}

void update(Buffers* p) {
	//Predict positions and update velocity
	predictPositions<<<dims, blockSize>>>(p->particles);

	//Update neighbors
	clearNeighbors<<<dims, blockSize>>>(p->numNeighbors, p->numContacts);
	clearGrid<<<gridDims, blockSize>>>(p->gridCounters);
	updateGrid<<<dims, blockSize>>>(p->particles, p->gridCells, p->gridCounters);
	updateNeighbors<<<dims, blockSize>>>(p->particles, p->gridCells, p->gridCounters, p->neighbors, p->numNeighbors, p->contacts, p->numContacts);

	for (int i = 0; i < 20; i++) {
		clearDeltaP<<<dims, blockSize>>>(p->particles, p->deltaPs, p->buffer3);
		particleCollisions<<<dims, blockSize>>>(p->particles, p->contacts, p->numContacts, p->deltaPs, p->buffer3);
		applyDeltaP<<<dims, blockSize>>>(p->particles, p->deltaPs, p->buffer3, 1);
	}

	//Solve constraints
	updateWater(p);
	updateCloth(p);
}

__global__ void updateVBO(Particle* particles, float* fluidPositions, float* clothPositions) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= NUM_PARTICLES) return;

	if (particles[index].phase == 0) {
		fluidPositions[3 * index] = particles[index].oldPos.x;
		fluidPositions[3 * index + 1] = particles[index].oldPos.y;
		fluidPositions[3 * index + 2] = particles[index].oldPos.z;
	} else {
		clothPositions[3 * index] = particles[index].oldPos.x;
		clothPositions[3 * index + 1] = particles[index].oldPos.y;
		clothPositions[3 * index + 2] = particles[index].oldPos.z;
	}
}

void setVBO(Particle* particles, float* fluidPositions, float* clothPositions) {
	updateVBO<<<dims, blockSize>>>(particles, fluidPositions, clothPositions);
}

#endif