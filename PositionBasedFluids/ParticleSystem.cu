#ifndef PARTICLE_SYSTEM_CU
#define PARTICLE_SYSTEM_CU

#include "common.h"
#include "parameters.h"

#define cudaCheck(x) { cudaError_t err = x; if (err != cudaSuccess) { printf("Cuda error: %d in %s at %s:%d\n", err, #x, __FILE__, __LINE__); assert(0); } }
static dim3 dims;
static dim3 gridDims;
static const int blockSize = 128;

__constant__ solverParams sp;
__constant__ float deltaT = 0.0083f;
__device__ int MAX_NEIGHBORS = 50;
__device__ int MAX_PARTICLES = 50;
__device__ int MAX_CONTACTS = 10;
__constant__ float distr[] =
{
	-0.34828757091811f, -0.64246175794046f, -0.15712936555833f, -0.28922267225069f, 0.70090742209037f,
	0.54293139350737f, 0.86755128105523f, 0.68346917800767f, -0.74589352018474f, 0.39762042062246f,
	-0.70243115988673f, -0.85088539675385f, -0.25780126697281f, 0.61167922970451f, -0.8751634423971f,
	-0.12334015086449f, 0.10898816916579f, -0.97167591190509f, 0.89839695948101f, -0.71134930649369f,
	-0.33928178406287f, -0.27579196788175f, -0.5057460942798f, 0.2341509513716f, 0.97802030852904f,
	0.49743173248015f, -0.92212845381448f, 0.088328595779989f, -0.70214782175708f, -0.67050553191011f
};

__device__ float WPoly6(glm::vec3 const &pi, glm::vec3 const &pj) {
	glm::vec3 r = pi - pj;
	float rLen = glm::length(r);
	if (rLen > sp.restDistance || rLen == 0) {
		return 0;
	}

	return sp.KPOLY * glm::pow((sp.restDistance * sp.restDistance - glm::length2(r)), 3);
}

__device__ glm::vec3 gradWPoly6(glm::vec3 const &pi, glm::vec3 const &pj) {
	glm::vec3 r = pi - pj;
	float rLen = glm::length(r);
	if (rLen > sp.restDistance || rLen == 0) {
		return glm::vec3(0.0f);
	}

	float coeff = glm::pow((sp.restDistance * sp.restDistance) - (rLen * rLen), 2);
	coeff *= -6 * sp.KPOLY;
	return r * coeff;
}

__device__ glm::vec3 WSpiky(glm::vec3 const &pi, glm::vec3 const &pj) {
	glm::vec3 r = pi - pj;
	float rLen = glm::length(r);
	if (rLen > sp.restDistance || rLen == 0) {
		return glm::vec3(0.0f);
	}

	float coeff = (sp.restDistance - rLen) * (sp.restDistance - rLen);
	coeff *= sp.SPIKY;
	coeff /= rLen;
	return r * -coeff;
}

__device__ float WAirPotential(glm::vec3 const &pi, glm::vec3 const &pj) {
	glm::vec3 r = pi - pj;
	float rLen = glm::length(r);
	if (rLen > sp.restDistance || rLen == 0) {
		return 0.0f;
	}

	return 1 - (rLen / sp.restDistance);
}

//Returns the eta vector that points in the direction of the corrective force
__device__ glm::vec3 eta(glm::vec4* newPos, int* phases, int* neighbors, int* numNeighbors, int &index, float &vorticityMag) {
	glm::vec3 eta = glm::vec3(0.0f);
	for (int i = 0; i < numNeighbors[index]; i++) {
		if (phases[neighbors[(index * MAX_NEIGHBORS) + i]] == 0)
			eta += WSpiky(glm::vec3(newPos[index]), glm::vec3(newPos[neighbors[(index * MAX_NEIGHBORS) + i]])) * vorticityMag;
	}

	return eta;
}

//Calculates the vorticity force for a particle
__device__ glm::vec3 vorticityForce(glm::vec4* newPos, glm::vec3* velocities, int* phases, int* neighbors, int* numNeighbors, int index) {
	//Calculate omega_i
	glm::vec3 omega = glm::vec3(0.0f);
	glm::vec3 velocityDiff;
	glm::vec3 gradient;

	for (int i = 0; i < numNeighbors[index]; i++) {
		if (phases[neighbors[(index * MAX_NEIGHBORS) + i]] == 0) {
			velocityDiff = velocities[neighbors[(index * MAX_NEIGHBORS) + i]] - velocities[index];
			gradient = WSpiky(glm::vec3(newPos[index]), glm::vec3(newPos[neighbors[(index * MAX_NEIGHBORS) + i]]));
			omega += glm::cross(velocityDiff, gradient);
		}
	}

	float omegaLength = glm::length(omega);
	if (omegaLength == 0.0f) {
		//No direction for eta
		return glm::vec3(0.0f);
	}

	glm::vec3 etaVal = eta(newPos, phases, neighbors, numNeighbors, index, omegaLength);
	if (etaVal == glm::vec3(0.0f)) {
		//Particle is isolated or net force is 0
		return glm::vec3(0.0f);
	}

	glm::vec3 n = glm::normalize(etaVal);

	return (glm::cross(n, omega) * sp.vorticityEps);
}

__device__ float sCorrCalc(glm::vec4 &pi, glm::vec4 &pj) {
	//Get Density from WPoly6
	float corr = WPoly6(glm::vec3(pi), glm::vec3(pj)) / sp.wQH;
	corr *= corr * corr * corr;
	return -sp.K * corr;
}

__device__ glm::vec3 xsphViscosity(glm::vec4* newPos, glm::vec3* velocities, int* phases, int* neighbors, int* numNeighbors, int index) {
	glm::vec3 visc = glm::vec3(0.0f);
	for (int i = 0; i < numNeighbors[index]; i++) {
		if (phases[neighbors[(index * MAX_NEIGHBORS) + i]] == 0) {
			glm::vec3 velocityDiff = velocities[neighbors[(index * MAX_NEIGHBORS) + i]] - velocities[index];
			velocityDiff *= WPoly6(glm::vec3(newPos[index]), glm::vec3(newPos[neighbors[(index * MAX_NEIGHBORS) + i]]));
			visc += velocityDiff;
		}
	}

	return visc * sp.C;
}

/*__device__ void confineToBox(Particle &p) {
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
}*/

__device__ glm::ivec3 getGridPos(glm::vec4 pos) {
	return glm::ivec3(int(pos.x / sp.restDistance) % sp.gridWidth, int(pos.y / sp.restDistance) % sp.gridHeight, int(pos.z / sp.restDistance) % sp.gridDepth);
}

__device__ int getGridIndex(glm::ivec3 pos) {
	return (pos.z * sp.gridHeight * sp.gridWidth) + (pos.y * sp.gridWidth) + pos.x;
}

__global__ void predictPositions(glm::vec4* newPos, glm::vec3* velocities) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= sp.numParticles) return;

	//update velocity vi = vi + dt * fExt
	velocities[index] += newPos[index].w * sp.gravity * deltaT;

	//predict position x* = xi + dt * vi
	newPos[index] += glm::vec4(velocities[index] * deltaT, 0);

	//confineToBox(particles[index]);
}

__global__ void clearNeighbors(int* numNeighbors, int* numContacts) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= sp.numParticles) return;

	numNeighbors[index] = 0;
	numContacts[index] = 0;
}

__global__ void clearGrid(int* gridCounters) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= sp.gridSize) return;

	gridCounters[index] = 0;
}

__global__ void updateGrid(glm::vec4* newPos, int* gridCells, int* gridCounters) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= sp.numParticles) return;

	glm::ivec3 pos = getGridPos(newPos[index]);
	int gIndex = getGridIndex(pos);

	int i = atomicAdd(&gridCounters[gIndex], 1);
	i = min(i, MAX_PARTICLES - 1);
	gridCells[gIndex * MAX_PARTICLES + i] = index;
}

__global__ void updateNeighbors(glm::vec4* newPos, int* phases, int* gridCells, int* gridCounters, int* neighbors, int* numNeighbors, int* contacts, int* numContacts) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= sp.numParticles) return;
	
	glm::ivec3 pos = getGridPos(newPos[index]);
	int pIndex;

	for (int z = -1; z < 2; z++) {
		for (int y = -1; y < 2; y++) {
			for (int x = -1; x < 2; x++) {
				glm::ivec3 n = glm::ivec3(pos.x + x, pos.y + y, pos.z + z);
				if (n.x >= 0 && n.x < sp.gridWidth && n.y >= 0 && n.y < sp.gridHeight && n.z >= 0 && n.z < sp.gridDepth) {
					int gIndex = getGridIndex(n);
					int cellParticles = min(gridCounters[gIndex], MAX_PARTICLES - 1);
					for (int i = 0; i < cellParticles; i++) {
						if (numNeighbors[index] >= MAX_NEIGHBORS) return;

						pIndex = gridCells[gIndex * MAX_PARTICLES + i];
						if (glm::distance(glm::vec3(newPos[index]), glm::vec3(newPos[pIndex])) <= sp.restDistance) {
							neighbors[(index * MAX_NEIGHBORS) + numNeighbors[index]] = pIndex;
							numNeighbors[index]++;
							if (phases[index] == 0 && phases[pIndex] == 1 && numContacts[index] < MAX_CONTACTS) {
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

__global__ void particleCollisions(glm::vec4* newPos, int* contacts, int* numContacts, glm::vec3* deltaPs, float* buffer0) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= sp.numParticles) return;

	for (int i = 0; i < numContacts[index]; i++) {
		int nIndex = contacts[index * MAX_CONTACTS + i];
		if (newPos[nIndex].w == 0) continue;
		glm::vec3 dir = glm::vec3(newPos[index]) - glm::vec3(newPos[nIndex]);
		float length = glm::length(dir);
		float invMass = newPos[index].w + newPos[nIndex].w;
		glm::vec3 dp;
		if ((length - sp.restDistance) > 0.0f || length == 0.0f || invMass == 0.0f) dp = glm::vec3(0);
		else dp = (1 / invMass) * (length - sp.restDistance) * (dir / length);
		deltaPs[index] -= dp;
		buffer0[index]++;

		atomicAdd(&deltaPs[nIndex].x, dp.x);
		atomicAdd(&deltaPs[nIndex].y, dp.y);
		atomicAdd(&deltaPs[nIndex].z, dp.z);
		atomicAdd(&buffer0[nIndex], 1);
	}
}

__global__ void calcDensities(glm::vec4* newPos, int* phases, int* neighbors, int* numNeighbors, float* densities) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= sp.numParticles || phases[index] != 0) return;

	float rhoSum = 0.0f;
	for (int i = 0; i < numNeighbors[index]; i++) {
		if (phases[neighbors[(index * MAX_NEIGHBORS) + i]] == 0)
			rhoSum += WPoly6(glm::vec3(newPos[index]), glm::vec3(newPos[neighbors[(index * MAX_NEIGHBORS) + i]]));
	}

	densities[index] = rhoSum;
}

__global__ void calcLambda(glm::vec4* newPos, int* phases, int* neighbors, int* numNeighbors, float* densities, float* buffer0) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= sp.numParticles || phases[index] != 0) return;

	float densityConstraint = (densities[index] / sp.restDensity) - 1;
	glm::vec3 gradientI = glm::vec3(0.0f);
	float sumGradients = 0.0f;
	for (int i = 0; i < numNeighbors[index]; i++) {
		if (phases[neighbors[(index * MAX_NEIGHBORS) + i]] == 0) {
			//Calculate gradient with respect to j
			glm::vec3 gradientJ = WSpiky(glm::vec3(newPos[index]), glm::vec3(newPos[neighbors[(index * MAX_NEIGHBORS) + i]])) / sp.restDensity;

			//Add magnitude squared to sum
			sumGradients += glm::length2(gradientJ);
			gradientI += gradientJ;
		}
	}

	//Add the particle i gradient magnitude squared to sum
	sumGradients += glm::length2(gradientI);
	buffer0[index] = (-1 * densityConstraint) / (sumGradients + sp.lambdaEps);
}

__global__ void calcDeltaP(glm::vec4* newPos, int* phases, int* neighbors, int* numNeighbors, glm::vec3* deltaPs, float* buffer0) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= sp.numParticles || phases[index] != 0) return;
	deltaPs[index] = glm::vec3(0);

	glm::vec3 deltaP = glm::vec3(0.0f);
	for (int i = 0; i < numNeighbors[index]; i++) {
		if (phases[neighbors[(index * MAX_NEIGHBORS) + i]] == 0) {
			float lambdaSum = buffer0[index] + buffer0[neighbors[(index * MAX_NEIGHBORS) + i]];
			float sCorr = sCorrCalc(newPos[index], newPos[neighbors[(index * MAX_NEIGHBORS) + i]]);
			deltaP += WSpiky(glm::vec3(newPos[index]), glm::vec3(newPos[neighbors[(index * MAX_NEIGHBORS) + i]])) * (lambdaSum + sCorr);

		}
	}

	deltaPs[index] = deltaP / sp.restDensity;
}

__global__ void applyDeltaP(glm::vec4* newPos, glm::vec3* deltaPs, float* buffer0, int flag) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= sp.numParticles) return;

	if (buffer0[index] > 0 && flag == 1) newPos[index] += glm::vec4(deltaPs[index] / buffer0[index], 0);
	else if (flag == 0) newPos[index] += glm::vec4(deltaPs[index], 0);
}

__global__ void updateVelocities(glm::vec4* oldPos, glm::vec4* newPos, glm::vec3* velocities, int* phases, int* neighbors, int* numNeighbors, glm::vec3* deltaPs) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= sp.numParticles || phases[index] != 0) return;

	//confineToBox(particles[index]);

	//set new velocity vi = (x*i - xi) / dt
	velocities[index] = (glm::vec3(newPos[index]) - glm::vec3(oldPos[index])) / deltaT;

	//apply vorticity confinement
	velocities[index] += vorticityForce(newPos, velocities, phases, neighbors, numNeighbors, index) * deltaT;

	//apply XSPH viscosity
	deltaPs[index] = xsphViscosity(newPos, velocities, phases, neighbors, numNeighbors, index);

	//update position xi = x*i
	oldPos[index] = newPos[index];
}

__global__ void updateXSPHVelocities(glm::vec4* newPos, glm::vec3* velocities, int* phases, glm::vec3* deltaPs) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= sp.numParticles || phases[index] != 0) return;

	velocities[index] += deltaPs[index] * deltaT;
}

/*__global__ void generateFoam(Particle* particles, FoamParticle* foamParticles, int* neighbors, int* numNeighbors, float* densities) {
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

}*/

__global__ void clearDeltaP(glm::vec3* deltaPs, float* buffer0) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= sp.numParticles) return;

	deltaPs[index] = glm::vec3(0);
	buffer0[index] = 0;
}

/*__global__ void solveDistance(Particle* particles, DistanceConstraint* dConstraints, int numConstraints, glm::vec3* deltaPs, float* buffer3) {
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
}*/

void updateWater(solver* s) {
	//------------------WATER-----------------
	for (int i = 0; i < sp.numIterations; i++) {
		//Calculate fluid densities and store in densities
		calcDensities<<<dims, blockSize>>>(s->newPos, s->phases, s->neighbors, s->numNeighbors, s->densities);

		//Calculate all lambdas and store in buffer0
		calcLambda<<<dims, blockSize>>>(s->newPos, s->phases, s->neighbors, s->numNeighbors, s->densities, s->buffer0);

		//calculate deltaP
		calcDeltaP<<<dims, blockSize>>>(s->newPos, s->phases, s->neighbors, s->numNeighbors, s->deltaPs, s->buffer0);

		//update position x*i = x*i + deltaPi
		applyDeltaP<<<dims, blockSize>>>(s->newPos, s->deltaPs, s->buffer0, 0);
	}

	//Update velocity, apply vorticity confinement, apply xsph viscosity, update position
	updateVelocities<<<dims, blockSize>>>(s->oldPos, s->newPos, s->velocities, s->phases, s->neighbors, s->numNeighbors, s->deltaPs);

	//Set new velocity
	updateXSPHVelocities<<<dims, blockSize>>>(s->newPos, s->velocities, s->phases, s->deltaPs);
}

/*void updateCloth(solver* p) {
	static const dim3 constraintDims = int(ceil(s->numConstraints / blockSize));
	clearDeltaP<<<dims, blockSize>>>(s->particles, s->deltaPs, s->buffer3);

	for (int i = 0; i < SOLVER_ITERATIONS; i++) {
		solveDistance<<<constraintDims, blockSize>>>(s->particles, s->dConstraints, s->numConstraints, s->deltaPs, s->buffer3);
		applyDeltaP<<<dims, blockSize>>>(s->particles, s->deltaPs, s->buffer3, 1);
	}

	updateClothVelocity<<<dims, blockSize>>>(s->particles);
}*/

void update(solver* s) {
	//Predict positions and update velocity
	predictPositions<<<dims, blockSize>>>(s->newPos, s->velocities);

	//Update neighbors
	clearNeighbors<<<dims, blockSize>>>(s->numNeighbors, s->numContacts);
	clearGrid<<<gridDims, blockSize>>>(s->gridCounters);
	updateGrid<<<dims, blockSize>>>(s->newPos, s->gridCells, s->gridCounters);
	updateNeighbors<<<dims, blockSize>>>(s->newPos, s->phases, s->gridCells, s->gridCounters, s->neighbors, s->numNeighbors, s->contacts, s->numContacts);

	/*for (int i = 0; i < 4; i++) {
		clearDeltaP<<<dims, blockSize>>>(s->particles, s->deltaPs, s->buffer3);
		particleCollisions<<<dims, blockSize>>>(s->particles, s->contacts, s->numContacts, s->deltaPs, s->buffer3);
		applyDeltaP<<<dims, blockSize>>>(s->particles, s->deltaPs, s->buffer3, 1);
	}*/

	//Solve constraints
	updateWater(s);
	//updateCloth(p);
}

__global__ void updateVBO(glm::vec4* oldPos, int* phases, float* fluidPositions, float* clothPositions) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	//if (index >= numParticles) return;

	if (phases[index] == 0) {
		fluidPositions[3 * index] = oldPos[index].x;
		fluidPositions[3 * index + 1] = oldPos[index].y;
		fluidPositions[3 * index + 2] = oldPos[index].z;
	} else {
		//clothPositions[3 * index] = particles[index].oldPos.x;
		//clothPositions[3 * index + 1] = particles[index].oldPos.y;
		//clothPositions[3 * index + 2] = particles[index].oldPos.z;
	}
}

void setVBO(glm::vec4* oldPos, int* phases, float* fluidPositions, float* clothPositions) {
	updateVBO<<<dims, blockSize>>>(oldPos, phases, fluidPositions, clothPositions);
}

void initParams(solver* s, int numParticles, int gridSize) {
	dims = int(ceil(numParticles / blockSize));
	cudaCheck(cudaMalloc((void**)&s->oldPos, numParticles * sizeof(glm::vec4)));
	cudaCheck(cudaMalloc((void**)&s->newPos, numParticles * sizeof(glm::vec4)));
	cudaCheck(cudaMalloc((void**)&s->velocities, numParticles * sizeof(glm::vec3)));
	cudaCheck(cudaMalloc((void**)&s->densities, numParticles * sizeof(float)));
	cudaCheck(cudaMalloc((void**)&s->phases, numParticles * sizeof(int)));
	//diffuse goes here
	cudaCheck(cudaMalloc((void**)&s->neighbors, MAX_NEIGHBORS * numParticles * sizeof(int)));
	cudaCheck(cudaMalloc((void**)&s->numNeighbors, numParticles * sizeof(int)));
	cudaCheck(cudaMalloc((void**)&s->gridCells, MAX_PARTICLES * gridSize * sizeof(int)));
	cudaCheck(cudaMalloc((void**)&s->gridCounters, gridSize * sizeof(int)));
	cudaCheck(cudaMalloc((void**)&s->contacts, MAX_CONTACTS * numParticles * sizeof(int)));
	cudaCheck(cudaMalloc((void**)&s->numContacts, numParticles * sizeof(int)));
	cudaCheck(cudaMalloc((void**)&s->deltaPs, numParticles * sizeof(glm::vec3)));

	cudaCheck(cudaMalloc((void**)&s, sizeof(solver)));
	cudaCheck(cudaMalloc((void**)&sp, sizeof(solverParams)));
}

void freeParams(solver* s) {
	cudaCheck(cudaFree(s->oldPos));
	cudaCheck(cudaFree(s->newPos));
	cudaCheck(cudaFree(s->velocities));
	cudaCheck(cudaFree(s->densities));
	cudaCheck(cudaFree(s->phases));
	//diffuse goes here
	cudaCheck(cudaFree(s->neighbors));
	cudaCheck(cudaFree(s->numNeighbors));
	cudaCheck(cudaFree(s->gridCells));
	cudaCheck(cudaFree(s->gridCounters));
	cudaCheck(cudaFree(s->contacts));
	cudaCheck(cudaFree(s->numContacts));
	cudaCheck(cudaFree(s->deltaPs));
	cudaCheck(cudaFree(s));
}

#endif