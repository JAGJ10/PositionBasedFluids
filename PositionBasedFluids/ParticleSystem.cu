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
__constant__ float distr[] =
{
	-0.34828757091811f, -0.64246175794046f, -0.15712936555833f, -0.28922267225069f, 0.70090742209037f,
	0.54293139350737f, 0.86755128105523f, 0.68346917800767f, -0.74589352018474f, 0.39762042062246f,
	-0.70243115988673f, -0.85088539675385f, -0.25780126697281f, 0.61167922970451f, -0.8751634423971f,
	-0.12334015086449f, 0.10898816916579f, -0.97167591190509f, 0.89839695948101f, -0.71134930649369f,
	-0.33928178406287f, -0.27579196788175f, -0.5057460942798f, 0.2341509513716f, 0.97802030852904f,
	0.49743173248015f, -0.92212845381448f, 0.088328595779989f, -0.70214782175708f, -0.67050553191011f
};

__device__ float WPoly6(float3 const &pi, float3 const &pj) {
	float3 r = pi - pj;
	float rLen = length(r);
	if (rLen > sp.radius || rLen == 0) {
		return 0;
	}

	return sp.KPOLY * pow((sp.radius * sp.radius - pow(length(r), 2)), 3);
}

__device__ float3 gradWPoly6(float3 const &pi, float3 const &pj) {
	float3 r = pi - pj;
	float rLen = length(r);
	if (rLen > sp.radius || rLen == 0) {
		return make_float3(0.0f);
	}

	float coeff = glm::pow((sp.radius * sp.radius) - (rLen * rLen), 2);
	coeff *= -6 * sp.KPOLY;
	return r * coeff;
}

__device__ float3 WSpiky(float3 const &pi, float3 const &pj) {
	float3 r = pi - pj;
	float rLen = length(r);
	if (rLen > sp.radius || rLen == 0) {
		return make_float3(0.0f);
	}

	float coeff = (sp.radius - rLen) * (sp.radius - rLen);
	coeff *= sp.SPIKY;
	coeff /= rLen;
	return r * -coeff;
}

__device__ float WAirPotential(float3 const &pi, float3 const &pj) {
	float3 r = pi - pj;
	float rLen = length(r);
	if (rLen > sp.radius || rLen == 0) {
		return 0.0f;
	}

	return 1 - (rLen / sp.radius);
}

//Returns the eta vector that points in the direction of the corrective force
__device__ float3 eta(float4* newPos, int* phases, int* neighbors, int* numNeighbors, int &index, float &vorticityMag) {
	float3 eta = make_float3(0.0f);
	for (int i = 0; i < numNeighbors[index]; i++) {
		if (phases[neighbors[(index * sp.MAX_NEIGHBORS) + i]] == 0)
			eta += WSpiky(make_float3(newPos[index]), make_float3(newPos[neighbors[(index * sp.MAX_NEIGHBORS) + i]])) * vorticityMag;
	}

	return eta;
}

//Calculates the vorticity force for a particle
__device__ float3 vorticityForce(float4* newPos, float3* velocities, int* phases, int* neighbors, int* numNeighbors, int index) {
	//Calculate omega_i
	float3 omega = make_float3(0.0f);
	float3 velocityDiff;
	float3 gradient;

	for (int i = 0; i < numNeighbors[index]; i++) {
		if (phases[neighbors[(index * sp.MAX_NEIGHBORS) + i]] == 0) {
			velocityDiff = velocities[neighbors[(index * sp.MAX_NEIGHBORS) + i]] - velocities[index];
			gradient = WSpiky(make_float3(newPos[index]), make_float3(newPos[neighbors[(index * sp.MAX_NEIGHBORS) + i]]));
			omega += cross(velocityDiff, gradient);
		}
	}

	float omegaLength = length(omega);
	if (omegaLength == 0.0f) {
		//No direction for eta
		return make_float3(0.0f);
	}

	float3 etaVal = eta(newPos, phases, neighbors, numNeighbors, index, omegaLength);
	if (etaVal.x == 0 && etaVal.y == 0 && etaVal.z == 0) {
		//Particle is isolated or net force is 0
		return make_float3(0.0f);
	}

	float3 n = normalize(etaVal);

	return (cross(n, omega) * sp.vorticityEps);
}

__device__ float sCorrCalc(float4 &pi, float4 &pj) {
	//Get Density from WPoly6
	float corr = WPoly6(make_float3(pi), make_float3(pj)) / sp.wQH;
	corr *= corr * corr * corr;
	return -sp.K * corr;
}

__device__ float3 xsphViscosity(float4* newPos, float3* velocities, int* phases, int* neighbors, int* numNeighbors, int index) {
	float3 visc = make_float3(0.0f);
	for (int i = 0; i < numNeighbors[index]; i++) {
		if (phases[neighbors[(index * sp.MAX_NEIGHBORS) + i]] == 0) {
			float3 velocityDiff = velocities[neighbors[(index * sp.MAX_NEIGHBORS) + i]] - velocities[index];
			velocityDiff *= WPoly6(make_float3(newPos[index]), make_float3(newPos[neighbors[(index * sp.MAX_NEIGHBORS) + i]]));
			visc += velocityDiff;
		}
	}

	return visc * sp.C;
}

__device__ void confineToBox(float4 &pos) {
	if (pos.x < 0) pos.x = 0.001f;
	else if (pos.x > sp.bounds.x) pos.x = sp.bounds.x - 0.001f;


	if (pos.y < 0) pos.y = 0.001f;
	else if (pos.y > sp.bounds.y) pos.y = sp.bounds.y - 0.001f;

	if (pos.z < 0) pos.z = 0.001f;
	else if (pos.z > sp.bounds.z) pos.z = sp.bounds.z - 0.001f;
}

__device__ int3 getGridPos(float4 pos) {
	return make_int3(int(pos.x / sp.radius) % sp.gridWidth, int(pos.y / sp.radius) % sp.gridHeight, int(pos.z / sp.radius) % sp.gridDepth);
}

__device__ int getGridIndex(int3 pos) {
	return (pos.z * sp.gridHeight * sp.gridWidth) + (pos.y * sp.gridWidth) + pos.x;
}

__global__ void predictPositions(float4* newPos, float3* velocities) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= sp.numParticles) return;

	//update velocity vi = vi + dt * fExt
	velocities[index] += newPos[index].w * sp.gravity * deltaT;

	//predict position x* = xi + dt * vi
	newPos[index] += make_float4(velocities[index] * deltaT, 0);

	confineToBox(newPos[index]);
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

__global__ void updateGrid(float4* newPos, int* gridCells, int* gridCounters) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= sp.numParticles) return;

	int3 pos = getGridPos(newPos[index]);
	int gIndex = getGridIndex(pos);

	int i = atomicAdd(&gridCounters[gIndex], 1);
	i = min(i, sp.MAX_PARTICLES - 1);
	gridCells[gIndex * sp.MAX_PARTICLES + i] = index;
}

__global__ void updateNeighbors(float4* newPos, int* phases, int* gridCells, int* gridCounters, int* neighbors, int* numNeighbors, int* contacts, int* numContacts) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= sp.numParticles) return;
	
	int3 pos = getGridPos(newPos[index]);
	int pIndex;

	for (int z = -1; z < 2; z++) {
		for (int y = -1; y < 2; y++) {
			for (int x = -1; x < 2; x++) {
				int3 n = make_int3(pos.x + x, pos.y + y, pos.z + z);
				if (n.x >= 0 && n.x < sp.gridWidth && n.y >= 0 && n.y < sp.gridHeight && n.z >= 0 && n.z < sp.gridDepth) {
					int gIndex = getGridIndex(n);
					int cellParticles = min(gridCounters[gIndex], sp.MAX_PARTICLES - 1);
					for (int i = 0; i < cellParticles; i++) {
						if (numNeighbors[index] >= sp.MAX_NEIGHBORS) return;

						pIndex = gridCells[gIndex * sp.MAX_PARTICLES + i];
						if (length(make_float3(newPos[index]) - make_float3(newPos[pIndex])) <= sp.radius) {
							neighbors[(index * sp.MAX_NEIGHBORS) + numNeighbors[index]] = pIndex;
							numNeighbors[index]++;
							//if (phases[index] == 0 && phases[pIndex] == 1 && numContacts[index] < sp.MAX_CONTACTS) {
							//	contacts[index * sp.MAX_CONTACTS + numContacts[index]] = pIndex;
							//	numContacts[index]++;
							//}
						}
					}
				}
			}
		}
	}
}

__global__ void particleCollisions(float4* newPos, int* contacts, int* numContacts, float3* deltaPs, float* buffer0) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= sp.numParticles) return;

	for (int i = 0; i < numContacts[index]; i++) {
		int nIndex = contacts[index * sp.MAX_CONTACTS + i];
		if (newPos[nIndex].w == 0) continue;
		float3 dir = make_float3(newPos[index]) - make_float3(newPos[nIndex]);
		float len = length(dir);
		float invMass = newPos[index].w + newPos[nIndex].w;
		float3 dp;
		if ((len - sp.radius) > 0.0f || len == 0.0f || invMass == 0.0f) dp = make_float3(0);
		else dp = (1 / invMass) * (len - sp.radius) * (dir / len);
		deltaPs[index] -= dp;
		buffer0[index]++;

		atomicAdd(&deltaPs[nIndex].x, dp.x);
		atomicAdd(&deltaPs[nIndex].y, dp.y);
		atomicAdd(&deltaPs[nIndex].z, dp.z);
		atomicAdd(&buffer0[nIndex], 1);
	}
}

__global__ void calcDensities(float4* newPos, int* phases, int* neighbors, int* numNeighbors, float* densities) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= sp.numParticles || phases[index] != 0) return;

	float rhoSum = 0.0f;
	for (int i = 0; i < numNeighbors[index]; i++) {
		if (phases[neighbors[(index * sp.MAX_NEIGHBORS) + i]] == 0)
			rhoSum += WPoly6(make_float3(newPos[index]), make_float3(newPos[neighbors[(index * sp.MAX_NEIGHBORS) + i]]));
	}

	densities[index] = rhoSum;
}

__global__ void calcLambda(float4* newPos, int* phases, int* neighbors, int* numNeighbors, float* densities, float* buffer0) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= sp.numParticles || phases[index] != 0) return;

	float densityConstraint = (densities[index] / sp.restDensity) - 1;
	float3 gradientI = make_float3(0.0f);
	float sumGradients = 0.0f;
	for (int i = 0; i < numNeighbors[index]; i++) {
		if (phases[neighbors[(index * sp.MAX_NEIGHBORS) + i]] == 0) {
			//Calculate gradient with respect to j
			float3 gradientJ = WSpiky(make_float3(newPos[index]), make_float3(newPos[neighbors[(index * sp.MAX_NEIGHBORS) + i]])) / sp.restDensity;

			//Add magnitude squared to sum
			sumGradients += pow(length(gradientJ), 2);
			gradientI += gradientJ;
		}
	}

	//Add the particle i gradient magnitude squared to sum
	sumGradients += pow(length(gradientI), 2);
	buffer0[index] = (-1 * densityConstraint) / (sumGradients + sp.lambdaEps);
}

__global__ void calcDeltaP(float4* newPos, int* phases, int* neighbors, int* numNeighbors, float3* deltaPs, float* buffer0) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= sp.numParticles || phases[index] != 0) return;
	deltaPs[index] = make_float3(0);

	float3 deltaP = make_float3(0.0f);
	for (int i = 0; i < numNeighbors[index]; i++) {
		if (phases[neighbors[(index * sp.MAX_NEIGHBORS) + i]] == 0) {
			float lambdaSum = buffer0[index] + buffer0[neighbors[(index * sp.MAX_NEIGHBORS) + i]];
			float sCorr = sCorrCalc(newPos[index], newPos[neighbors[(index * sp.MAX_NEIGHBORS) + i]]);
			deltaP += WSpiky(make_float3(newPos[index]), make_float3(newPos[neighbors[(index * sp.MAX_NEIGHBORS) + i]])) * (lambdaSum + sCorr);

		}
	}

	deltaPs[index] = deltaP / sp.restDensity;
}

__global__ void applyDeltaP(float4* newPos, float3* deltaPs, float* buffer0, int flag) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= sp.numParticles) return;

	//if (buffer0[index] > 0 && flag == 1) newPos[index] += make_float4(deltaPs[index] / buffer0[index], 0);
	//else if (flag == 0) newPos[index] += make_float4(deltaPs[index], 0);
	newPos[index] += make_float4(deltaPs[index], 0);
}

__global__ void updateVelocities(float4* oldPos, float4* newPos, float3* velocities, int* phases, int* neighbors, int* numNeighbors, float3* deltaPs) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= sp.numParticles || phases[index] != 0) return;

	confineToBox(newPos[index]);

	//set new velocity vi = (x*i - xi) / dt
	velocities[index] = (make_float3(newPos[index]) - make_float3(oldPos[index])) / deltaT;

	//apply vorticity confinement
	velocities[index] += vorticityForce(newPos, velocities, phases, neighbors, numNeighbors, index) * deltaT;

	//apply XSPH viscosity
	deltaPs[index] = xsphViscosity(newPos, velocities, phases, neighbors, numNeighbors, index);

	//update position xi = x*i
	oldPos[index] = newPos[index];
}

__global__ void updateXSPHVelocities(float4* newPos, float3* velocities, int* phases, float3* deltaPs) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= sp.numParticles || phases[index] != 0) return;

	velocities[index] += deltaPs[index] * deltaT;
}

/*__global__ void generateFoam(Particle* particles, FoamParticle* foamParticles, int* neighbors, int* numNeighbors, float* densities) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= NUM_PARTICLES || foamCount >= NUM_FOAM) return;

	float velocityDiff = 0.0f;
	for (int i = 0; i < numNeighbors[index]; i++) {
		int nIndex = neighbors[(index * sp.MAX_NEIGHBORS) + i];
		if (index != nIndex) {
			float wAir = WAirPotential(particles[index].newPos, particles[nIndex].newPos);
			float3 xij = glm::normalize(particles[index].newPos - particles[nIndex].newPos);
			float3 vijHat = glm::normalize(particles[index].velocity - particles[nIndex].velocity);
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

		float3 xd = particles[index].newPos + float3(rx * rd, ry * rd, rz * rd);
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

__global__ void clearDeltaP(float3* deltaPs, float* buffer0) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= sp.numParticles) return;

	deltaPs[index] = make_float3(0);
	buffer0[index] = 0;
}

/*__global__ void solveDistance(Particle* particles, DistanceConstraint* dConstraints, int numConstraints, float3* deltaPs, float* buffer3) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= numConstraints) return;

	DistanceConstraint &c = dConstraints[index];
	float3 dir = particles[c.p1].newPos - particles[c.p2].newPos;
	float length = glm::length(dir);
	float invMass = particles[c.p1].invMass + particles[c.p2].invMass;
	float3 dp;
	if (length == 0.0f || invMass == 0.0f) dp = float3(0);
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
	for (int i = 0; i < 4; i++) {
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

__global__ void getParticlePositions(float4* oldPos, float* positions) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= sp.numParticles) return;

	positions[4 * index] = oldPos[index].x;
	positions[4 * index + 1] = oldPos[index].y;
	positions[4 * index + 2] = oldPos[index].z;
	positions[4 * index + 3] = oldPos[index].w;
}

void getPositions(float4* oldPos, float* positions) {
	getParticlePositions<<<dims, blockSize>>>(oldPos, positions);
}

void setParams(solverParams *tempParams) {
	dims = int(ceil(tempParams->numParticles / blockSize));
	gridDims = int(ceil(tempParams->gridSize / blockSize));
	cudaCheck(cudaMemcpyToSymbol(sp, tempParams, sizeof(solverParams)));
}

#endif