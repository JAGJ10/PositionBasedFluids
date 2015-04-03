#include "ParticleSystem.h"
#include "ParticleSystem.cuh"

using namespace std;

static float t = 0.0f;
static int flag = 1;
static int frameCounter = 0;
static const float deltaT = 0.0083f;

ParticleSystem::ParticleSystem() : running(false), moveWall(false), s(new solver) {
	//Initialize cloth particles
	/*float stretchStiffness = 0.9f;
	float bendStiffness = 1.0f;
	float shearStiffness = 0.9f;

	int baseIndex = count;
	int c1, c2, c3, c4;
	for (float i = 1; i < 33; i++) {
		for (float j = 1; j < 33; j++) {
			tempParticles[count].invMass = .5f;
			tempParticles[count].newPos = glm::vec3(float(i) / 20, 2.0f, float(j) / 20);
			tempParticles[count].oldPos = glm::vec3(float(i) / 20, 2.0f, float(j) / 20);
			tempParticles[count].velocity = glm::vec3(0.0f);
			tempParticles[count].phase = 1;
			if (j == 1.0f && i == 1.0f) {
				tempParticles[count].invMass = 0;
				c1 = count;
			}
			if (i == 33 - 1 && j == 1.0f) {
				tempParticles[count].invMass = 0;
				c2 = count;
			}
			if (j == 33 - 1 && i == 33 - 1) {
				tempParticles[count].invMass = 0;
				c3 = count;
			}
			if (i == 1.0f && j == 33 - 1) {
				tempParticles[count].invMass = 0;
				c4 = count;
			}
			count++;
		}
	}
	
	//Horizontal Distance constraints
	p->numConstraints = 0;
	for (int j = 0; j < 32; j++) {
		for (int i = 0; i < 32; i++) {
			int i0 = j * 32 + i;
			if (i > 0) {
				int i1 = j * 32 + i - 1;
				tempdConstraints.push_back(DistanceConstraint(baseIndex + i0, baseIndex + i1, glm::length(tempParticles[baseIndex + i0].oldPos - tempParticles[baseIndex + i1].oldPos), stretchStiffness));
				p->numConstraints++;
			}

			if (i > 1) {
				int i2 = j * 32 + i - 2;
				tempdConstraints.push_back(DistanceConstraint(baseIndex + i0, baseIndex + i2, glm::length(tempParticles[baseIndex + i0].oldPos - tempParticles[baseIndex + i2].oldPos), bendStiffness));
				p->numConstraints++;
			}

			if (j > 0 && i < 31) {
				int iDiag = (j - 1) * 32 + i + 1;
				tempdConstraints.push_back(DistanceConstraint(baseIndex + i0, baseIndex + iDiag, glm::length(tempParticles[baseIndex + i0].oldPos - tempParticles[baseIndex + iDiag].oldPos), shearStiffness));
				p->numConstraints++;
			}

			if (j > 0 && i > 0) {
				int iDiag = (j - 1) * 32 + i - 1;
				tempdConstraints.push_back(DistanceConstraint(baseIndex + i0, baseIndex + iDiag, glm::length(tempParticles[baseIndex + i0].oldPos - tempParticles[baseIndex + iDiag].oldPos), shearStiffness));
				p->numConstraints++;
			}
		}
	}

	//Vertical Distance constraints
	for (int i = 0; i < 32; i++) {
		for (int j = 0; j < 32; j++) {
			int i0 = j * 32 + i;
			if (j > 0) {
				int i1 = (j - 1) * 32 + i;
				tempdConstraints.push_back(DistanceConstraint(baseIndex + i0, baseIndex + i1, glm::length(tempParticles[baseIndex + i0].oldPos - tempParticles[baseIndex + i1].oldPos), stretchStiffness));
				p->numConstraints++;
			}

			if (j > 1) {
				int i1 = (j - 2) * 32 + i;
				tempdConstraints.push_back(DistanceConstraint(baseIndex + i0, baseIndex + i1, glm::length(tempParticles[baseIndex + i0].oldPos - tempParticles[baseIndex + i1].oldPos), bendStiffness));
				p->numConstraints++;
			}
		}
	}

	//Tethers
	float stiffness = -0.5f;
	for (int i = baseIndex; i < count; i++) {
		if (tempParticles[i].invMass > 0) {
			tempdConstraints.push_back(DistanceConstraint(c1, i, glm::length(tempParticles[c1].oldPos - tempParticles[i].oldPos), stiffness));
			tempdConstraints.push_back(DistanceConstraint(c2, i, glm::length(tempParticles[c2].oldPos - tempParticles[i].oldPos), stiffness));
			tempdConstraints.push_back(DistanceConstraint(c3, i, glm::length(tempParticles[c3].oldPos - tempParticles[i].oldPos), stiffness));
			tempdConstraints.push_back(DistanceConstraint(c4, i, glm::length(tempParticles[c4].oldPos - tempParticles[i].oldPos), stiffness));
			p->numConstraints += 4;
		}
	}*/

}

ParticleSystem::~ParticleSystem() {
	cudaCheck(cudaFree(s->oldPos));
	cudaCheck(cudaFree(s->newPos));
	cudaCheck(cudaFree(s->velocities));
	cudaCheck(cudaFree(s->densities));
	cudaCheck(cudaFree(s->phases));
	cudaCheck(cudaFree(s->diffusePos));
	cudaCheck(cudaFree(s->diffuseVelocities));
	cudaCheck(cudaFree(s->neighbors));
	cudaCheck(cudaFree(s->numNeighbors));
	cudaCheck(cudaFree(s->gridCells));
	cudaCheck(cudaFree(s->gridCounters));
	cudaCheck(cudaFree(s->contacts));
	cudaCheck(cudaFree(s->numContacts));
	cudaCheck(cudaFree(s->deltaPs));
	cudaCheck(cudaFree(s->buffer0));
	delete s;
}

void ParticleSystem::initialize(tempSolver &tp, solverParams &tempParams) {
	//General particle info
	cudaCheck(cudaMalloc((void**)&s->oldPos, tempParams.numParticles * sizeof(float4)));
	cudaCheck(cudaMalloc((void**)&s->newPos, tempParams.numParticles * sizeof(float4)));
	cudaCheck(cudaMalloc((void**)&s->velocities, tempParams.numParticles * sizeof(float3)));
	cudaCheck(cudaMalloc((void**)&s->densities, tempParams.numParticles * sizeof(float)));
	cudaCheck(cudaMalloc((void**)&s->phases, tempParams.numParticles * sizeof(int)));
	//Diffuse
	cudaCheck(cudaMalloc((void**)&s->diffusePos, tempParams.numDiffuse * sizeof(float4)));
	cudaCheck(cudaMalloc((void**)&s->diffuseVelocities, tempParams.numDiffuse * sizeof(float3)));
	//Cloth
	cudaCheck(cudaMalloc((void**)&s->clothIndices, tempParams.numConstraints * 2 * sizeof(int)));
	cudaCheck(cudaMalloc((void**)&s->restLengths, tempParams.numConstraints * sizeof(int)));
	cudaCheck(cudaMalloc((void**)&s->stiffness, tempParams.numConstraints * sizeof(int)));
	//Neighbor finding and buffers
	cudaCheck(cudaMalloc((void**)&s->neighbors, tempParams.maxNeighbors * tempParams.numParticles * sizeof(int)));
	cudaCheck(cudaMalloc((void**)&s->numNeighbors, tempParams.numParticles * sizeof(int)));
	cudaCheck(cudaMalloc((void**)&s->gridCells, tempParams.maxParticles * tempParams.gridSize * sizeof(int)));
	cudaCheck(cudaMalloc((void**)&s->gridCounters, tempParams.gridSize * sizeof(int)));
	cudaCheck(cudaMalloc((void**)&s->contacts, tempParams.maxContacts * tempParams.numParticles * sizeof(int)));
	cudaCheck(cudaMalloc((void**)&s->numContacts, tempParams.numParticles * sizeof(int)));
	cudaCheck(cudaMalloc((void**)&s->deltaPs, tempParams.numParticles * sizeof(float3)));
	cudaCheck(cudaMalloc((void**)&s->buffer0, tempParams.numParticles * sizeof(float)));

	cudaCheck(cudaMemset(s->oldPos, 0, tempParams.numParticles * sizeof(float4)));
	cudaCheck(cudaMemset(s->newPos, 0, tempParams.numParticles * sizeof(float4)));
	cudaCheck(cudaMemset(s->velocities, 0, tempParams.numParticles * sizeof(float3)));
	cudaCheck(cudaMemset(s->densities, 0, tempParams.numParticles * sizeof(float)));
	cudaCheck(cudaMemset(s->phases, 0, tempParams.numParticles * sizeof(int)));
	cudaCheck(cudaMemset(s->diffusePos, 0, tempParams.numDiffuse * sizeof(float4)));
	cudaCheck(cudaMemset(s->diffuseVelocities, 0, tempParams.numDiffuse * sizeof(float3)));
	cudaCheck(cudaMemset(s->clothIndices, 0, tempParams.numConstraints * 2 * sizeof(int)));
	cudaCheck(cudaMemset(s->restLengths, 0, tempParams.numConstraints * sizeof(int)));
	cudaCheck(cudaMemset(s->stiffness, 0, tempParams.numConstraints * sizeof(int)));
	cudaCheck(cudaMemset(s->neighbors, 0, tempParams.maxNeighbors * tempParams.numParticles * sizeof(int)));
	cudaCheck(cudaMemset(s->numNeighbors, 0, tempParams.numParticles * sizeof(int)));
	cudaCheck(cudaMemset(s->gridCells, 0, tempParams.maxParticles * tempParams.gridSize * sizeof(int)));
	cudaCheck(cudaMemset(s->gridCounters, 0, tempParams.gridSize * sizeof(int)));

	cudaCheck(cudaMemcpy(s->oldPos, &tp.positions[0], tempParams.numParticles * sizeof(float4), cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(s->newPos, &tp.positions[0], tempParams.numParticles * sizeof(float4), cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(s->velocities, &tp.velocities[0], tempParams.numParticles * sizeof(float3), cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(s->phases, &tp.phases[0], tempParams.numParticles * sizeof(int), cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(s->diffusePos, &tp.diffusePos[0], tempParams.numDiffuse * sizeof(float4), cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(s->diffuseVelocities, &tp.diffuseVelocities[0], tempParams.numDiffuse * sizeof(float3), cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(s->clothIndices, &tp.clothIndices[0], tempParams.numConstraints * 2 * sizeof(int), cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(s->restLengths, &tp.restLengths[0], tempParams.numConstraints * sizeof(int), cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(s->stiffness, &tp.stiffness[0], tempParams.numConstraints * sizeof(int), cudaMemcpyHostToDevice));
	setParams(&tempParams);
}

void ParticleSystem::updateWrapper(solverParams &tempParams) {
	if (running) {
		if (moveWall) {
			if (frameCounter >= 300) {
				//width = (1 - abs(sin((frameCounter - 400) * (deltaT / 1.25f)  * 0.5f * PI)) * 1) + 4;
				t += flag * deltaT / 1.0f;
				if (t >= 1) {
					t = 1;
					flag *= -1;
				} else if (t <= 0) {
					t = 0;
					flag *= -1;
				}

				tempParams.bounds.x = easeInOutQuad(t, tempParams.gridWidth * tempParams.radius, -1.5f, 1.0f);
			}

			frameCounter++;
			setParams(&tempParams);
		}

		update(s, &tempParams);
	}
}

void ParticleSystem::getPositions(float* positionsPtr, int numParticles) {
	cudaCheck(cudaMemcpy(positionsPtr, s->oldPos, numParticles * sizeof(float4), cudaMemcpyDeviceToDevice));
}

void ParticleSystem::getDiffuse(float* diffusePosPtr, float* diffuseVelPtr, int numDiffuse) {
	cudaCheck(cudaMemset(diffusePosPtr, 0, numDiffuse * sizeof(float4)));
	cudaCheck(cudaMemcpy(diffusePosPtr, s->diffusePos, numDiffuse * sizeof(float4), cudaMemcpyDeviceToDevice));
	cudaCheck(cudaMemcpy(diffuseVelPtr, s->diffuseVelocities, numDiffuse * sizeof(float3), cudaMemcpyDeviceToDevice));
}

int ParticleSystem::getIndex(float i, float j) {
	return int(i * 20 + j);
}

float ParticleSystem::easeInOutQuad(float t, float b, float c, float d) {
	t /= d / 2;
	if (t < 1) return c / 2 * t*t + b;
	t--;
	return -c / 2 * (t*(t - 2) - 1) + b;
};