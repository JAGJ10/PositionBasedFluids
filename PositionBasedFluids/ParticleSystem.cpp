#define NOMINMAX
#include "ParticleSystem.h"
#include "ParticleSystem.cuh"

using namespace std;

static float t = 0.0f;
static int flag = 1;
static int frameCounter = 0;

ParticleSystem::ParticleSystem() {
	//Initialize particles
	gpuErrchk(cudaMalloc((void**)&particles, NUM_PARTICLES * sizeof(Particle)));
	gpuErrchk(cudaMalloc((void**)&foamParticles, NUM_FOAM * sizeof(FoamParticle)));
	gpuErrchk(cudaMalloc((void**)&neighbors, MAX_NEIGHBORS * NUM_PARTICLES * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&numNeighbors, NUM_PARTICLES * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&gridCells, MAX_PARTICLES * gridSize * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&gridCounters, gridSize * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&buffer1, NUM_PARTICLES * sizeof(glm::vec3)));
	gpuErrchk(cudaMalloc((void**)&buffer2, NUM_PARTICLES * sizeof(float)));

	//Clear memory in case it's left over from last time?
	gpuErrchk(cudaMemset(particles, 0, NUM_PARTICLES * sizeof(Particle)));
	gpuErrchk(cudaMemset(foamParticles, 0, NUM_FOAM * sizeof(FoamParticle)));
	gpuErrchk(cudaMemset(neighbors, 0, MAX_NEIGHBORS * NUM_PARTICLES * sizeof(int)));
	gpuErrchk(cudaMemset(numNeighbors, 0, NUM_PARTICLES * sizeof(int)));
	gpuErrchk(cudaMemset(gridCells, 0, MAX_PARTICLES * gridSize * sizeof(int)));
	gpuErrchk(cudaMemset(gridCounters, 0, gridSize * sizeof(int)));
	gpuErrchk(cudaMemset(buffer1, 0, NUM_PARTICLES * sizeof(glm::vec3)));
	gpuErrchk(cudaMemset(buffer2, 0, NUM_PARTICLES * sizeof(float)));

	tempParticles = new Particle[NUM_PARTICLES];

	int count = 0;
	for (int i = 0; i < 30; i += 1) {
		for (int j = 0; j < 64; j += 1) {
			for (int k = 20; k < 52; k += 1) {
				tempParticles[count].invMass = 1;
				tempParticles[count].newPos = glm::vec3(float(i) / 20, float(j) / 20, float(k) / 20);
				tempParticles[count].oldPos = glm::vec3(float(i) / 20, float(j) / 20, float(k) / 20);
				tempParticles[count].velocity = glm::vec3(0.0f);
				tempParticles[count].phase = 0;
				count++;
			}
		}
	}
	
	gpuErrchk(cudaMemcpy(particles, tempParticles, NUM_PARTICLES * sizeof(Particle), cudaMemcpyHostToDevice));
	delete[] tempParticles;
	//srand((unsigned int)time(0));
}

ParticleSystem::~ParticleSystem() {
	gpuErrchk(cudaFree(particles));
	gpuErrchk(cudaFree(neighbors));
	gpuErrchk(cudaFree(numNeighbors));
	gpuErrchk(cudaFree(gridCells));
	gpuErrchk(cudaFree(gridCounters));
	gpuErrchk(cudaFree(buffer1));
	gpuErrchk(cudaFree(buffer2));
}

void ParticleSystem::updateWrapper() {
	update(particles, gridCells, gridCounters, neighbors, numNeighbors, buffer1, buffer2);
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
}

void ParticleSystem::setVBOWrapper(float* vboPtr) {
	setVBO(particles, vboPtr);
}

void ParticleSystem::confineToBox(FoamParticle &p) {
	/*if (p.pos.x < 0 || p.pos.x > width) {
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
	}*/
}

void ParticleSystem::updatePositions2() {
	/*#pragma omp parallel for num_threads(8)
	for (int i = 0; i < particles.size(); i++) {
		Particle &p = particles.at(i);
		//p.sumWeight = 0.0f;
		//p.weightedPos = glm::vec3(0.0f);
	}*/

	//for (auto &p : particles) fluidPositions.push_back(getWeightedPosition(p));
	//for (auto &p : particles) fluidPositions.push_back(p.oldPos);
	
	//for (int i = 0; i < foam.size(); i++) {
		//FoamParticle &p = foam.at(i);
		//int r = rand() % foam.size();
		//foamPositions.push_back(glm::vec4(p.pos.x, p.pos.y, p.pos.z, (p.type * 1000) + float(i) + abs(p.lifetime - lifetime) / lifetime));
	//}
}

void ParticleSystem::updateFoam() {
	//Kill dead foam
	/*for (int i = 0; i < foam.size(); i++) {
		FoamParticle &p = foam.at(i);
		if (p.type == 2) {
			p.lifetime -= deltaT;
			if (p.lifetime <= 0) {
				foam.erase(foam.begin() + i);
				i--;
			}
		}
	}

	//Update velocities
	#pragma omp parallel for num_threads(8)
	for (int i = 0; i < foam.size(); i++) {
		FoamParticle &p = foam.at(i);
		confineToBox(p);

		glm::ivec3 pos = p.pos; //* 10;
		glm::vec3 vfSum = glm::vec3(0.0f);
		float kSum = 0;
		int numNeighbors = 0;
		for (auto &c : grid.cells[pos.x][pos.y][pos.z].neighbors) {
			for (auto &n : c->particles) {
				if (glm::distance(p.pos, n->newPos) <= H) {
					numNeighbors++;
					float k = WPoly6(p.pos, n->newPos);
					vfSum += n->velocity * k;
					kSum += k;
				}
			}
		}

		if (numNeighbors >= 8) p.type = 2;
		else p.type = 1;

		if (p.type == 1) {
			//Spray
			p.velocity.x *= 0.8f;
			p.velocity.z *= 0.8f;
			p.velocity += GRAVITY * deltaT;
			p.pos += p.velocity * deltaT;
		} else if (p.type == 2) {
			//Foam
			p.pos += (1.0f * (vfSum / kSum)) * deltaT;
		}
	}*/
}

void ParticleSystem::generateFoam() {
	/*for (int i = 0; i < particles.size(); i++) {
		Particle &p = particles.at(i);
		float velocityDiff = 0.0f;
		for (auto &n : p.neighbors) {
			if (p.newPos != n->newPos) {
				float wAir = WAirPotential(p.newPos, n->newPos);
				glm::vec3 xij = glm::normalize(p.newPos - n->newPos);
				glm::vec3 vijHat = glm::normalize(p.velocity - n->velocity);
				velocityDiff += glm::length(p.velocity - n->velocity) * (1 - glm::dot(vijHat, xij)) * wAir;
			}
		}

		float ek = 0.5f * glm::length2(p.velocity);

		float potential = velocityDiff * ek * glm::max(1.0f - (1.0f * buffer2[i] / REST_DENSITY), 0.0f);

		int nd = 0;
		if (potential > 0.7f) nd = 5 + (rand() % 30);

		for (int i = 0; i < nd; i++) {
			float rx = (0.05f + static_cast <float> (rand()) / static_cast <float> (RAND_MAX / 0.9f)) * H;
			float ry = (0.05f + static_cast <float> (rand()) / static_cast <float> (RAND_MAX / 0.9f)) * H;
			float rz = (0.05f + static_cast <float> (rand()) / static_cast <float> (RAND_MAX / 0.9f)) * H;
			int rd = rand() > 0.5 ? 1 : -1;

			glm::vec3 xd = p.newPos + glm::vec3(rx*rd, ry*rd, rz*rd);
			glm::vec3 vd = p.velocity;     

			glm::ivec3 pos = p.newPos; //* 10;
			int type;
			int numNeighbors = int(p.neighbors.size()) + 1;

			if (numNeighbors < 8) type = 1;
			else type = 2;

			foam.push_back(FoamParticle(xd, vd, lifetime, type));

			confineToBox(foam.back());
		}
	}*/
}


float ParticleSystem::easeInOutQuad(float t, float b, float c, float d) {
	t /= d / 2;
	if (t < 1) return c / 2 * t*t + b;
	t--;
	return -c / 2 * (t*(t - 2) - 1) + b;
};