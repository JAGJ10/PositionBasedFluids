#include "ParticleSystem.h"
#include "ParticleSystem.cuh"

using namespace std;

static float t = 0.0f;
static int flag = 1;
static int frameCounter = 0;

//---------------------Cloth Constants----------------------
static const int SOLVER_ITERATIONS = 16;
static const float kStretch = 0.75f;
static const float kDamp = 0.05f;
static const float kLin = 1.0f - glm::pow(1.0f - kStretch, 1.0f / SOLVER_ITERATIONS);
static const float globalK = 0.0f; //0 means you aren't forcing it into a shape (like a plant)

ParticleSystem::ParticleSystem() {
	//Initialize particles
	gpuErrchk(cudaMalloc((void**)&particles, NUM_PARTICLES * sizeof(Particle)));
	//gpuErrchk(cudaMalloc((void**)&foamParticles, NUM_FOAM * sizeof(FoamParticle)));
	//gpuErrchk(cudaMalloc((void**)&freeList, NUM_FOAM * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&neighbors, MAX_NEIGHBORS * NUM_PARTICLES * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&numNeighbors, NUM_PARTICLES * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&gridCells, MAX_PARTICLES * gridSize * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&gridCounters, gridSize * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&buffer0, NUM_PARTICLES * sizeof(glm::vec3)));
	gpuErrchk(cudaMalloc((void**)&buffer1, NUM_PARTICLES * sizeof(glm::vec3)));
	gpuErrchk(cudaMalloc((void**)&densities, NUM_PARTICLES * sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&buffer3, NUM_PARTICLES * sizeof(float)));

	//Clear memory in case it's left over from last time?
	gpuErrchk(cudaMemset(particles, 0, NUM_PARTICLES * sizeof(Particle)));
	//gpuErrchk(cudaMemset(foamParticles, 0, NUM_FOAM * sizeof(FoamParticle)));
	//gpuErrchk(cudaMemset(freeList, 0, NUM_FOAM * sizeof(int)));
	gpuErrchk(cudaMemset(neighbors, 0, MAX_NEIGHBORS * NUM_PARTICLES * sizeof(int)));
	gpuErrchk(cudaMemset(numNeighbors, 0, NUM_PARTICLES * sizeof(int)));
	gpuErrchk(cudaMemset(gridCells, 0, MAX_PARTICLES * gridSize * sizeof(int)));
	gpuErrchk(cudaMemset(gridCounters, 0, gridSize * sizeof(int)));
	gpuErrchk(cudaMemset(buffer0, 0, NUM_PARTICLES * sizeof(glm::vec3)));
	gpuErrchk(cudaMemset(buffer1, 0, NUM_PARTICLES * sizeof(glm::vec3)));
	gpuErrchk(cudaMemset(densities, 0, NUM_PARTICLES * sizeof(float)));
	gpuErrchk(cudaMemset(buffer3, 0, NUM_PARTICLES * sizeof(float)));

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

	//Initialize cloth particles
	/*count = 0;
	for (float i = 0; i < cols; i++) {
		for (float j = 0; j < cols; j++) {
			clothParticles.push_back(Particle(glm::vec3(i / cols, 4, j / cols), 1, count, 1));
			if (j == 0.0f && i == 0.0f) clothParticles.back().invMass = 0;
			if (i == cols - 1 && j == 0.0f) clothParticles.back().invMass = 0;
			if (j == cols - 1 && i == cols - 1) clothParticles.back().invMass = 0;
			if (i == 0.0f && j == cols - 1) clothParticles.back().invMass = 0;
			count++;
		}
	}
	//Distance constraints
	for (float i = 0; i < cols; i++) {
		for (float j = 0; j < cols; j++) {
			if (j > 0) dConstraints.push_back(DistanceConstraint(&getIndex(i, j), &getIndex(i, j - 1)));
			if (i > 0) dConstraints.push_back(DistanceConstraint(&getIndex(i, j), &getIndex(i - 1, j)));
		}
	}
	//Need shearing constraint?
	//Bending constraints
	for (float i = 0; i < cols; i++) {
		for (float j = 0; j < cols - 2; j++) {
			bConstraints.push_back(BendingConstraint(&getIndex(i, j), &getIndex(i, j + 1), &getIndex(i, j + 2)));
		}
	}
	for (float i = 0; i < cols - 2; i++) {
		for (float j = 0; j < cols; j++) {
			bConstraints.push_back(BendingConstraint(&getIndex(i, j), &getIndex(i + 1, j), &getIndex(i + 2, j)));
		}
	}*/
}

ParticleSystem::~ParticleSystem() {
	gpuErrchk(cudaFree(particles));
	//gpuErrchk(cudaFree(foamParticles));
	//gpuErrchk(cudaFree(freeList));
	gpuErrchk(cudaFree(neighbors));
	gpuErrchk(cudaFree(numNeighbors));
	gpuErrchk(cudaFree(gridCells));
	gpuErrchk(cudaFree(gridCounters));
	gpuErrchk(cudaFree(buffer0));
	gpuErrchk(cudaFree(buffer1));
	gpuErrchk(cudaFree(densities));
	gpuErrchk(cudaFree(buffer3));
}

void ParticleSystem::updateWrapper() {
	update(particles, gridCells, gridCounters, neighbors, numNeighbors, buffer0, buffer1, densities, buffer3);
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

void ParticleSystem::setVBOWrapper(float* positionVBO) {
	setVBO(particles, positionVBO);
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

/*void ParticleSystem::clothUpdate() {
	updateClothPositions();
	for (auto &p : clothParticles) {
		//update velocity vi = vi + dt * wi * fext
		p.velocity += deltaT * p.invMass * GRAVITY;
	}
	//Dampening
	glm::vec3 xcm = glm::vec3(0.0f);
	glm::vec3 vcm = glm::vec3(0.0f);
	float sumM = 0.0f;
	for (auto &p : clothParticles) {
		if (p.invMass != 0) {
			xcm += p.oldPos * (1 / p.invMass);
			vcm += p.velocity * (1 / p.invMass);
			sumM += (1 / p.invMass);
		}
	}
	xcm /= sumM;
	vcm /= sumM;
	glm::mat3 I = glm::mat3(1.0f);
	glm::vec3 L = glm::vec3(0.0f);
	glm::vec3 omega = glm::vec3(0.0f);
	for (int i = 0; i < clothParticles.size(); i++) {
		Particle &p = clothParticles.at(i);
		if (p.invMass > 0) {
			buffer1[i] = p.oldPos - xcm; //ri
			L += glm::cross(buffer1[i], (1 / p.invMass) * p.velocity);
			glm::mat3 temp = glm::mat3(0, buffer1[i].z, -buffer1[i].y,
				-buffer1[i].z, 0, buffer1[i].x,
				buffer1[i].y, -buffer1[i].x, 0);
			I += (temp*glm::transpose(temp)) * (1 / p.invMass);
		}
	}
	omega = glm::inverse(I) * L;
	// deltaVi = vcm + (omega x ri) - vi
	// vi <- vi + kDamp * deltaVi
	for (int i = 0; i < clothParticles.size(); i++) {
		Particle &p = clothParticles.at(i);
		if (p.invMass > 0) {
			glm::vec3 deltaVi = vcm + glm::cross(omega, buffer1[i]) - p.velocity;
			p.velocity += kDamp * deltaVi;
		}
	}
	//Predict new positions -> pi = xi + dt * vi
	for (int i = 0; i < clothParticles.size(); i++) {
		Particle &p = clothParticles.at(i);
		if (p.invMass == 0) {
			p.newPos = p.oldPos;
		}
		else {
			p.newPos = p.oldPos + (p.velocity * deltaT);
		}
	}
	//Collision with ground
	for (int i = 0; i < clothParticles.size(); i++) {
		Particle &p = clothParticles.at(i);
		if (p.newPos.y < 0) {
			p.newPos.y = 0;
		}
	}
	for (int si = 0; si < SOLVER_ITERATIONS; si++) {
		//Stretching/Distance constraints
		for (auto &c : dConstraints) {
			glm::vec3 dir = c.p1->newPos - c.p2->newPos;
			float length = glm::length(dir);
			float invMass = c.p1->invMass + c.p2->invMass;
			glm::vec3 deltaP = (1 / invMass) * (length - c.restLength) * (dir / length) * kLin;
			if (c.p1->invMass > 0) c.p1->newPos -= deltaP * c.p1->invMass;
			if (c.p2->invMass > 0) c.p2->newPos += deltaP * c.p2->invMass;
		}
		//Bending constraints
		for (auto &c : bConstraints) {
			glm::vec3 center = (1.0f / 3.0f) * (c.p1->newPos + c.p2->newPos + c.p3->newPos);
			glm::vec3 dir = c.p2->newPos - center;
			float d = glm::length(dir);
			float diff;
			if (d != 0.0f) diff = 1.0f - ((globalK + c.restLength) / d);
			else diff = 0;
			glm::vec3 force = dir * diff;
			glm::vec3 b0 = kLin * ((2.0f * c.p1->invMass) / c.w) * force;
			glm::vec3 b1 = kLin * ((2.0f * c.p3->invMass) / c.w) * force;
			glm::vec3 delV = kLin * ((-4.0f * c.p2->invMass) / c.w) * force;
			if (c.p1->invMass > 0) c.p1->newPos += b0;
			if (c.p2->invMass > 0) c.p2->newPos += delV;
			if (c.p3->invMass > 0) c.p3->newPos += b1;
		}
	}
	for (auto &p : clothParticles) {
		// vi <- (pi - xi) / dt
		p.velocity = (p.newPos - p.oldPos) / deltaT;
		// xi <- pi
		p.oldPos = p.newPos;
	}
}

Particle& ParticleSystem::getIndex(float i, float j) {
	return clothParticles.at(int(i * cols + j));
}*/

float ParticleSystem::easeInOutQuad(float t, float b, float c, float d) {
	t /= d / 2;
	if (t < 1) return c / 2 * t*t + b;
	t--;
	return -c / 2 * (t*(t - 2) - 1) + b;
};