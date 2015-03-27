#include "ParticleSystem.h"
#include "ParticleSystem.cuh"

using namespace std;

static float t = 0.0f;
static int flag = 1;
static int frameCounter = 0;

//---------------------Cloth Constants----------------------
//static const int SOLVER_ITERATIONS = 16;
//static const float kStretch = 0.75f;
//static const float kDamp = 0.05f;
//static const float kLin = 1.0f - glm::pow(1.0f - kStretch, 1.0f / SOLVER_ITERATIONS);
//static const float globalK = 0.0f; //0 means you aren't forcing it into a shape (like a plant)

ParticleSystem::ParticleSystem() {
	p = new Buffers;
	//Initialize particles
	gpuErrchk(cudaMalloc((void**)&p->particles, NUM_PARTICLES * sizeof(Particle)));
	//gpuErrchk(cudaMalloc((void**)&foamParticles, NUM_FOAM * sizeof(FoamParticle)));
	gpuErrchk(cudaMalloc((void**)&p->neighbors, MAX_NEIGHBORS * NUM_PARTICLES * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&p->numNeighbors, NUM_PARTICLES * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&p->gridCells, MAX_PARTICLES * gridSize * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&p->gridCounters, gridSize * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&p->contacts, MAX_CONTACTS * NUM_PARTICLES * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&p->numContacts, NUM_PARTICLES * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&p->deltaPs, NUM_PARTICLES * sizeof(glm::vec3)));
	gpuErrchk(cudaMalloc((void**)&p->buffer1, NUM_PARTICLES * sizeof(glm::vec3)));
	gpuErrchk(cudaMalloc((void**)&p->densities, NUM_PARTICLES * sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&p->buffer3, NUM_PARTICLES * sizeof(float)));

	//Clear memory in case it's left over from last time?
	gpuErrchk(cudaMemset(p->particles, 0, NUM_PARTICLES * sizeof(Particle)));
	//gpuErrchk(cudaMemset(foamParticles, 0, NUM_FOAM * sizeof(FoamParticle)));
	gpuErrchk(cudaMemset(p->neighbors, 0, MAX_NEIGHBORS * NUM_PARTICLES * sizeof(int)));
	gpuErrchk(cudaMemset(p->numNeighbors, 0, NUM_PARTICLES * sizeof(int)));
	gpuErrchk(cudaMemset(p->gridCells, 0, MAX_PARTICLES * gridSize * sizeof(int)));
	gpuErrchk(cudaMemset(p->gridCounters, 0, gridSize * sizeof(int)));
	gpuErrchk(cudaMemset(p->contacts, 0, MAX_CONTACTS * NUM_PARTICLES * sizeof(int)));
	gpuErrchk(cudaMemset(p->numContacts, 0, NUM_PARTICLES * sizeof(int)));
	gpuErrchk(cudaMemset(p->deltaPs, 0, NUM_PARTICLES * sizeof(glm::vec3)));
	gpuErrchk(cudaMemset(p->buffer1, 0, NUM_PARTICLES * sizeof(glm::vec3)));
	gpuErrchk(cudaMemset(p->densities, 0, NUM_PARTICLES * sizeof(float)));
	gpuErrchk(cudaMemset(p->buffer3, 0, NUM_PARTICLES * sizeof(float)));

	tempParticles = new Particle[NUM_PARTICLES];

	int count = 0;
	for (int i = 1; i < 33; i += 1) {
		for (int j = 48; j < 49; j += 1) {
			for (int k = 12; k < 16; k += 1) {
				tempParticles[count].invMass = 1;
				tempParticles[count].newPos = glm::vec3(float(i) / 20, float(j) / 20, float(k) / 20);
				tempParticles[count].oldPos = glm::vec3(float(i) / 20, float(j) / 20, float(k) / 20);
				tempParticles[count].velocity = glm::vec3(0.0f);
				tempParticles[count].phase = 0;
				count++;
			}
		}
	}
	
	//srand((unsigned int)time(0));

	//Initialize cloth particles
	float stretchStiffness = 0.9f;
	float bendStiffness = 1.0f;
	float shearStiffness = 0.9f;

	int baseIndex = count;
	int c1, c2, c3, c4;
	for (float i = 1; i < 33; i++) {
		for (float j = 1; j < 33; j++) {
			tempParticles[count].invMass = 1;
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
	float stiffness = -0.8f;
	for (int i = baseIndex; i < count; i++) {
		if (tempParticles[i].invMass > 0) {
			//tempdConstraints.push_back(DistanceConstraint(c1, i, glm::length(tempParticles[c1].oldPos - tempParticles[i].oldPos), stiffness));
			//tempdConstraints.push_back(DistanceConstraint(c2, i, glm::length(tempParticles[c2].oldPos - tempParticles[i].oldPos), stiffness));
			//tempdConstraints.push_back(DistanceConstraint(c3, i, glm::length(tempParticles[c3].oldPos - tempParticles[i].oldPos), stiffness));
			//tempdConstraints.push_back(DistanceConstraint(c4, i, glm::length(tempParticles[c4].oldPos - tempParticles[i].oldPos), stiffness));
			//p->numConstraints += 4;
		}
	}
	
	gpuErrchk(cudaMemcpy(p->particles, tempParticles, NUM_PARTICLES * sizeof(Particle), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMalloc((void**)&p->dConstraints, p->numConstraints * sizeof(DistanceConstraint)));
	gpuErrchk(cudaMemcpy(p->dConstraints, tempdConstraints.data(), p->numConstraints * sizeof(DistanceConstraint), cudaMemcpyHostToDevice));
	delete[] tempParticles;
	tempdConstraints.clear();
}

ParticleSystem::~ParticleSystem() {
	gpuErrchk(cudaFree(p->particles));
	gpuErrchk(cudaFree(p->dConstraints));
	//gpuErrchk(cudaFree(foamParticles));
	//gpuErrchk(cudaFree(freeList));
	gpuErrchk(cudaFree(p->neighbors));
	gpuErrchk(cudaFree(p->numNeighbors));
	gpuErrchk(cudaFree(p->gridCells));
	gpuErrchk(cudaFree(p->gridCounters));
	gpuErrchk(cudaFree(p->contacts));
	gpuErrchk(cudaFree(p->numContacts));
	gpuErrchk(cudaFree(p->deltaPs));
	gpuErrchk(cudaFree(p->buffer1));
	gpuErrchk(cudaFree(p->densities));
	gpuErrchk(cudaFree(p->buffer3));
	delete p;
}

void ParticleSystem::updateWrapper() {
	update(p);
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

void ParticleSystem::setVBOWrapper(float* fluidPositions, float* clothPositions) {
	setVBO(p->particles, fluidPositions, clothPositions);
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
}*/

int ParticleSystem::getIndex(float i, float j) {
	return int(i * 20 + j);
}

float ParticleSystem::easeInOutQuad(float t, float b, float c, float d) {
	t /= d / 2;
	if (t < 1) return c / 2 * t*t + b;
	t--;
	return -c / 2 * (t*(t - 2) - 1) + b;
};