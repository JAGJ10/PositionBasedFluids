#define NOMINMAX
#include "ParticleSystem.h"

using namespace std;

//---------------------Fluid Constants----------------------
static const float deltaT = 0.0083f;
static const float PI = 3.14159265358979323846f;
static const glm::vec3 GRAVITY = glm::vec3(0, -9.8f, 0);
static const int PRESSURE_ITERATIONS = 4;
static const float H = 0.1f;
static const float KPOLY = 315 / (64 * PI * glm::pow(H, 9));
static const float SPIKY = 45 / (PI * glm::pow(H, 6));
static const float REST_DENSITY = 6378.0f;
static const float EPSILON_LAMBDA = 600.0f;
static const float EPSILON_VORTICITY = 0.0001f;
static const float C = 0.01f;
static const float K = 0.00001f;
static const float deltaQMag = 0.3f * H;
static const float wQH = KPOLY * glm::pow((H * H - deltaQMag * deltaQMag), 3);
static const float lifetime = 1.0f;
static const float offset = 0.05f;
static const float cols = 1.0f / offset;

static float width = 8;
static float height = 8;
static float depth = 7;

//---------------------Cloth Constants----------------------
static const int SOLVER_ITERATIONS = 16;
static const float kBend = 0.5f; //unused
static const float kStretch = 0.75f;
static const float kDamp = 0.05f;
static const float kLin = 1.0f - glm::pow(1.0f - kStretch, 1.0f / SOLVER_ITERATIONS);
static const float globalK = 0.0f; //0 means you aren't forcing it into a shape (like a plant)

static vector<glm::vec3> buffer1;
static vector<float> buffer2;

static float t = 0.0f;
static int flag = 1;
static int frameCounter = 0;

ParticleSystem::ParticleSystem() : grid((int)width, (int)height, (int)depth) {
	//Initialize fluid particles
	int count = 0;
	/*for (float i = 0; i < 3; i += .05f) {
		for (float j = 0; j < 3; j += .05f) {
			for (float k = 1; k < 4; k += .05f) {
				particles.push_back(Particle(glm::vec3(i, j, k), 1.0f, count, 0));
				count++;
			}
		}
	}*/

	//Initialize cloth particles
	count = 0;
	for (float i = 0; i < cols; i++) {
		for (float j = 0; j < cols; j++) {
			clothParticles.push_back(Particle(glm::vec3(i / cols, 4, j / cols), 1, count, 1));
			if (j == 0.0f && i == 0.0f) clothParticles.back().invMass = 0;
			if (i == cols - 1 && j == 0.0f) clothParticles.back().invMass = 0;
			//if (j == cols - 1 && i == cols - 1) clothParticles.back().invMass = 0;
			//if (i == 0.0f && j == cols - 1) clothParticles.back().invMass = 0;
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
	}

	//foam.reserve(2000000);
	//foamPositions.reserve(2000000);
	//fluidPositions.reserve(particles.capacity());
	//buffer1.resize(particles.capacity());
	buffer1.resize(clothParticles.capacity());
	//buffer2.resize(particles.capacity());

	srand((unsigned int)time(0));
}

ParticleSystem::~ParticleSystem() {}

void ParticleSystem::update() {
	//Move wall
	if (frameCounter >= 500) {
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
	frameCounter++;

	//------------------WATER-----------------
	#pragma omp parallel for num_threads(8)
	for (int i = 0; i < particles.size(); i++) {
		Particle &p = particles.at(i);

		//update velocity vi = vi + dt * fExt
		p.velocity += GRAVITY * deltaT;

		//predict position x* = xi + dt * vi
		p.newPos += p.velocity * deltaT;

		confineToBox(p);
	}

	//Update neighbors
	grid.updateCells(particles);
	setNeighbors();

	//Needs to be after neighbor finding for weighted positions
	updatePositions();

	for (int pi = 0; pi < PRESSURE_ITERATIONS; pi++) {
		//set lambda
		#pragma omp parallel for num_threads(8)
		for (int i = 0; i < particles.size(); i++) {
			Particle &p = particles.at(i);
			buffer2[i] = lambda(p, p.neighbors);
		}

		//calculate deltaP
		#pragma omp parallel for num_threads(8)
		for (int i = 0; i < particles.size(); i++) {
			Particle &p = particles.at(i);
			glm::vec3 deltaP = glm::vec3(0.0f);
			for (auto &n : p.neighbors) {
				float lambdaSum = buffer2[i] + buffer2[n->index];
				float sCorr = sCorrCalc(p, n);
				deltaP += WSpiky(p.newPos, n->newPos) * (lambdaSum + sCorr);
			}

			buffer1[i] = deltaP / REST_DENSITY;
		}

		//update position x*i = x*i + deltaPi
		#pragma omp parallel for num_threads(8)
		for (int i = 0; i < particles.size(); i++) {
			Particle &p = particles.at(i);
			p.newPos += buffer1[i];
		}
	}

	#pragma omp parallel for num_threads(8)
	for (int i = 0; i < particles.size(); i++) {
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
	}

	//Set new velocity
	#pragma omp parallel for num_threads(8)
	for (int i = 0; i < particles.size(); i++) {
		Particle &p = particles.at(i);
		p.velocity += buffer1[i] * deltaT;
	}

	//----------------FOAM-----------------
	updateFoam();
	calcDensities();
	generateFoam();
}

float ParticleSystem::WPoly6(glm::vec3 &pi, glm::vec3 &pj) {
	glm::vec3 r = pi - pj;
	float rLen = glm::length(r);
	if (rLen > H || rLen == 0) {
		return 0;
	}

	return KPOLY * glm::pow((H * H - glm::length2(r)), 3);
}

glm::vec3 ParticleSystem::gradWPoly6(glm::vec3 &pi, glm::vec3 &pj) {
	glm::vec3 r = pi - pj;
	float rLen = glm::length(r);
	if (rLen > H || rLen == 0) {
		return glm::vec3(0.0f);
	}

	float coeff = glm::pow((H * H) - (rLen * rLen), 2);
	coeff *= -6 * KPOLY;
	return r * coeff;
}

glm::vec3 ParticleSystem::WSpiky(glm::vec3 &pi, glm::vec3 &pj) {
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

float ParticleSystem::WAirPotential(glm::vec3 &pi, glm::vec3 &pj) {
	glm::vec3 r = pi - pj;
	float rLen = glm::length(r);
	if (rLen > H || rLen == 0) {
		return 0.0f;
	}

	return 1 - (rLen / H);
}

//Calculate the lambda value for pressure correction
float ParticleSystem::lambda(Particle &p, vector<Particle*> &neighbors) {
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
	return ((-1) * densityConstraint) / (sumGradients + EPSILON_LAMBDA);
}

//Returns density constraint of a particle
float ParticleSystem::calcDensityConstraint(Particle &p, vector<Particle*> &neighbors) {
	float sum = 0.0f;
	for (auto &n : neighbors) {
		sum += WPoly6(p.newPos, n->newPos);
	}

	return (sum / REST_DENSITY) - 1;
}

//Returns the eta vector that points in the direction of the corrective force
glm::vec3 ParticleSystem::eta(Particle &p, float &vorticityMag) {
	glm::vec3 eta = glm::vec3(0.0f);
	for (auto &n : p.neighbors) {
		eta += WSpiky(p.newPos, n->newPos) * vorticityMag;
	}

	return eta;
}

//Calculates the vorticity force for a particle
glm::vec3 ParticleSystem::vorticityForce(Particle &p) {
	//Calculate omega_i
	glm::vec3 omega = glm::vec3(0.0f);
	glm::vec3 velocityDiff;
	glm::vec3 gradient;

	for (auto &n : p.neighbors) {
		velocityDiff = n->velocity - p.velocity;
		gradient = WSpiky(p.newPos, n->newPos);
		omega += glm::cross(velocityDiff, gradient);
	}

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

float ParticleSystem::sCorrCalc(Particle &pi, Particle* &pj) {
	//Get Density from WPoly6
	float corr = WPoly6(pi.newPos, pj->newPos) / wQH;
	corr *= corr * corr * corr;
	return -K * corr;
}

glm::vec3 ParticleSystem::xsphViscosity(Particle &p) {
	glm::vec3 visc = glm::vec3(0.0f);
	for (auto &n : p.neighbors) {
		glm::vec3 velocityDiff = n->velocity - p.velocity;
		velocityDiff *= WPoly6(p.newPos, n->newPos);
		visc += velocityDiff;
	}

	return visc * C;
}

void ParticleSystem::confineToBox(Particle &p) {
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

void ParticleSystem::confineToBox(FoamParticle &p) {
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

void ParticleSystem::updatePositions() {
	fluidPositions.clear();
	foamPositions.clear();

	/*#pragma omp parallel for num_threads(8)
	for (int i = 0; i < particles.size(); i++) {
		Particle &p = particles.at(i);
		//p.sumWeight = 0.0f;
		//p.weightedPos = glm::vec3(0.0f);
	}*/

	//for (auto &p : particles) fluidPositions.push_back(getWeightedPosition(p));
	for (auto &p : particles) fluidPositions.push_back(p.oldPos);
	
	for (int i = 0; i < foam.size(); i++) {
		FoamParticle &p = foam.at(i);
		int r = rand() % foam.size();
		foamPositions.push_back(glm::vec4(p.pos.x, p.pos.y, p.pos.z, (p.type * 1000) + float(i) + abs(p.lifetime - lifetime) / lifetime));
	}
}

void ParticleSystem::updateClothPositions() {
	clothPositions.clear();
	for (auto &p : clothParticles) clothPositions.push_back(p.oldPos);
}

vector<glm::vec3>& ParticleSystem::getClothPositions() {
	return clothPositions;
}

vector<glm::vec3>& ParticleSystem::getFluidPositions() {
	return fluidPositions;
}

vector<glm::vec4>& ParticleSystem::getFoamPositions() {
	return foamPositions;
}

glm::vec3 ParticleSystem::getWeightedPosition(Particle &p) {
	/*for (auto &n : p.neighbors) {
		float weight = 1 - (glm::distance(p.newPos, n->newPos) / H);

		p.sumWeight += weight;

		p.weightedPos += n->newPos * weight;
	}

	if (p.sumWeight != 0.0f) {
		p.weightedPos /= p.sumWeight;
	} else {
		p.weightedPos = p.newPos;
	}

	return p.weightedPos;*/
	return glm::vec3(0);
}

void ParticleSystem::setNeighbors() {
	//#pragma omp parallel for num_threads(8)
	for (int i = 0; i < particles.size(); i++) {
		Particle &p = particles.at(i);
		p.neighbors.clear();
		glm::ivec3 pos = p.newPos * 10;
		for (auto &c : grid.cells[pos.x][pos.y][pos.z].neighbors) {
			for (auto &n : c->particles) {
				//if (glm::distance(p.newPos, n->newPos) <= 2 * H) {
				//p.renderNeighbors.push_back(n);
				if (glm::distance(p.newPos, n->newPos) <= H) {
					p.neighbors.push_back(n);
				}
				//}
			}
		}
	}
}

void ParticleSystem::calcDensities() {
#pragma omp parallel for num_threads(8)
	for (int i = 0; i < particles.size(); i++) {
		Particle &p = particles.at(i);
		float rhoSum = 0;
		for (auto &n : p.neighbors) {
			rhoSum += WPoly6(p.newPos, n->newPos);
		}

		buffer2[i] = rhoSum;
	}
}

void ParticleSystem::updateFoam() {
	//Kill dead foam
	for (int i = 0; i < foam.size(); i++) {
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

		glm::ivec3 pos = p.pos * 10;
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
	}
}

void ParticleSystem::generateFoam() {
	for (int i = 0; i < particles.size(); i++) {
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

			glm::ivec3 pos = p.newPos * 10;
			int type;
			int numNeighbors = int(p.neighbors.size()) + 1;

			if (numNeighbors < 8) type = 1;
			else type = 2;

			foam.push_back(FoamParticle(xd, vd, lifetime, type));

			confineToBox(foam.back());
		}
	}
}


float ParticleSystem::easeInOutQuad(float t, float b, float c, float d) {
	t /= d / 2;
	if (t < 1) return c / 2 * t*t + b;
	t--;
	return -c / 2 * (t*(t - 2) - 1) + b;
};

void ParticleSystem::clothUpdate() {
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
		} else {
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
}