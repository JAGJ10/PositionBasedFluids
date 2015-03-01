#include "ParticleSystem.h"

using namespace std;

static const float deltaT = 0.0083f;
static const float PI = 3.14159265358979323846f;
static const glm::vec3 GRAVITY = glm::vec3(0, -9.8f, 0);
static const int PRESSURE_ITERATIONS = 4;
static const float H = 0.1f;
static const float FH = H;
static const float KPOLY = 315 / (64 * PI * glm::pow(H, 9));
static const float SPIKY = 45 / (PI * glm::pow(H, 6));
static const float REST_DENSITY = 6378.0f;
static const float EPSILON_LAMBDA = 600.0f;
static const float EPSILON_VORTICITY = 0.0001f;
static const float C = 0.01f;
static const float K = 0.000001f;
static const float deltaQMag = 0.0f; //0.3f * H;
static const float wQH = KPOLY * glm::pow((H * H - deltaQMag * deltaQMag), 3);
static const float lifetime = 1.0f;

static const int wcmin = 2;
static const int wcmax = 8;
static const int tamin = 5;
static const int tamax = 20;
static const int kmin = 5;
static const int kmax = 50;

static const int kta = 1200;
static const int kwc = 3600;

static float width = 7;
static float height = 8;
static float depth = 4;

int frameCounter = 0;

ParticleSystem::ParticleSystem() : grid((int)width, (int)height, (int)depth) {
	for (float i = 0; i < 2; i+=.05f) {
		for (float j = 0; j < 2; j+=.05f) {
			for (float k = 1; k < 3; k+=.05f) {
				particles.push_back(Particle(glm::vec3(i, j, k)));
			}
		}
	}

	foam.reserve(100000);
	fluidPositions.reserve(particles.capacity());

	srand((unsigned int)time(0));
}

ParticleSystem::~ParticleSystem() {}

void ParticleSystem::update() {
	//Move wall
	frameCounter++;
	if (frameCounter >= 400)
		width = (1 - abs(sin((frameCounter - 400) * (deltaT / 2)  * 0.5f * PI)) * 3) + 6;

	//----------------FOAM-----------------
	//Kill dead foam
	for (int i = 0; i < foam.size(); i++) {
		FoamParticle &p = foam.at(i);
		if (p.type == 3) {
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
		imposeConstraints(p);

		glm::ivec3 pos = p.pos * 10;
		glm::vec3 vfSum = glm::vec3(0.0f);
		float kSum = 0;
		int numNeighbors = 0;
		for (auto &c : grid.cells[pos.x][pos.y][pos.z].neighbors) {
			for (auto &n : c->particles) {
				if (glm::distance(p.pos, n->newPos) <= FH) {
					numNeighbors++;
					float k = WPoly6(p.pos, n->newPos);
					glm::vec3 vf = ((n->newPos + n->velocity * deltaT) - n->newPos) / deltaT;
					vfSum += vf * k;
					kSum += k;
				}
			}
		}

		if (numNeighbors >= 6 && numNeighbors <= 20) p.type = 3;
		if (numNeighbors == 0 && p.type != 0) p.type = 1;

		if (p.type == 1) {
			//Spray
			p.velocity += GRAVITY * deltaT;
			p.pos += p.velocity * deltaT;
		} else if (p.type == 2) {
			//Bubbles
			p.velocity += ((-1.0 * GRAVITY) + ((0.5 * ((vfSum / kSum) - p.velocity)) / deltaT)) * deltaT;
			p.pos += p.velocity * deltaT;
		} else if (p.type == 3) {
			//Foam
			p.pos += (vfSum / kSum) * deltaT;
		}
	}

	calcDensities();
	calcNormals();
	for (int i = 0; i < particles.size(); i++) {
		Particle &p = particles.at(i);
		float velocityDiff = 0.0f;
		float curvature = 0.0f;
		for (auto &n : p.neighbors) {
			float wAir = WAirPotential(p.newPos, n->newPos);
			glm::vec3 xij = glm::normalize(p.newPos - n->newPos);
			glm::vec3 xji = glm::normalize(n->newPos - p.newPos);
			glm::vec3 vijHat = glm::normalize(p.velocity - n->velocity);
			velocityDiff += glm::length(p.velocity - n->velocity) * (1 - glm::dot(vijHat, xij)) * wAir;
			if (glm::dot(xji, p.normal) < 0) {
				curvature += (1 - glm::dot(p.normal, n->normal)) * wAir;
			}
		}

		float ita = foamPotential(velocityDiff, tamin, tamax);

		int deltaVN = glm::dot(glm::normalize(p.velocity), p.normal) < 0.6f ? 0 : 1;
		float iwc = foamPotential(curvature * deltaVN, wcmin, wcmax);

		float ek = 0.5f * glm::length2(p.velocity);
		float ik = foamPotential(ek, kmin, kmax);

		int nd = int(ik * (kta * ita + kwc * iwc) * deltaT);

		glm::vec3 e1 = glm::cross(glm::vec3(-p.velocity.y, p.velocity.x, 0), p.velocity);
		e1 = glm::normalize(e1);
		glm::vec3 e2 = glm::cross(e1, p.velocity);
		e2 = glm::normalize(e2);

		for (int i = 0; i < nd; i++) {
			float xr = 0.05f + static_cast <float> (rand()) / static_cast <float> (RAND_MAX / 0.9f);
			float xtheta = 0.05f + static_cast <float> (rand()) / static_cast <float> (RAND_MAX / 0.9f);
			float xh = 0.05f + static_cast <float> (rand()) / static_cast <float> (RAND_MAX / 0.9f);

			float r = H * glm::sqrt(xr);
			float theta = xtheta * 2 * PI;
			float distH = xh * glm::length(deltaT * p.velocity);

			glm::vec3 xd = p.newPos + (r * glm::cos(theta) * e1) + (r * glm::sin(theta) * e2) + (distH * glm::normalize(p.velocity));
			glm::vec3 vd = (r * glm::cos(theta) * e1) + (r * glm::sin(theta) * e2) + p.velocity;

			glm::ivec3 pos = p.newPos * 10;
			int type;
			int numNeighbors = 0;
			for (auto &c : grid.cells[pos.x][pos.y][pos.z].neighbors) {
				for (auto &n : c->particles) {
					if (glm::distance(xd, n->newPos) <= FH) {
						numNeighbors++;
					}
				}
			}

			if (numNeighbors < 6) type = 1;
			else if (numNeighbors > 20) type = 2;
			else type = 3;

			foam.push_back(FoamParticle(xd, vd, lifetime, type));

			imposeConstraints(foam.back());
		}
	}

	//------------------WATER-----------------
	#pragma omp parallel for num_threads(8)
	for (int i = 0; i < particles.size(); i++) {
		Particle &p = particles.at(i);

		//update velocity vi = vi + deltaT * fExt
		p.velocity += GRAVITY * deltaT;

		//predict position x* = xi + deltaT * vi
		p.newPos += p.velocity * deltaT;

		imposeConstraints(p);
	}

	//get neighbors
	grid.updateCells(particles);
	#pragma omp parallel for num_threads(8)
	for (int i = 0; i < particles.size(); i++) {
		Particle &p = particles.at(i);
		p.neighbors.clear();
		glm::ivec3 pos = p.newPos * 10;
		for (auto &c : grid.cells[pos.x][pos.y][pos.z].neighbors) {
			for (auto &n : c->particles) {
				//if (p.newPos != n->newPos) {
					//if (glm::distance(p.newPos, n->newPos) <= 2 * H) {
					//p.renderNeighbors.push_back(n);
					if (glm::distance(p.newPos, n->newPos) <= H) {
						p.neighbors.push_back(n);
					}
					//}
				//}
			}
		}
	}

	//Needs to be after neighbor finding for weighted positions
	updatePositions();

	for (int i = 0; i < PRESSURE_ITERATIONS; i++) {
		//set lambda
		#pragma omp parallel for num_threads(8)
		for (int i = 0; i < particles.size(); i++) {
			Particle &p = particles.at(i);
			p.lambda = lambda(p, p.neighbors);
		}

		//calculate deltaP
		#pragma omp parallel for num_threads(8)
		for (int i = 0; i < particles.size(); i++) {
			Particle &p = particles.at(i);
			glm::vec3 deltaP = glm::vec3(0.0f);
			for (auto &n : p.neighbors) {
				float lambdaSum = p.lambda + n->lambda;
				float sCorr = sCorrCalc(p, n);
				deltaP += WSpiky(p.newPos, n->newPos) * (lambdaSum + sCorr);
			}

			p.deltaP = deltaP / REST_DENSITY;
		}

		//update position x*i = x*i + deltaPi
		#pragma omp parallel for num_threads(8)
		for (int i = 0; i < particles.size(); i++) {
			Particle &p = particles.at(i);
			p.newPos += p.deltaP;
		}
	}

	#pragma omp parallel for num_threads(8)
	for (int i = 0; i < particles.size(); i++) {
		Particle &p = particles.at(i);
		imposeConstraints(p);

		//set new velocity vi = (x*i - xi) / deltaT
		p.velocity = (p.newPos - p.oldPos) / deltaT;

		//apply vorticity confinement
		p.velocity += vorticityForce(p) * deltaT;

		//apply XSPH viscosity
		p.newVelocity = xsphViscosity(p);

		//update position xi = x*i
		p.oldPos = p.newPos;
	}

	#pragma omp parallel for num_threads(8)
	for (int i = 0; i < particles.size(); i++) {
		Particle &p = particles.at(i);
		p.velocity += p.newVelocity * deltaT;
	}
}

//Poly6 Kernel
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

//Spiky Kernel
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
	if (rLen > H) {
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

void ParticleSystem::imposeConstraints(Particle &p) {
	if (outOfRange(p.newPos.x, 0, width)) {
		p.velocity.x = 0;
	}

	if (outOfRange(p.newPos.y, 0, height)) {
		p.velocity.y = 0;
	}

	if (outOfRange(p.newPos.z, 0, depth)) {
		p.velocity.z = 0;
	}

	p.newPos.x = clampedConstraint(p.newPos.x, width);
	p.newPos.y = clampedConstraint(p.newPos.y, height);
	p.newPos.z = clampedConstraint(p.newPos.z, depth);
}

void ParticleSystem::imposeConstraints(FoamParticle &p) {
	if (outOfRange(p.pos.x, 0, width)) {
		p.velocity.x = 0;
	}

	if (outOfRange(p.pos.y, 0, height)) {
		p.velocity.y = 0;
	}

	if (outOfRange(p.pos.z, 0, depth)) {
		p.velocity.z = 0;
	}

	p.pos.x = clampedConstraint(p.pos.x, width);
	p.pos.y = clampedConstraint(p.pos.y, height);
	p.pos.z = clampedConstraint(p.pos.z, depth);
}

float ParticleSystem::clampedConstraint(float x, float max) {
	if (x < 0.0f) {
		return 0.01f;
	} else if (x > max) {
		return max - 0.01f;
	} else {
		return x;
	}
}

bool ParticleSystem::outOfRange(float x, float min, float max) {
	return x <= min || x >= max;
}

void ParticleSystem::updatePositions() {
	fluidPositions.clear();
	foamPositions.clear();

	#pragma omp parallel for num_threads(8)
	for (int i = 0; i < particles.size(); i++) {
		Particle &p = particles.at(i);
		p.sumWeight = 0.0f;
		p.weightedPos = glm::vec3(0.0f);
	}

	//for (auto &p : particles) fluidPositions.push_back(getWeightedPosition(p));
	for (auto &p : particles) fluidPositions.push_back(p.oldPos);
	
	for (int i = 0; i < foam.size(); i++) {
		FoamParticle &p = foam.at(i);
		int r = rand() % foam.size();
		foamPositions.push_back(glm::vec4(p.pos.x, p.pos.y, p.pos.z, (p.type * 1000) + float(i) + abs(p.lifetime - lifetime) / lifetime));
	}
}

vector<glm::vec3>& ParticleSystem::getFluidPositions() {
	return fluidPositions;
}

vector<glm::vec4>& ParticleSystem::getFoamPositions() {
	return foamPositions;
}

glm::vec3 ParticleSystem::getWeightedPosition(Particle &p) {
	for (auto &n : p.neighbors) {
		float weight = 1 - (glm::distance(p.newPos, n->newPos) / H);

		p.sumWeight += weight;

		p.weightedPos += n->newPos * weight;
	}

	if (p.sumWeight != 0.0f) {
		p.weightedPos /= p.sumWeight;
	} else {
		p.weightedPos = p.newPos;
	}

	return p.weightedPos;
}

void ParticleSystem::calcDensities() {
	#pragma omp parallel for num_threads(8)
	for (int i = 0; i < particles.size(); i++) {
		Particle &p = particles.at(i);
		float rhoSum = 0;
		for (auto &n : p.neighbors) {
			rhoSum += WPoly6(p.newPos, n->newPos);
		}

		p.rho = rhoSum;
	}
}

void ParticleSystem::calcNormals() {
	#pragma omp parallel for num_threads(8)
	for (int i = 0; i < particles.size(); i++) {
		Particle &p = particles.at(i);
		glm::vec3 sum = glm::vec3(0.0f);
		for (auto &n : p.neighbors) {
			sum += gradWPoly6(p.newPos, n->newPos) / n->rho;
		}

		p.normal = glm::normalize(sum);
	}
}

float ParticleSystem::foamPotential(float i, int rmin, int rmax) {
	return (min(i, rmax) - min(i, rmin)) / (rmax - rmin);
}