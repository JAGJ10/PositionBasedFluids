#include "ParticleSystem.h"

using namespace std;

static const float deltaT = 0.016f;
static const float PI = 3.14159265358979323846f;
static const glm::vec3 GRAVITY = glm::vec3(0, -9.8f, 0);
static const int PRESSURE_ITERATIONS = 10;
static const float H = 0.95f;
static const float KPOLY = 300 / (64 * PI * glm::pow(H, 9));
static const float SPIKY = 15 / (PI * glm::pow(H, 6));
static const float VISC = 15 / (2 * PI * (H * H * H));
static const float REST_DENSITY = 1.0f;
static const float EPSILON_LAMBDA = 150.0f;
static const float EPSILON_VORTICITY = 0.5f;
static const float C = 0.01f;
static const float K = 0.01f;
static const float deltaQMag = 0; //.1f * H;
static const float wQH = KPOLY * glm::pow((H * H - deltaQMag * deltaQMag), 3);
static float width = 90;
static float height = 500;
static float depth = 90;

ParticleSystem::ParticleSystem() : grid((int)width, (int)height, (int)depth) {
	for (float i = 5; i < 20; i+=.9f) {
		for (float j = 0; j < .9f; j+=.9f) {
			for (float k = 5; k < 20; k +=.9f) {
				particles.push_back(Particle(glm::vec3(i, j, k), 1));
			}
		}
	}

	positions.reserve(particles.capacity());
}

ParticleSystem::~ParticleSystem() {}

void ParticleSystem::update() {
	for (auto &p : particles) {
		applyGravity(p);
		p.newPos = p.oldPos;

		//update velocity vi = vi + deltaT * fExt
		p.velocity += p.force * deltaT;

		//predict position x* = xi + deltaT * vi
		p.newPos += p.velocity * deltaT;

		imposeConstraints(p);
	}

	//get neighbors
	grid.updateCells(particles);
	for (auto &p : particles) {
		p.neighbors.clear();
		glm::ivec3 pos = p.newPos;
		vector<Cell*> neighborCells = grid.cells[pos.x][pos.y][pos.z].neighbors;
		for (auto &c : neighborCells) {
			vector<Particle*> allParticles = c->particles;
			for (auto &n : allParticles) {
				if (p.newPos != n->newPos) {
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

	updatePositions();

	for (int i = 0; i < PRESSURE_ITERATIONS; i++) {
		//set lambda
		for (auto &p : particles) {
			p.lambda = lambda(p, p.neighbors);
		}

		//calculate deltaP
		for (auto &p : particles) {
			glm::vec3 deltaP = glm::vec3(0.0f);
			for (auto &n : p.neighbors) {
				float lambdaSum = p.lambda + n->lambda;
				float sCorr = sCorrCalc(p, n);
				deltaP += WSpiky(p.newPos, n->newPos) * (lambdaSum + sCorr);
			}

			p.deltaP = deltaP / REST_DENSITY;
		}

		//update position x*i = x*i + deltaPi
		for (auto &p : particles) {
			p.newPos += p.deltaP;
		}
	}

	for (auto &p : particles) {
		//set new velocity vi = (x*i - xi) / deltaT
		p.velocity = (p.newPos - p.oldPos) / deltaT;

		//apply vorticity confinement
		p.velocity += vorticityForce(p) * deltaT;

		//apply XSPH viscosity
		p.velocity += xsphViscosity(p);

		//update position xi = x*i
		p.oldPos = p.newPos;
	}
}

void ParticleSystem::applyGravity(Particle &p) {
	p.force = glm::vec3(0.0f);
	p.force += GRAVITY;
}

//Poly6 Kernel
float ParticleSystem::WPoly6(glm::vec3 &pi, glm::vec3 &pj) {
	glm::vec3 r = pi - pj;
	float rLen = glm::length(r);
	if (rLen > H || rLen == 0) {
		return 0;
	}

	return KPOLY * glm::pow((H * H - glm::length2(rLen)), 3);
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

//Viscosity Kernel
glm::vec3 ParticleSystem::WViscosity(glm::vec3 &pi, glm::vec3 &pj) {
	glm::vec3 r = pi - pj;
	float rLen = glm::length(r);
	if (rLen > H || rLen == 0) {
		return glm::vec3(0.0f);
	}

	float coeff = (-1 * (rLen * rLen * rLen)) / (2 * (H * H * H));
	coeff *= VISC;
	coeff += ((rLen * rLen) / (H * H));
	coeff += (H / (2 * rLen)) - 1;
	return r * coeff;
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
		sum += n->mass * WPoly6(p.newPos, n->newPos);
	}

	return (sum / REST_DENSITY) - 1;
}

//Returns the eta vector that points in the direction of the corrective force
glm::vec3 ParticleSystem::eta(Particle &p, float &vorticityMag) {
	glm::vec3 eta = glm::vec3(0.0f);
	for (auto &n : p.neighbors) {
		eta += WViscosity(p.newPos, n->newPos) * vorticityMag;
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
		gradient = WViscosity(p.newPos, n->newPos);
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
	// Get Density from WPoly6 and divide by constant from paper
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

float ParticleSystem::clampedConstraint(float x, float max) {
	if (x <= 0.0f) {
		return 0.0f;
	} else if (x >= max) {
		return max - 0.1f;
	} else {
		return x;
	}
}

bool ParticleSystem::outOfRange(float x, float min, float max) {
	return x <= min || x >= max;
}

void ParticleSystem::updatePositions() {
	positions.clear();
	for (int i = 0; i < particles.size(); i++) {
		particles[i].sumWeight = 0.0f;
		particles[i].weightedPos = glm::vec3(0.0f);
	}

	for (int i = 0; i < particles.size(); i++) {
		//positions.push_back(particles[i].oldPos);
		positions.push_back(getWeightedPosition(particles[i]));
	}
}

vector<glm::vec3>& ParticleSystem::getPositions() {
	return positions;
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