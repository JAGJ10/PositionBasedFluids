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

static const int wcmin = 2;
static const int wcmax = 8;
static const int tamin = 5;
static const int tamax = 20;
static const int kmin = 5;
static const int kmax = 50;

static const int kta = 1000;
static const int kwc = 1000;

static float width = 150;
static float height = 500;
static float depth = 150;

ParticleSystem::ParticleSystem() : grid((int)width, (int)height, (int)depth) {
	for (float i = 0; i < 30; i+=.9f) {
		for (float j = 0; j < 30; j+=.9f) {
			for (float k = 0; k < 30; k +=.9f) {
				particles.push_back(Particle(glm::vec3(i, j, k)));
			}
		}
	}

	spray.reserve(100000);
	bubbles.reserve(100000);
	foam.reserve(100000);
	positions.reserve(particles.capacity());

	srand(time(0));
}

ParticleSystem::~ParticleSystem() {}

void ParticleSystem::update() {
	for (auto &p : particles) {
		//update velocity vi = vi + deltaT * fExt
		p.velocity += GRAVITY * deltaT;

		//predict position x* = xi + deltaT * vi
		p.newPos += p.velocity * deltaT;

		imposeConstraints(p);
	}

	//get neighbors
	grid.updateCells(particles);
	for (auto &p : particles) {
		p.neighbors.clear();
		glm::ivec3 pos = p.newPos;
		for (auto &c : grid.cells[pos.x][pos.y][pos.z].neighbors) {
			for (auto &n : c->particles) {
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

	//----------------FOAM-----------------
	
	//Update velocities
	for (auto &p : spray) {
		p.velocity += GRAVITY * deltaT;
		p.pos += p.velocity * deltaT;
	}

	for (auto &p : bubbles) {
		p.velocity += 
	}

	for (auto &p : foam) {
		glm::vec3 vfk = glm::vec3(0.0f);
		glm::vec3 k = glm::vec3(0.0f);
		for (auto &pf : p.fluidNeighbors) {
			glm::vec3 cs = cubicSpline(p.pos, pf->newPos);
			vfk += pf->velocity * cs; //todo
			k += cs;
		}

		p.velocity = vfk / k;
		p.pos += p.velocity * deltaT;
	}


	calcNormals();
	for (auto &p : particles) {
		float velocityDiff = 0.0f;
		float curvature = 0.0f;
		for (auto &n : p.neighbors) {
			float wAir = WAirPotential(p.newPos, n->newPos);
			glm::vec3 xij = glm::normalize(p.newPos - n->newPos);
			glm::vec3 xji = glm::normalize(n->newPos - p.newPos);
			glm::vec3 normVel = glm::normalize(p.velocity - n->velocity);
			velocityDiff += glm::length(n->velocity - p.velocity) * (1 - glm::dot(normVel, xij)) * wAir;
			if (glm::dot(xji, p.normal) < 0) {
				curvature += (1 - glm::dot(p.normal, n->normal) * wAir);
			}
		}

		float ita = foamPotential(velocityDiff, tamin, tamax);

		int deltaVN = (glm::dot(glm::normalize(p.velocity), p.normal) < 0.6f) ? 0 : 1;
		float iwc = foamPotential(curvature * deltaVN, wcmin, wcmax);
		
		float ek = 0.5f * glm::length2(p.velocity);
		float ik = foamPotential(ek, kmin, kmax);

		int nd = int(ik * (kta * ita + kwc * iwc) * deltaT);

		float xr = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
		float xtheta = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
		float xh = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);

		float r = H * glm::sqrt(xr);
		float theta = xtheta * 2 * PI;
		float distH = xh * glm::length(deltaT * p.velocity);

		glm::vec3 xd = p.newPos + (r * glm::cos(theta * 2)) + (r * glm::sin(theta * 2)) + (distH * glm::normalize(p.velocity));
		glm::vec3 vd = (r * glm::cos(theta * 2)) + (r * glm::sin(theta * 2)) + p.velocity;
		
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

	float coeff = (H - rLen) * (H - rLen);
	coeff *= 6 * KPOLY;
	coeff /= rLen;
	return r * -coeff * glm::pow(H * H - glm::length2(r), 2);
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

glm::vec3 ParticleSystem::cubicSpline(glm::vec3 &pi, glm::vec3 &pj) {
	glm::vec3 r = pi - pj;
	float rLen = glm::length(r);
	if (rLen > H || rLen == 0) {
		return glm::vec3(0.0f);
	}

	float coeff = 1 / 6;
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

void ParticleSystem::calcNormals() {
	for (auto &p : particles) {
		glm::vec3 sum = glm::vec3(0.0f);
		for (auto &n : p.neighbors) {
			sum += gradWPoly6(p.newPos, n->newPos);
		}

		p.normal = glm::normalize(sum);
	}
}

float ParticleSystem::foamPotential(float i, int rmin, int rmax) {
	return (min(i, rmax) - min(i, rmin)) / (rmax - rmin);
}