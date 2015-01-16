#ifndef PARTICLE_H
#define PARTICLE_H

class Particle {
public:
	glm::vec3 oldPos;
	glm::vec3 newPos;
	glm::vec3 velocity;
	glm::vec3 force;
	glm::vec3 deltaP;
	float mass;
	float lambda;
	std::vector<Particle*> neighbors;
	std::vector<Particle*> renderNeighbors;

	Particle(glm::vec3 pos, float mass) {
		this->oldPos = pos;
		this->mass = mass;
		newPos = glm::vec3(0, 0, 0);
		velocity = glm::vec3(0, 0, 0);
		force = glm::vec3(0, 0, 0);
		deltaP = glm::vec3(0, 0, 0);
		lambda = 0;
	}

public:
	glm::vec3 weightedPos;
	float sumWeight;
};

#endif
