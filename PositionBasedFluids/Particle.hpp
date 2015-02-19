#ifndef PARTICLE_H
#define PARTICLE_H

class Particle {
public:
	glm::vec3 oldPos;
	glm::vec3 newPos;
	glm::vec3 velocity;
	glm::vec3 deltaP;
	glm::vec3 normal;
	float lambda;
	std::vector<Particle*> neighbors;
	std::vector<Particle*> renderNeighbors;

	Particle(glm::vec3 pos) {
		this->oldPos = pos;
		this->newPos = pos;
		velocity = glm::vec3(0, 0, 0);
		deltaP = glm::vec3(0, 0, 0);
		normal = glm::vec3(0, 0, 0);
		lambda = 0;
	}

public:
	glm::vec3 weightedPos;
	float sumWeight;
};

#endif
