#ifndef PARTICLE_H
#define PARTICLE_H

class Particle {
public:
	glm::vec3 oldPos;
	glm::vec3 newPos;
	glm::vec3 velocity;
	std::vector<Particle*> neighbors;
	float invMass;
	int index;

	Particle(glm::vec3 pos, float invMass, int index) {
		this->oldPos = pos;
		this->newPos = pos;
		this->invMass = invMass;
		this->index = index;
		velocity = glm::vec3(0, 0, 0);
	}

public:
	//glm::vec3 weightedPos;
	//float sumWeight;
};

#endif
