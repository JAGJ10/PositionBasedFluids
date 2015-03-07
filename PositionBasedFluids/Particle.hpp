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
	int phase;

	Particle(glm::vec3 pos, float invMass, int index, int phase) : oldPos(pos), newPos(pos), 
		invMass(invMass), index(index), phase(phase) 
	{
		velocity = glm::vec3(0, 0, 0);
	}

public:
	//glm::vec3 weightedPos;
	//float sumWeight;
};

#endif
