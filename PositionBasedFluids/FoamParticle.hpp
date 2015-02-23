#ifndef FOAM_PARTICLE_H
#define FOAM_PARTICLE_H

class FoamParticle {
public:
	glm::vec3 pos;
	glm::vec3 velocity;
	float lifetime;
	int type;
	std::vector<Particle*> fluidNeighbors;

	FoamParticle(glm::vec3 pos, glm::vec3 velocity, float lifetime, int type) {
		this->pos = pos;
		this->velocity = velocity;
		this->lifetime = lifetime;
		this->type = type;
	}
};

#endif