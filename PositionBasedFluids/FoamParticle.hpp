#ifndef FOAM_PARTICLE_H
#define FOAM_PARTICLE_H

class FoamParticle {
public:
	glm::vec3 pos;
	glm::vec3 velocity;
	float mass;
	float lifetime;
	std::vector<Particle*> fluidNeighbors;

	FoamParticle(glm::vec3 pos, glm::vec3 velocity, float mass, float lifetime) {
		this->pos = pos;
		this->velocity = velocity;
		this->mass = mass;
		this->lifetime = lifetime;
	}
};

#endif