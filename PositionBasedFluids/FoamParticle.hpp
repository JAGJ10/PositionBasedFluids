#ifndef FOAM_PARTICLE_H
#define FOAM_PARTICLE_H

class FoamParticle {
public:
	glm::vec3 pos;
	glm::vec3 velocity;
	float lifetime;
	int type;

	FoamParticle(glm::vec3 pos, glm::vec3 velocity, float lifetime, int type) : pos(pos), 
		velocity(velocity), lifetime(lifetime), type(type) {}
};

#endif