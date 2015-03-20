#ifndef FOAM_PARTICLE_H
#define FOAM_PARTICLE_H

struct FoamParticle {
public:
	glm::vec3 pos;
	glm::vec3 velocity;
	float ttl;
	int type;
};

#endif