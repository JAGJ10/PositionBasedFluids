#ifndef CELL_H
#define CELL_H

#include "Particle.hpp"

class Cell {
public:
	std::vector<Particle*> particles;
	std::vector<Cell*> neighbors;

	void addParticle(Particle &p) {
		particles.push_back(&p);
	}

	void addNeighbor(Cell &c) {
		neighbors.push_back(&c);
	}
};

#endif
