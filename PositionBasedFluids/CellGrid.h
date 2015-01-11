#ifndef CELL_GRID_H
#define CELL_GRID_H

#include "common.h"
#include "Cell.hpp"

class CellGrid {
public:
	CellGrid();
	CellGrid(int width, int height, int depth);
	~CellGrid();

	CellGrid(const CellGrid&) = delete;
	CellGrid& operator=(const CellGrid&) = delete;

	void updateCells(std::vector<Particle> &particles);
	void clearCells();

	int w;
	int h;
	int d;
	std::vector<std::vector<std::vector<Cell>>> cells;
};

#endif
