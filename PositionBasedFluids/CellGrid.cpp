#include "CellGrid.h"

using namespace std;

CellGrid::CellGrid() {}

CellGrid::CellGrid(int width, int height, int depth) {
	this->w = width;
	this->h = height;
	this->d = depth;
	cells.resize(w);
	for (int i = 0; i < w; i++) {
		cells[i].resize(h);
		for (int j = 0; j < h; j++) {
			cells[i][j].resize(d);
		}
	}

	for (int i = 0; i < w; i++) {
		for (int j = 0; j < h; j++) {
			for (int k = 0; k < d; k++) {
				cells[i][j][k] = Cell();
				cells[i][j][k].neighbors.reserve(27);
				cells[i][j][k].particles.reserve(8);
			}
		}
	}

	for (int i = 0; i < w; i++) {
		for (int j = 0; j < h; j++) {
			for (int k = 0; k < d; k++) {
				for (int x = -1; x < 2; x++) {
					for (int y = -1; y < 2; y++) {
						for (int z = -1; z < 2; z++) {
							if (i + x >= 0 && i + x < w && j + y >= 0 && j + y < h && k + z >= 0 && k + z < d) {
								cells[i][j][k].addNeighbor(cells[i + x][j + y][k + z]);
							}
						}
					}
				}
			}
		}
	}
}

CellGrid::~CellGrid() {}

void CellGrid::updateCells(vector<Particle> &particles) {
	clearCells();
	for (auto &p : particles) {
		glm::ivec3 pos = p.newPos;
		//assuming indices are always valid because the box keeps the particles contained
		cells[pos.x][pos.y][pos.z].addParticle(p);
	}
}

void CellGrid::clearCells() {
	for (int i = 0; i < w; i++) {
		for (int j = 0; j < h; j++) {
			for (int k = 0; k < d; k++) {
				cells[i][j][k].particles.clear();
			}
		}
	}
}