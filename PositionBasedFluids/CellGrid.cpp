#include "CellGrid.h"

using namespace std;

CellGrid::CellGrid(int width, int height, int depth) : w{ width * 10 }, h{ height * 10 }, d{ depth * 10 },
	cells(width * 10 + 1, vector<vector<Cell>>(height * 10 + 1, vector<Cell>(depth * 10 + 1, Cell()))) {

	for (auto &&row : cells) {
		for (auto &&col : row) {
			for (auto &&cell : col) {
				cell.neighbors.reserve(27);
				cell.particles.reserve(8);
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
	//#pragma omp parallel for num_threads(8)
	for (int i = 0; i < particles.size(); i++) {
		Particle &p = particles.at(i);
		glm::ivec3 pos = p.newPos;//* 10;
		//assuming indices are always valid because the box keeps the particles contained
		cells[pos.x][pos.y][pos.z].addParticle(p);
	}
}

void CellGrid::clearCells() {
	#pragma omp parallel for num_threads(8)
	for (int i = 0; i < w; i++) {
		for (int j = 0; j < h; j++) {
			for (int k = 0; k < d; k++) {
				cells[i][j][k].particles.clear();
			}
		}
	}
}