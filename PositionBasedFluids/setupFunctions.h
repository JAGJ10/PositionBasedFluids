#ifndef SETUP_FUNCTIONS_H
#define SETUP_FUNCTIONS_H

#include "common.h"
#include "parameters.h"
#include "Mesh.h"
#include "tiny_obj_loader.h"

void createParticleGrid(tempSolver* s, solverParams* sp, float3 lower, int3 dims, float radius) {
	for (int x = 0; x < dims.x; x++) {
		for (int y = 0; y < dims.y; y++) {
			for (int z = 0; z < dims.z; z++) {
				float3 pos = lower + make_float3(float(x), float(y), float(z)) * radius;
				s->positions.push_back(make_float4(pos, 1.0f));
				s->velocities.push_back(make_float3(0));
				s->phases.push_back(0);
			}
		}
	}
}

void createParticleShape(std::string sdfFile, tempSolver* s, float3 lower, bool rigid) {
	std::ifstream infile(sdfFile);
	if (!infile) {
		std::cerr << "Unable to open file: " << sdfFile;
		exit(-1);
	}

	float3 dims, origin = make_float3(0);
	std::string dimsString, originString, cellSizeString;
	std::getline(infile, dimsString);
	std::stringstream data(dimsString);
	data >> dims.x >> dims.y >> dims.z;
	std::getline(infile, originString);
	data = std::stringstream(originString);
	data >> origin.x >> origin.y >> origin.z;
	std::getline(infile, cellSizeString);
	float value;
	int count = 0;
	while (infile >> value) {
		if (value < 0) {
			float z = count / (int(dims.y) * int(dims.x));
			float y = count % (int(dims.y) * int(dims.x)) / (int(dims.x));
			float x = count % int(dims.x);
			float3 pos = make_float3(x, y, z) * 0.1f;
			s->positions.push_back(make_float4(lower + pos + origin, 1.0f));
			s->velocities.push_back(make_float3(0));
			s->phases.push_back(0);
		}
		count++;
	}

	infile.close();
}

int getIndex(int x, int y, int dx) {
	return y * dx + x;
}

void addConstraint(tempSolver* s, solverParams* sp, int p1, int p2, float stiffness) {
	s->clothIndices.push_back(p1);
	s->clothIndices.push_back(p2);
	s->restLengths.push_back(length(make_float3(s->positions[p1] - s->positions[p2])));
	s->stiffness.push_back(stiffness);
}

void createCloth(tempSolver* s, solverParams* sp, float3 lower, int3 dims, float radius, int phase, float stretch, float bend, float shear, float invMass) {
	//Create grid of particles and add triangles
	for (int z = 0; z < dims.z; z++) {
		for (int y = 0; y < dims.y; y++) {
			for (int x = 0; x < dims.x; x++) {
				float3 pos = lower + make_float3(float(x), float(y), float(z)) * radius;
				s->positions.push_back(make_float4(pos, invMass));
				s->velocities.push_back(make_float3(0));
				s->phases.push_back(phase);

				if (x > 0 && z > 0) {
					s->triangles.push_back(getIndex(x - 1, z - 1, dims.x));
					s->triangles.push_back(getIndex(x, z - 1, dims.x));
					s->triangles.push_back(getIndex(x, z, dims.x));

					s->triangles.push_back(getIndex(x - 1, z - 1, dims.x));
					s->triangles.push_back(getIndex(x, z, dims.x));
					s->triangles.push_back(getIndex(x - 1, z, dims.x));
				}
			}
		}
	}

	//Horizontal constraints
	for (int j = 0; j < dims.z; j++) {
		for (int i = 0; i < dims.x; i++) {
			int i0 = getIndex(i, j, dims.x);
			if (i > 0) {
				int i1 = j * dims.x + i - 1;
				addConstraint(s, sp, i0, i1, stretch);
			}

			if (i > 1) {
				int i2 = j * dims.x + i - 2;
				addConstraint(s, sp, i0, i2, bend);
			}

			if (j > 0 && i < dims.x - 1) {
				int iDiag = (j - 1) * dims.x + i + 1;
				addConstraint(s, sp, i0, iDiag, shear);
			}

			if (j > 0 && i > 0) {
				int iDiag = (j - 1) * dims.x + i - 1;
				addConstraint(s, sp, i0, iDiag, shear);
			}
		}
	}

	//Vertical constraints
	for (int i = 0; i < dims.x; i++) {
		for (int j = 0; j < dims.z; j++) {
			int i0 = getIndex(i, j, dims.x);
			if (j > 0) {
				int i1 = (j - 1) * dims.x + i;
				addConstraint(s, sp, i0, i1, stretch);
			}

			if (j > 1) {
				int i2 = (j - 2) * dims.x + i;
				addConstraint(s, sp, i0, i2, bend);
			}
		}
	}
}

std::pair<std::vector<tinyobj::shape_t>, std::vector<tinyobj::material_t>> read(std::istream& stream) {
	assert(sizeof(float) == sizeof(int));
	const auto sz = sizeof(int);

	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;

	int nMeshes = 0;
	int nMatProperties = 0;
	stream.read((char*)&nMeshes, sz);
	stream.read((char*)&nMatProperties, sz);
	shapes.resize(nMeshes);
	materials.resize(nMeshes);

	for (size_t i = 0; i < nMeshes; ++i) {
		int nVertices = 0, nNormals = 0, nTexcoords = 0, nIndices = 0;
		stream.read((char*)&nVertices, sz);
		stream.read((char*)&nNormals, sz);
		stream.read((char*)&nTexcoords, sz);
		stream.read((char*)&nIndices, sz);

		shapes[i].mesh.positions.resize(nVertices);
		shapes[i].mesh.normals.resize(nNormals);
		shapes[i].mesh.texcoords.resize(nTexcoords);
		shapes[i].mesh.indices.resize(nIndices);

		stream.read((char*)&shapes[i].mesh.positions[0], nVertices * sz);
		if (nNormals > 0) stream.read((char*)&shapes[i].mesh.normals[0], nNormals * sz);
		if (nTexcoords > 0) stream.read((char*)&shapes[i].mesh.texcoords[0], nTexcoords * sz);
		stream.read((char*)&shapes[i].mesh.indices[0], nIndices * sz);
		if (materials.size()) {
			stream.read((char*)&materials[i].ambient[0], 3 * sz);
			stream.read((char*)&materials[i].diffuse[0], 3 * sz);
			stream.read((char*)&materials[i].specular[0], 3 * sz);
		}
	}

	std::pair<std::vector<tinyobj::shape_t>, std::vector<tinyobj::material_t>> ret(shapes, materials);

	return ret;
}

void loadMeshes(std::string meshFile, std::vector<Mesh> &meshes) {
	std::ifstream infile(meshFile, std::ifstream::binary);
	if (!infile) {
		std::cerr << "Unable to open file: " << meshFile;
		exit(-1);
	}

	std::pair<std::vector<tinyobj::shape_t>, std::vector<tinyobj::material_t>> sm = read(infile);

	meshes.clear();
	meshes.resize(sm.first.size());

	for (int i = 0; i < sm.first.size(); i++) {
		meshes[i].create();
		meshes[i].updateBuffers(sm.first[i].mesh.positions, sm.first[i].mesh.indices, sm.first[i].mesh.normals);
		meshes[i].ambient = glm::vec3(sm.second[i].ambient[0], sm.second[i].ambient[1], sm.second[i].ambient[2]);
		meshes[i].diffuse = glm::vec3(sm.second[i].diffuse[0], sm.second[i].diffuse[1], sm.second[i].diffuse[2]);
		meshes[i].specular = sm.second[i].specular[0];
	}
}

#endif