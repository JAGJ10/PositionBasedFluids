#ifndef SETUP_FUNCTIONS_H
#define SETUP_FUNCTIONS_H

#include "common.h"
#include "parameters.h"
#include "Mesh.h"
#include "tiny_obj_loader.h"
#define VOXELIZER_IMPLEMENTATION
#include "voxelizer.h"

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

	//std::pair<std::vector<tinyobj::shape_t>, std::vector<tinyobj::material_t>> sm = read(infile);

	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;
	std::string err;
	bool ret = tinyobj::LoadObj(shapes, materials, err, sdfFile.c_str());

	vx_mesh_t* mesh;
	vx_mesh_t* result;

	mesh = vx_mesh_alloc(shapes[0].mesh.positions.size(), shapes[0].mesh.indices.size());

	for (size_t f = 0; f < shapes[0].mesh.indices.size(); f++) {
		mesh->indices[f] = shapes[0].mesh.indices[f];
	}
	for (size_t v = 0; v < shapes[0].mesh.positions.size() / 3; v++) {
		mesh->vertices[v].x = shapes[0].mesh.positions[3 * v + 0];
		mesh->vertices[v].y = shapes[0].mesh.positions[3 * v + 1];
		mesh->vertices[v].z = shapes[0].mesh.positions[3 * v + 2];
	}

	std::cout << "before voxelize" << std::endl;
	result = vx_voxelize(mesh, 0.05, 0.05, 0.05, 0.05);
	std::cout << "after voxelize" << std::endl;
	for (int i = 0; i < result->nvertices; i++) {
		s->positions.push_back(make_float4(lower + make_float3(result->vertices[i].x, result->vertices[i].y, result->vertices[i].z), 1.0f));
		s->velocities.push_back(make_float3(0));
		s->phases.push_back(2);
	}

	vx_mesh_free(result);
	vx_mesh_free(mesh);

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

void loadMeshes(std::string meshFile, std::vector<Mesh> &meshes, float3 offset, float scale, int s) {
	std::ifstream infile(meshFile, std::ifstream::binary);
	if (!infile) {
		std::cerr << "Unable to open file: " << meshFile;
		exit(-1);
	}

	std::pair<std::vector<tinyobj::shape_t>, std::vector<tinyobj::material_t>> sm = read(infile);

	for (int i = 0; i < sm.first.size(); i++) {
		for (int j = 0; j < sm.first[i].mesh.positions.size(); j+=3) {
			sm.first[i].mesh.positions[j] *= scale;
			sm.first[i].mesh.positions[j + 1] *= scale;
			sm.first[i].mesh.positions[j + 2] *= scale;
			sm.first[i].mesh.positions[j] += offset.x;
			sm.first[i].mesh.positions[j+1] += offset.y;
			sm.first[i].mesh.positions[j+2] += offset.z;
		}
	}

	for (int i = 0; i < sm.first.size(); i++) {
		meshes[s].create();
		meshes[s].updateBuffers(sm.first[i].mesh.positions, sm.first[i].mesh.indices, sm.first[i].mesh.normals);
		meshes[s].ambient = glm::vec3(sm.second[i].ambient[0], sm.second[i].ambient[1], sm.second[i].ambient[2]);
		meshes[s].diffuse = glm::vec3(sm.second[i].diffuse[0], sm.second[i].diffuse[1], sm.second[i].diffuse[2]);
		meshes[s].specular = sm.second[i].specular[0];
	}

	infile.close();
}

#endif