#ifndef MESH_H
#define MESH_H

#include "common.h"
#include "VBO.h"

class Mesh {
public:
	int numIndices;
	glm::vec3 ambient;
	glm::vec3 diffuse;
	float specular;

	Mesh();
	~Mesh();

	void create();
	void updateBuffers(std::vector<float>& positions, std::vector<GLuint>& indices, std::vector<float>& normals);
	void updateBuffers(std::vector<glm::vec3>& positions, std::vector<GLuint>& indices);
	void clear();
	void render();
	void setAttributes();
	bool hasBuffer() const;

private:
	GLuint vao;
	VBO positionBuffer;
	VBO indexBuffer;
	VBO normalBuffer;
};

#endif