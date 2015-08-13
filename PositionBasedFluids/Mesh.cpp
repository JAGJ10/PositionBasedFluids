#include "Mesh.h"

using namespace std;

Mesh::Mesh() : numIndices(0), vao(0), positionBuffer(VBO()), indexBuffer(VBO()), normalBuffer(VBO()) {}

Mesh::~Mesh() {
	glDeleteVertexArrays(1, &vao);
}

void Mesh::create() {
	glGenVertexArrays(1, &vao);
	positionBuffer.create();
	indexBuffer.create();
	normalBuffer.create();
}

void Mesh::updateBuffers(vector<float>& positions, vector<GLuint>& indices, vector<float>& normals) {
	numIndices = int(indices.size());

	positionBuffer.bind(GL_ARRAY_BUFFER);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * positions.size(), &positions[0], GL_STATIC_DRAW);

	if (normals.size() != 0) {
		normalBuffer.bind(GL_ARRAY_BUFFER);
		glBufferData(GL_ARRAY_BUFFER, sizeof(float) * normals.size(), &normals[0], GL_STATIC_DRAW);
	}

	indexBuffer.bind(GL_ELEMENT_ARRAY_BUFFER);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint) * indices.size(), &indices[0], GL_STATIC_DRAW);
}

void Mesh::updateBuffers(vector<glm::vec3>& positions, vector<GLuint>& indices) {
	numIndices = int(indices.size());
	
	positionBuffer.bind(GL_ARRAY_BUFFER);
	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * positions.size(), &positions[0], GL_STATIC_DRAW);
	
	indexBuffer.bind(GL_ELEMENT_ARRAY_BUFFER);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint) * indices.size(), &indices[0], GL_STATIC_DRAW);
}

void Mesh::clear() {
	numIndices = 0;
}

void Mesh::render() {
	glBindVertexArray(vao);

	positionBuffer.bind(GL_ARRAY_BUFFER);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(0);
	
	normalBuffer.bind(GL_ARRAY_BUFFER);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(1);

	indexBuffer.bind(GL_ELEMENT_ARRAY_BUFFER);
	
	glDrawElements(GL_TRIANGLES, numIndices, GL_UNSIGNED_INT, 0);
}

void Mesh::setAttributes() {
	glBindVertexArray(vao);
	if (hasBuffer()) {
		positionBuffer.bind(GL_ARRAY_BUFFER);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
		glEnableVertexAttribArray(0);

		normalBuffer.bind(GL_ARRAY_BUFFER);
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);
		glEnableVertexAttribArray(1);

		indexBuffer.bind(GL_ELEMENT_ARRAY_BUFFER);
	} else {
		//glBindBuffer(GL_ARRAY_BUFFER, 0);
		//glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
		//glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float), &positions[0]);
		//glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(float), &normals[0]);
	}
}

bool Mesh::hasBuffer() const {
	return positionBuffer.created();
}