#include "FullscreenQuad.h"

using namespace std;

FullscreenQuad::FullscreenQuad() : numIndices(0), vao(0), positionBuffer(VBO()), indexBuffer(VBO()) {
	vector<GLfloat> positions = {
		1.0f, 1.0f,	   //Top Right
		1.0f, -1.0f,   //Bottom Right
		-1.0f, -1.0f,  //Bottom Left
		-1.0f, 1.0f	   //Top Left 
	};
	vector<GLuint> indices = {
		0, 1, 3,	//First Triangle
		1, 2, 3		//Second Triangle
	};

	glGenVertexArrays(1, &vao);
	positionBuffer.create();
	indexBuffer.create();
	updateBuffers(positions, indices);
}

FullscreenQuad::~FullscreenQuad() {
	glDeleteVertexArrays(1, &vao);
}

void FullscreenQuad::clear() {
	numIndices = 0;
}

void FullscreenQuad::updateBuffers(vector<float>& positions, vector<GLuint>& indices) {
	numIndices = int(indices.size());

	positionBuffer.bind(GL_ARRAY_BUFFER);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * positions.size(), &positions[0], GL_STATIC_DRAW);

	indexBuffer.bind(GL_ELEMENT_ARRAY_BUFFER);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint) * indices.size(), &indices[0], GL_STATIC_DRAW);
}

void FullscreenQuad::render() {
	positionBuffer.bind(GL_ARRAY_BUFFER);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(0);

	indexBuffer.bind(GL_ELEMENT_ARRAY_BUFFER);
	glDrawElements(GL_TRIANGLES, numIndices, GL_UNSIGNED_INT, 0);
}