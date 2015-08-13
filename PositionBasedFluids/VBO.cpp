#include "VBO.h"

VBO::VBO() : id(0), bufferType(0) {}

VBO::~VBO() {
	if (id != 0) glDeleteBuffers(1, &id);
}

void VBO::create() {
	glGenBuffers(1, &id);
}

void VBO::destroy() {
	if (bufferType != 0) {
		switch (bufferType) {
		case GL_ARRAY_BUFFER:
			glBindBuffer(bufferType, 0);
			break;
		case GL_ELEMENT_ARRAY_BUFFER:
			glBindBuffer(bufferType, 0);
			break;
		}
	}

	if (id != 0) glDeleteBuffers(1, &id);

	id = 0;
}

void VBO::bind(GLuint type) {
	bufferType = type;
	glBindBuffer(bufferType, id);
}

GLuint VBO::getID() const {
	return id;
}

GLuint VBO::getBufferType() const {
	return bufferType;
}

bool VBO::created() const {
	return id != 0;
}