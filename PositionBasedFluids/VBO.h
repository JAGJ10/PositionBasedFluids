#ifndef VBO_H
#define VBO_H

#include "common.h"

class VBO {
public:
	VBO();
	~VBO();

	void create();
	void destroy();
	void bind(GLuint type);
	GLuint getID() const;
	GLuint getBufferType() const;
	bool created() const;

private:
	GLuint id;
	GLuint bufferType;
};

#endif