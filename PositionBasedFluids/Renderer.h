#ifndef RENDERER_H
#define RENDERER_H

#include "common.h"
#include "Cell.hpp"
#include "ParticleSystem.h"

class Renderer {
public:
	Renderer();
	~Renderer();
	void run();

	ParticleSystem system;
	Shader depth;
	Shader normals;
	Shader blur;
	Shader thickness;
	Shader composite;

private:
	void initShaders();
	void initFramebuffers();
};

#endif
