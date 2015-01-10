#ifndef RENDERER_H
#define RENDERER_H

#include "common.h"
#include "containers.h"
#include "ParticleSystem.h"

class Renderer {
public:
	Renderer();
	~Renderer();
	void run();

	ParticleSystem system;

private:
	void initShaders();
	void initFramebuffers();
};

#endif