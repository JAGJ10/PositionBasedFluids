#ifndef RENDERER_H
#define RENDERER_H

#include "common.h"
#include "Shader.h"
#include "Camera.hpp"
#include "FluidBuffer.h"
#include "FullscreenQuad.h"

struct buffers {
	GLuint fbo, vao, vbo, ebo, tex;
};

class Renderer {
public:
	Renderer(int width, int height);
	~Renderer();
	void initVBOS(int numParticles, int numDiffuse, int numCloth, std::vector<int> triangles);
	void run(int numParticles, int numDiffuse, int numCloth, std::vector<int> triangles, Camera &cam);

	cudaGraphicsResource *resources[3];
	GLuint positionVBO;
	GLuint indicesVBO;
	GLuint diffusePosVBO;
	GLuint diffuseVelVBO;

private:
	void renderPlane(buffers &buf);
	void renderWater(glm::mat4 &projection, glm::mat4 &mView, Camera &cam, int numParticles, int numCloth);
	void renderFoam(glm::mat4 &projection, glm::mat4 &mView, Camera &cam, int numDiffuse);
	void renderCloth(glm::mat4 &projection, glm::mat4 &mView, Camera &cam, int numCloth, std::vector<int> triangles);
	void initFramebuffers();

	int width, height;
	glm::mat4 mView, normalMatrix, projection;
	float aspectRatio;
	glm::vec2 screenSize, blurDirX, blurDirY;

	buffers planeBuf;
	FluidBuffer fluidBuffer;
	GLuint positionVAO;
	GLuint diffusePosVAO;
	GLuint indicesVAO;

	Shader plane;
	Shader cloth;
	Shader depth;
	Shader blur;
	Shader thickness;
	Shader fluidFinal;
	Shader foamDepth;
	Shader foamThickness;
	Shader foamIntensity;
	Shader foamRadiance;
	Shader finalFS;
	FullscreenQuad fsQuad;
};

#endif
