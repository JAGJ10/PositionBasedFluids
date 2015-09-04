#ifndef RENDERER_H
#define RENDERER_H

#include "common.h"
#include "Shader.h"
#include "Camera.hpp"
#include "FluidBuffer.h"
#include "FullscreenQuad.h"
#include "ShadowMap.h"
#include "Mesh.h"
#include "GBuffer.h"

struct buffers {
	GLuint fbo, vao, vbo, ebo, tex;
};

class Renderer {
public:
	cudaGraphicsResource *resources[3];
	GLuint positionVBO;
	GLuint indicesVBO;
	GLuint diffusePosVBO;
	GLuint diffuseVelVBO;

	Renderer(int width, int height);
	~Renderer();

	void setMeshes(const std::vector<Mesh> &meshes);
	void initVBOS(int numParticles, int numDiffuse, int numCloth, std::vector<int> triangles);
	void run(int numParticles, int numDiffuse, int numCloth, std::vector<int> triangles, Camera &cam);

private:
	int width, height;

	std::vector<Mesh> meshes;

	glm::mat4 mView, projection, dLightMView, dLightProjection;
	glm::mat3 normalMatrix;

	float aspectRatio;
	glm::vec2 screenSize, blurDirX, blurDirY;

	buffers planeBuf;
	GBuffer gBuffer;
	FluidBuffer fluidBuffer;
	ShadowMap dLightShadow;
	GLuint positionVAO;
	GLuint diffusePosVAO;
	GLuint indicesVAO;

	Shader plane;
	Shader geometry;
	Shader finalPass;
	Shader shadow;
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

	void renderPlane(buffers &buf);
	void geometryPass();
	void compositePass();
	void shadowPass(Camera &cam, int numParticles);
	void renderWater(glm::mat4 &projection, glm::mat4 &mView, Camera &cam, int numParticles, int numCloth);
	void renderFoam(glm::mat4 &projection, glm::mat4 &mView, Camera &cam, int numDiffuse);
	void renderCloth(glm::mat4 &projection, glm::mat4 &mView, Camera &cam, int numCloth, std::vector<int> triangles);
	void initFramebuffers();
};

#endif
