#ifndef RENDERER_H
#define RENDERER_H

#include "common.h"
#include "Shader.hpp"
#include "Camera.hpp"

class Renderer {
public:
	Renderer();
	~Renderer();
	void initVBOS(int numParticles, int numDiffuse, std::vector<int> triangles);
	void run(int numParticles, int numDiffuse, int numCloth, std::vector<int> triangles, Camera &cam);

	cudaGraphicsResource *resources[3];
	GLuint positionVBO;
	GLuint indicesVBO;
	GLuint diffusePosVBO;
	GLuint diffuseVelVBO;

	Shader plane;
	Shader cloth;
	Shader depth;
	BlurShader blur;
	Shader thickness;
	Shader fluidFinal;
	Shader foamDepth;
	Shader foamThickness;
	Shader foamIntensity;
	Shader foamRadiance;
	Shader finalFS;

private:
	void renderWater(glm::mat4 &projection, glm::mat4 &mView, Camera &cam, int numParticles, int numCloth);
	void renderFoam(glm::mat4 &projection, glm::mat4 &mView, Camera &cam, int numDiffuse);
	void renderCloth(glm::mat4 &projection, glm::mat4 &mView, Camera &cam, int numCloth, std::vector<int> triangles);
	void initFramebuffers();
	void setInt(Shader &shader, const int &x, const GLchar* name);
	void setFloat(Shader &shader, const float &x, const GLchar* name);
	void setVec2(Shader &shader, const glm::vec2 &v, const GLchar* name);
	void setVec3(Shader &shader, const glm::vec3 &v, const GLchar* name);
	void setVec4(Shader &shader, const glm::vec4 &v, const GLchar* name);
	void setMatrix(Shader &shader, const glm::mat4 &m, const GLchar* name);
};

#endif
