#ifndef RENDERER_H
#define RENDERER_H

#include "common.h"
#include "Cell.hpp"
#include "ParticleSystem.h"
#include "Shader.hpp"
#include <GLFW/glfw3.h>
#include "Camera.hpp"

class Renderer {
public:
	Renderer();
	~Renderer();
	void run(Camera &cam);

	ParticleSystem system;
	std::vector<glm::vec3> positions;

	Shader depth;
	Shader normals;
	BlurShader blur;
	Shader thickness;
	Shader composite;

private:
	void initFramebuffers();
	void setInt(Shader shader, int x, const GLchar* name);
	void setFloat(Shader shader, float x, const GLchar* name);
	void setVec2(Shader shader, glm::vec2 v, const GLchar* name);
	void setVec3(Shader shader, glm::vec3 v, const GLchar* name);
	void setVec4(Shader shader, glm::vec4 v, const GLchar* name);
	void setMatrix(Shader shader, glm::mat4 m, const GLchar* name);
};

#endif
