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

	bool running;

	Shader depth;
	Shader normals;
	BlurShader blur;
	Shader thickness;
	BlurShader thicknessBlur;
	Shader composite;

private:
	void initFramebuffers();
	void setInt(Shader &shader, const int &x, const GLchar* name);
	void setFloat(Shader &shader, const float &x, const GLchar* name);
	void setVec2(Shader &shader, const glm::vec2 &v, const GLchar* name);
	void setVec3(Shader &shader, const glm::vec3 &v, const GLchar* name);
	void setVec4(Shader &shader, const glm::vec4 &v, const GLchar* name);
	void setMatrix(Shader &shader, const glm::mat4 &m, const GLchar* name);
};

#endif
