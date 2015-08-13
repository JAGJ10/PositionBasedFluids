#ifndef SHADER_H
#define SHADER_H

#include "common.h"
#include <string>
#include <sstream>
#include <fstream>

class Shader {
public:
	GLuint program;

	Shader(const GLchar* vertexPath, const GLchar* fragmentPath);
	~Shader();

	int setUniformf(const GLchar* name, float param);
	int setUniformv2f(const GLchar* name, float param1, float param2);
	int setUniformv2f(const GLchar* name, const glm::vec2 &param);
	int setUniformv3f(const GLchar* name, float param1, float param2, float param3);
	int setUniformv3f(const GLchar* name, const glm::vec3 &param);
	int setUniformv4f(const GLchar* name, float param1, float param2, float param3, float param4);
	int setUniformv4f(const GLchar* name, const glm::vec4 &param);
	int setUniformi(const GLchar* name, int param);
	int setUniformv2i(const GLchar* name, int param1, int param2);
	int setUniformmat3(const GLchar* name, const glm::mat3 &param);
	int setUniformmat4(const GLchar* name, const glm::mat4 &param);

	//Array uniforms
	int setUniform1iv(const GLchar* name, GLuint numParams, const int* params);
	int setUniform2iv(const GLchar* name, GLuint numParams, const int* params);
	int setUniform3iv(const GLchar* name, GLuint numParams, const int* params);
	int setUniform4iv(const GLchar* name, GLuint numParams, const int* params);
	int setUniform1fv(const GLchar* name, GLuint numParams, const float* params);
	int setUniform2fv(const GLchar* name, GLuint numParams, const float* params);
	int setUniform3fv(const GLchar* name, GLuint numParams, const float* params);
	int setUniform4fv(const GLchar* name, GLuint numParams, const float* params);

	//---------------------------- Using Attribute Location ----------------------------

	void setUniformf(int paramLoc, float param);
	void setUniformv2f(int paramLoc, float param1, float param2);
	void setUniformv2f(int paramLoc, const glm::vec2 &param);
	void setUniformv3f(int paramLoc, float param1, float param2, float param3);
	void setUniformv3f(int paramLoc, const glm::vec3 &param);
	void setUniformv4f(int paramLoc, float param1, float param2, float param3, float param4);
	void setUniformv4f(int paramLoc, const glm::vec4 &param);
	void setUniformi(int paramLoc, int param);
	void setUniformv2i(int paramLoc, int param1, int param2);
	void setUniformmat3(int paramLoc, const glm::mat3 &param);
	void setUniformmat4(int paramLoc, const glm::mat4 &param);

	//Array uniforms
	void setUniform1iv(int paramLoc, GLuint numParams, const int* params);
	void setUniform2iv(int paramLoc, GLuint numParams, const int* params);
	void setUniform3iv(int paramLoc, GLuint numParams, const int* params);
	void setUniform4iv(int paramLoc, GLuint numParams, const int* params);
	void setUniform1fv(int paramLoc, GLuint numParams, const float* params);
	void setUniform2fv(int paramLoc, GLuint numParams, const float* params);
	void setUniform3fv(int paramLoc, GLuint numParams, const float* params);
	void setUniform4fv(int paramLoc, GLuint numParams, const float* params);
};

#endif