#ifndef SHADER_H
#define SHADER_H

#include <GL/glew.h>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include "common.h"

class Shader {
public:
	GLuint program, tex, tex2, fbo, vao, vbo, ebo;

	Shader(const GLchar* vertexPath, const GLchar* fragmentPath) {
		std::string vertexCode;
		std::string fragmentCode;
		try {
			// Open files
			std::ifstream vShaderFile(vertexPath);
			std::ifstream fShaderFile(fragmentPath);
			std::stringstream vShaderStream, fShaderStream;
			// Read file's buffer contents into streams
			vShaderStream << vShaderFile.rdbuf();
			fShaderStream << fShaderFile.rdbuf();
			// close file handlers
			vShaderFile.close();
			fShaderFile.close();
			// Convert stream into string
			vertexCode = vShaderStream.str();
			fragmentCode = fShaderStream.str();
		} catch (std::exception e) {
			std::cout << "ERROR::SHADER::FILE_NOT_SUCCESFULLY_READ" << std::endl;
		}

		const GLchar* vShaderCode = vertexCode.c_str();
		const GLchar * fShaderCode = fragmentCode.c_str();
		// 2. Compile shaders
		GLuint vertex, fragment;
		GLint success;
		GLchar infoLog[512];
		// Vertex Shader
		vertex = glCreateShader(GL_VERTEX_SHADER);
		glShaderSource(vertex, 1, &vShaderCode, NULL);
		glCompileShader(vertex);
		// Print compile errors if any
		glGetShaderiv(vertex, GL_COMPILE_STATUS, &success);

		if (!success) {
			glGetShaderInfoLog(vertex, 512, NULL, infoLog);
			std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
		}

		// Fragment Shader
		fragment = glCreateShader(GL_FRAGMENT_SHADER);
		glShaderSource(fragment, 1, &fShaderCode, NULL);
		glCompileShader(fragment);
		// Print compile errors if any
		glGetShaderiv(fragment, GL_COMPILE_STATUS, &success);

		if (!success) {
			glGetShaderInfoLog(fragment, 512, NULL, infoLog);
			std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
		}

		// Shader Program
		this->program = glCreateProgram();
		glAttachShader(this->program, vertex);
		glAttachShader(this->program, fragment);
		glLinkProgram(this->program);
		// Print linking errors if any
		glGetProgramiv(this->program, GL_LINK_STATUS, &success);

		if (!success) {
			glGetProgramInfoLog(this->program, 512, NULL, infoLog);
			std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog <<  std::endl;
			std::cout << vertexPath << std::endl;
			std::cout << fragmentPath << std::endl;
		}
		
		// Delete the shaders as they're linked into our program now and no longer necessery
		glDeleteShader(vertex);
		glDeleteShader(fragment);

	}

	~Shader() {
		glDeleteBuffers(1, &fbo);
		glDeleteBuffers(1, &vbo);
		glDeleteBuffers(1, &ebo);
		glDeleteVertexArrays(1, &vao);
		glDeleteTextures(1, &tex);
		glDeleteTextures(1, &tex2);
		glDeleteProgram(program);
	}

	void initFBO(GLuint &fbo) {
		glGenBuffers(1, &ebo);
		glGenBuffers(1, &vbo);
		glGenVertexArrays(1, &vao);
		glGenFramebuffers(1, &fbo);
		glBindFramebuffer(GL_FRAMEBUFFER, fbo);
	}

	void initTexture(int width, int height, GLenum format, GLenum internalFormat, GLuint &tex) {
		glGenTextures(1, &tex);
		glBindTexture(GL_TEXTURE_2D, tex);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, width, height, 0, format, GL_FLOAT, NULL);
		glBindTexture(GL_TEXTURE_2D, 0);
	}

	void bindPositionVAO(GLuint &posVBO, int offset) {
		glBindVertexArray(vao);

		glBindBuffer(GL_ARRAY_BUFFER, posVBO);
		glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, (GLvoid*)(offset*sizeof(float4)));
		glEnableVertexAttribArray(0);
	}

	void bindVelocityVAO(GLuint &diffuseVBO) {
		glBindVertexArray(vao);

		glBindBuffer(GL_ARRAY_BUFFER, diffuseVBO);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (GLvoid*)0);
		glEnableVertexAttribArray(0);
	}

	void shaderVAOQuad() {
		GLfloat vertices[] = {
			1.0f, 1.0f,	   // Top Right
			1.0f, -1.0f,   // Bottom Right
			-1.0f, -1.0f,  // Bottom Left
			-1.0f, 1.0f	   // Top Left 
		};
		GLuint indices[] = {  // Note that we start from 0!
			0, 1, 3,	// First Triangle
			1, 2, 3	// Second Triangle
		};

		glBindVertexArray(vao);
		
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, (GLvoid*)0);
		glEnableVertexAttribArray(0);
	}

	void shaderVAOInfinitePlane() {
		GLfloat vertices[] = {
			100.0f, 0.0f, 100.0f,
			100.0f, 0.0f, -100.0f,
			-100.0f, 0.0f, -100.0f,
			-100.0f, 0.0f, 100.0f
		};
		GLuint indices[] = {
			0, 1, 3,
			1, 2, 3
		};

		glBindVertexArray(vao);

		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (GLvoid*)0);
		glEnableVertexAttribArray(0);
	}
};

class BlurShader : public Shader {
public:
	GLuint fboV, fboH, texV, texH;

	BlurShader(const GLchar* vertexPath, const GLchar* fragmentPath) : Shader(vertexPath, fragmentPath) {}
};

#endif