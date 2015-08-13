#include "Shader.h"

using namespace std;

Shader::Shader(const GLchar* vertexPath, const GLchar* fragmentPath) {
	string vertexCode;
	string fragmentCode;
	try {
		//Open files
		ifstream vShaderFile(vertexPath);
		ifstream fShaderFile(fragmentPath);
		stringstream vShaderStream, fShaderStream;
		// Read file's buffer contents into streams
		vShaderStream << vShaderFile.rdbuf();
		fShaderStream << fShaderFile.rdbuf();
		//close file handlers
		vShaderFile.close();
		fShaderFile.close();
		//Convert stream into string
		vertexCode = vShaderStream.str();
		fragmentCode = fShaderStream.str();
	}
	catch (exception e) {
		cout << "ERROR::SHADER::FILE_NOT_SUCCESFULLY_READ" << endl;
	}

	const GLchar* vShaderCode = vertexCode.c_str();
	const GLchar * fShaderCode = fragmentCode.c_str();
	//Compile shaders
	GLuint vertex, fragment;
	GLint success;
	GLchar infoLog[512];
	//Vertex Shader
	vertex = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertex, 1, &vShaderCode, NULL);
	glCompileShader(vertex);
	//Print compile errors if any
	glGetShaderiv(vertex, GL_COMPILE_STATUS, &success);

	if (!success) {
		glGetShaderInfoLog(vertex, 512, NULL, infoLog);
		cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << endl;
	}

	//Fragment Shader
	fragment = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragment, 1, &fShaderCode, NULL);
	glCompileShader(fragment);
	//Print compile errors if any
	glGetShaderiv(fragment, GL_COMPILE_STATUS, &success);

	if (!success) {
		glGetShaderInfoLog(fragment, 512, NULL, infoLog);
		cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << endl;
	}

	//Shader Program
	this->program = glCreateProgram();
	glAttachShader(this->program, vertex);
	glAttachShader(this->program, fragment);
	glLinkProgram(this->program);
	//Print linking errors if any
	glGetProgramiv(this->program, GL_LINK_STATUS, &success);

	if (!success) {
		glGetProgramInfoLog(this->program, 512, NULL, infoLog);
		cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << endl;
		std::cout << vertexPath << std::endl;
		std::cout << fragmentPath << std::endl;
	}

	//Delete the shaders as they're linked into our program now and no longer necessery
	glDeleteShader(vertex);
	glDeleteShader(fragment);
}

Shader::~Shader() {
	if (program) glDeleteProgram(program);
}

int Shader::setUniformf(const GLchar* name, float param) {
	GLint loc = glGetUniformLocation(program, name);

	if (loc != -1)
		glUniform1f(loc, param);

	return loc;
}

int Shader::setUniformv2f(const GLchar* name, float param1, float param2) {
	GLint loc = glGetUniformLocation(program, name);

	if (loc != -1)
		glUniform2f(loc, param1, param2);

	return loc;
}

int Shader::setUniformv2f(const GLchar* name, const glm::vec2 &param) {
	return setUniformv2f(name, param.x, param.y);
}

int Shader::setUniformv3f(const GLchar* name, float param1, float param2, float param3) {
	GLint loc = glGetUniformLocation(program, name);

	if (loc != -1)
		glUniform3f(loc, param1, param2, param3);

	return loc;
}

int Shader::setUniformv3f(const GLchar* name, const glm::vec3 &param) {
	return setUniformv3f(name, param.x, param.y, param.z);
}

int Shader::setUniformv4f(const GLchar* name, float param1, float param2, float param3, float param4) {
	GLint loc = glGetUniformLocation(program, name);

	if (loc != -1)
		glUniform4f(loc, param1, param2, param3, param4);

	return loc;
}

int Shader::setUniformv4f(const GLchar* name, const glm::vec4 &param) {
	return setUniformv4f(name, param.x, param.y, param.z, param.w);
}

int Shader::setUniformi(const GLchar* name, int param) {
	GLint loc = glGetUniformLocation(program, name);

	if (loc != -1)
		glUniform1i(loc, param);

	return loc;
}

int Shader::setUniformv2i(const GLchar* name, int param1, int param2) {
	GLint loc = glGetUniformLocation(program, name);

	if (loc != -1)
		glUniform2i(loc, param1, param2);

	return loc;
}

int Shader::setUniformmat3(const GLchar* name, const glm::mat3& param) {
	GLint loc = glGetUniformLocation(program, name);

	if (loc != -1)
		glUniformMatrix3fv(loc, 1, GL_FALSE, glm::value_ptr(param));

	return loc;
}

int Shader::setUniformmat4(const GLchar* name, const glm::mat4& param) {
	GLint loc = glGetUniformLocation(program, name);

	if (loc != -1)
		glUniformMatrix4fv(loc, 1, GL_FALSE, glm::value_ptr(param));

	return loc;
}

int Shader::setUniform1iv(const GLchar* name, GLuint numParams, const int* params) {
	GLint loc = glGetUniformLocation(program, name);

	if (loc != -1)
		glUniform1iv(loc, numParams, params);

	return loc;
}

int Shader::setUniform2iv(const GLchar* name, GLuint numParams, const int* params) {
	GLint loc = glGetUniformLocation(program, name);

	if (loc != -1)
		glUniform2iv(loc, numParams, params);

	return loc;
}

int Shader::setUniform3iv(const GLchar* name, GLuint numParams, const int* params) {
	GLint loc = glGetUniformLocation(program, name);

	if (loc != -1)
		glUniform3iv(loc, numParams, params);

	return loc;
}

int Shader::setUniform4iv(const GLchar* name, GLuint numParams, const int* params) {
	GLint loc = glGetUniformLocation(program, name);

	if (loc != -1)
		glUniform4iv(loc, numParams, params);

	return loc;
}

int Shader::setUniform1fv(const GLchar* name, GLuint numParams, const float* params) {
	GLint loc = glGetUniformLocation(program, name);

	if (loc != -1)
		glUniform1fv(loc, numParams, params);

	return loc;
}

int Shader::setUniform2fv(const GLchar* name, GLuint numParams, const float* params) {
	GLint loc = glGetUniformLocation(program, name);

	if (loc != -1)
		glUniform2fv(loc, numParams, params);

	return loc;
}

int Shader::setUniform3fv(const GLchar* name, GLuint numParams, const float* params) {
	GLint loc = glGetUniformLocation(program, name);

	if (loc != -1)
		glUniform3fv(loc, numParams, params);

	return loc;
}

int Shader::setUniform4fv(const GLchar* name, GLuint numParams, const float* params) {
	GLint loc = glGetUniformLocation(program, name);

	if (loc != -1)
		glUniform4fv(loc, numParams, params);

	return loc;
}

//-------------------- Attribute Location Versions ----------------------

void Shader::setUniformf(int loc, float param) {
	if (loc != -1)
		glUniform1f(loc, param);
}

void Shader::setUniformv2f(int loc, float param1, float param2) {
	if (loc != -1)
		glUniform2f(loc, param1, param2);
}

void Shader::setUniformv2f(int loc, const glm::vec2 &param) {
	setUniformv2f(loc, param.x, param.y);
}

void Shader::setUniformv3f(int loc, float param1, float param2, float param3) {
	if (loc != -1)
		glUniform3f(loc, param1, param2, param3);
}

void Shader::setUniformv3f(int loc, const glm::vec3 &param) {
	setUniformv3f(loc, param.x, param.y, param.z);
}

void Shader::setUniformv4f(int loc, float param1, float param2, float param3, float param4) {
	if (loc != -1)
		glUniform4f(loc, param1, param2, param3, param4);
}

void Shader::setUniformv4f(int loc, const glm::vec4 &param) {
	setUniformv4f(loc, param.x, param.y, param.z, param.w);
}

void Shader::setUniformi(int loc, int param) {
	if (loc != -1)
		glUniform1i(loc, param);
}

void Shader::setUniformv2i(int loc, int param1, int param2) {
	if (loc != -1)
		glUniform2i(loc, param1, param2);
}

void Shader::setUniformmat3(int loc, const glm::mat3 &param) {
	if (loc != -1)
		glUniformMatrix3fv(loc, 1, false, glm::value_ptr(param));
}

void Shader::setUniformmat4(int loc, const glm::mat4 &param) {
	if (loc != -1)
		glUniformMatrix4fv(loc, 1, false, glm::value_ptr(param));
}

void Shader::setUniform1iv(int loc, GLuint numParams, const int* params) {
	if (loc != -1)
		glUniform1iv(loc, numParams, params);
}

void Shader::setUniform2iv(int loc, GLuint numParams, const int* params) {
	if (loc != -1)
		glUniform2iv(loc, numParams, params);
}

void Shader::setUniform3iv(int loc, GLuint numParams, const int* params) {
	if (loc != -1)
		glUniform3iv(loc, numParams, params);
}

void Shader::setUniform4iv(int loc, GLuint numParams, const int* params) {
	if (loc != -1)
		glUniform4iv(loc, numParams, params);
}

void Shader::setUniform1fv(int loc, GLuint numParams, const float* params) {
	if (loc != -1)
		glUniform1fv(loc, numParams, params);
}

void Shader::setUniform2fv(int loc, GLuint numParams, const float* params) {
	if (loc != -1)
		glUniform2fv(loc, numParams, params);
}

void Shader::setUniform3fv(int loc, GLuint numParams, const float* params) {
	if (loc != -1)
		glUniform3fv(loc, numParams, params);
}

void Shader::setUniform4fv(int loc, GLuint numParams, const float* params) {
	if (loc != -1)
		glUniform4fv(loc, numParams, params);
}