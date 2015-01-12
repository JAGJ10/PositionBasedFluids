#include "Renderer.h"

using namespace std;

static const float PI = 3.14159265358979323846f;
static const int width = 512;
static const int height = 512;
static const float zFar = 200;
static const float zNear = 1;
static const float aspectRatio = width / height;
static const glm::vec2 screenSize = glm::vec2(width, height);
static const glm::vec3 color;
static const int filterRadius = 10;

Renderer::Renderer() :
	depth( Shader("depth.vert", "depth.frag") ),
	normals{ Shader("normal.vert", "normal.frag") },
	blur{ BlurShader("blur.vert", "blur.frag") },
	thickness{ Shader("depth.vert", "thickness.frag") },
	composite{ Shader("composite.vert", "composite.frag") },
	system{ ParticleSystem() }
{
	initFramebuffers();
}

Renderer::~Renderer() {}

void Renderer::run() {
	//Get particle positions
	positions = system.getPositions();
	glm::vec3 eye = glm::vec3(0, 0, -30.0f);
	glm::vec3 target = glm::vec3(10, 0, 0);
	glm::vec3 up = glm::vec3(0, -1, 0);
	glm::mat4 mView = glm::lookAt(eye, target, up);
	glm::mat4 projection = glm::perspective(30.0f, aspectRatio, zNear, zFar);

	//----------------------Particle Depth----------------------
	
	glUseProgram(depth.program);
	glBindFramebuffer(GL_FRAMEBUFFER, depth.fbo);
	glDrawBuffer(GL_NONE);
	glReadBuffer(GL_NONE);

	glClear(GL_DEPTH_BUFFER_BIT);

	depth.shaderVAOPoints(positions);

	setMatrix(depth, mView, "mView");
	setMatrix(depth, projection, "projection");
	setVec2(depth, screenSize, "screenSize");

	glDisable(GL_BLEND);
	glEnable(GL_DEPTH_TEST);
		
	glBindVertexArray(depth.vao);
		
	glDrawArrays(GL_POINTS, 0, (GLsizei)positions.size());
	

	//--------------------Particle Normals-------------------------
	
	glUseProgram(normals.program);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	normals.shaderVAOQuad();
		
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, depth.tex);
	GLint depthMap = glGetUniformLocation(normals.program, "depthMap");
	glUniform1i(depthMap, 0);
		
	setMatrix(normals, mView, "mView");
	setMatrix(normals, projection, "projection");
	setVec2(normals, screenSize, "screenSize");
	setFloat(normals, zFar, "zFar");
	setFloat(normals, zNear, "zNear");
		
	glDisable(GL_DEPTH_TEST);
		
	glBindVertexArray(normals.vao);
		
	glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
	

	//glUseProgram(simple.program);
	//simple.shaderVAOQuad();
	//glBindVertexArray(simple.vao);
	//glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
	//glBindVertexArray(0);
	system.update();
}

void Renderer::initFramebuffers() {
	// Depth buffer
	depth.initFBO(depth.fbo);
	glBindFramebuffer(GL_FRAMEBUFFER, depth.fbo);
	depth.initTexture(width, height, GL_DEPTH_COMPONENT, GL_DEPTH_COMPONENT, depth.tex);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depth.tex, 0);
	
	// Thickness buffer
	thickness.initFBO(thickness.fbo);
	glBindFramebuffer(GL_FRAMEBUFFER, thickness.fbo);
	thickness.initTexture(width, height, GL_RGBA, GL_RGBA32F, thickness.tex);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, thickness.tex, 0);

	// Blur buffer
	blur.initFBO(blur.fboV);
	glBindFramebuffer(GL_FRAMEBUFFER, blur.fboV);
	blur.initTexture(width, height, GL_DEPTH_COMPONENT, GL_DEPTH_COMPONENT, blur.texV);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, blur.texV, 0);

	blur.initFBO(blur.fboH);
	glBindFramebuffer(GL_FRAMEBUFFER, blur.fboH);
	blur.initTexture(width, height, GL_DEPTH_COMPONENT, GL_DEPTH_COMPONENT, blur.texH);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, blur.texH, 0);

	// Normal buffer
	normals.initFBO(normals.fbo);
	glBindFramebuffer(GL_FRAMEBUFFER, normals.fbo);
	normals.initTexture(width, height, GL_RGBA, GL_RGBA32F, normals.tex);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, normals.tex, 0);

	// Composite buffer
	composite.initFBO(composite.fbo);
	glBindFramebuffer(GL_FRAMEBUFFER, composite.fbo);
	composite.initTexture(width, height, GL_RGBA, GL_RGBA32F, composite.tex);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, composite.tex, 0);
}

void Renderer::setInt(Shader shader, int x, const GLchar* name) {
	GLint loc = glGetUniformLocation(shader.program, name);
	glUniform1i(loc, x);
}

void Renderer::setFloat(Shader shader, float x, const GLchar* name) {
	GLint loc = glGetUniformLocation(shader.program, name);
	glUniform1f(loc, x);
}

void Renderer::setVec2(Shader shader, glm::vec2 v, const GLchar* name) {
	GLint loc = glGetUniformLocation(shader.program, name);
	glUniform2f(loc, v.x, v.y);
}

void Renderer::setVec3(Shader shader, glm::vec3 v, const GLchar* name) {
	GLint loc = glGetUniformLocation(shader.program, name);
	glUniform3f(loc, v.x, v.y, v.z);
}

void Renderer::setVec4(Shader shader, glm::vec4 v, const GLchar* name) {
	GLint loc = glGetUniformLocation(shader.program, name);
	glUniform4f(loc, v.x, v.y, v.z, v.w);
}

void Renderer::setMatrix(Shader shader, glm::mat4 m, const GLchar* name) {
	GLint loc = glGetUniformLocation(shader.program, name);
	glUniformMatrix4fv(loc, 1, GL_FALSE, glm::value_ptr(m));
}