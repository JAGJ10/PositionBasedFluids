#include "Renderer.h"

using namespace std;

static const float PI = 3.14159265358979323846f;
static const int width = 1024;
static const int height = 512;
static const float zFar = 250;
static const float zNear = 30.0f;
static const float aspectRatio = width / height;
static const glm::vec2 screenSize = glm::vec2(width, height);
static const glm::vec2 blurDirX = glm::vec2(1.0f / screenSize.x, 0.0f);
static const glm::vec2 blurDirY = glm::vec2(0.0f, 1.0f / screenSize.y);
static const glm::vec4 color = glm::vec4(0.1f, 0.7f, 0.9f, 0.9f);
static float filterRadius = 5;
static const float radius = 0.95f;

Renderer::Renderer() :
	running(true),
	depth(Shader("depth.vert", "depth.frag")),
	blur(BlurShader("blur.vert", "blur.frag")),
	thickness(Shader("depth.vert", "thickness.frag")),
	composite(Shader("composite.vert", "composite.frag")),
	system(ParticleSystem())
{
	initFramebuffers();
}

Renderer::~Renderer() {}

void Renderer::run(Camera &cam) {
	if (running) {
		system.update();
	}
	//Get particle positions
	positions = system.getPositions();

	//Set camera
	glm::mat4 mView = cam.getMView();
	glm::mat4 projection = glm::perspective(cam.zoom, aspectRatio, zNear, zFar);

	//Clear buffer
	glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	//----------------------Particle Depth----------------------
	glUseProgram(depth.program);
	glBindFramebuffer(GL_FRAMEBUFFER, depth.fbo);
	glDrawBuffer(GL_NONE);
	glReadBuffer(GL_NONE);

	glClear(GL_DEPTH_BUFFER_BIT);

	depth.shaderVAOPoints(positions);

	setMatrix(depth, mView, "mView");
	setMatrix(depth, projection, "projection");
	setFloat(depth, radius, "pointRadius");
	setFloat(depth, width / aspectRatio * (1.0f / tanf(cam.zoom * 0.5f)), "pointScale");

	glDisable(GL_BLEND);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
	glEnable(GL_POINT_SPRITE);
		
	glBindVertexArray(depth.vao);
		
	glDrawArrays(GL_POINTS, 0, (GLsizei)positions.size());

	glDisable(GL_VERTEX_PROGRAM_POINT_SIZE);
	glDisable(GL_POINT_SPRITE);
	
	//--------------------Particle Blur-------------------------
	glUseProgram(blur.program);

	//Vertical blur
	glBindFramebuffer(GL_FRAMEBUFFER, blur.fboV);
	glDrawBuffer(GL_NONE);
	glReadBuffer(GL_NONE);

	glClear(GL_DEPTH_BUFFER_BIT);

	blur.shaderVAOQuad();

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, depth.tex);
	GLint depthMap = glGetUniformLocation(blur.program, "depthMap");
	glUniform1i(depthMap, 0);

	setMatrix(blur, projection, "projection");
	setVec2(blur, screenSize, "screenSize");
	setVec2(blur, blurDirY, "blurDir");
	setFloat(blur, filterRadius, "filterRadius");
	//setFloat(blur, width / aspectRatio * (1.0f / (tanf(cam.zoom*0.5f))), "blurScale");
	setFloat(blur, 0.1f, "blurScale");

	glEnable(GL_DEPTH_TEST);

	glBindVertexArray(blur.vao);

	glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

	//Horizontal blur
	glBindFramebuffer(GL_FRAMEBUFFER, blur.fboH);
	glDrawBuffer(GL_NONE);
	glReadBuffer(GL_NONE);

	glClear(GL_DEPTH_BUFFER_BIT);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, blur.texV);
	depthMap = glGetUniformLocation(blur.program, "depthMap");
	glUniform1i(depthMap, 0);

	setVec2(blur, screenSize, "screenSize");
	setMatrix(blur, projection, "projection");
	setVec2(blur, blurDirX, "blurDir");
	setFloat(blur, filterRadius, "filterRadius");
	//setFloat(blur, width / aspectRatio * (1.0f / (tanf(cam.zoom*0.5f))), "blurScale");
	setFloat(blur, 0.1f, "blurScale");

	glEnable(GL_DEPTH_TEST);

	glBindVertexArray(blur.vao);

	glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

	glDisable(GL_DEPTH_TEST);

	//--------------------Particle Thickness-------------------------
	glUseProgram(thickness.program);
	glBindFramebuffer(GL_FRAMEBUFFER, thickness.fbo);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	thickness.shaderVAOPoints(positions);

	setMatrix(thickness, mView, "mView");
	setMatrix(thickness, projection, "projection");
	setFloat(depth, radius * 4.0f, "pointRadius");
	setFloat(depth, width / aspectRatio * (1.0f / tanf(cam.zoom * 0.5f)), "pointScale");

	glEnable(GL_BLEND);
	glBlendFunc(GL_ONE, GL_ONE);
	glBlendEquation(GL_FUNC_ADD);
	glDepthMask(GL_FALSE);
	//glEnable(GL_DEPTH_TEST);
	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
	glEnable(GL_POINT_SPRITE);

	glBindVertexArray(thickness.vao);

	glDrawArrays(GL_POINTS, 0, (GLsizei)positions.size());

	glDisable(GL_VERTEX_PROGRAM_POINT_SIZE);
	glDisable(GL_POINT_SPRITE);
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_BLEND);

	//--------------------Particle Composite-------------------------
	glUseProgram(composite.program);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	composite.shaderVAOQuad();

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, blur.texH);
	depthMap = glGetUniformLocation(composite.program, "depthMap");
	glUniform1i(depthMap, 0);

	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, thickness.tex);
	GLint thicknessMap = glGetUniformLocation(composite.program, "thicknessMap");
	glUniform1i(thicknessMap, 1);

	setMatrix(composite, projection, "projection");
	setMatrix(composite, mView, "mView");
	setVec4(composite, color, "color");
	setVec2(composite, glm::vec2(1.0f / width, 1.0f / height), "invTexScale");

	glEnable(GL_DEPTH_TEST);
	glDepthMask(GL_TRUE);

	glBindVertexArray(composite.vao);

	glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

	glDisable(GL_DEPTH_TEST);
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

	// Composite buffer
	composite.initFBO(composite.fbo);
	glBindFramebuffer(GL_FRAMEBUFFER, composite.fbo);
	composite.initTexture(width, height, GL_RGBA, GL_RGBA32F, composite.tex);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, composite.tex, 0);
}

void Renderer::setInt(Shader &shader, const int &x, const GLchar* name) {
	GLint loc = glGetUniformLocation(shader.program, name);
	glUniform1i(loc, x);
}

void Renderer::setFloat(Shader &shader, const float &x, const GLchar* name) {
	GLint loc = glGetUniformLocation(shader.program, name);
	glUniform1f(loc, x);
}

void Renderer::setVec2(Shader &shader, const glm::vec2 &v, const GLchar* name) {
	GLint loc = glGetUniformLocation(shader.program, name);
	glUniform2f(loc, v.x, v.y);
}

void Renderer::setVec3(Shader &shader, const glm::vec3 &v, const GLchar* name) {
	GLint loc = glGetUniformLocation(shader.program, name);
	glUniform3f(loc, v.x, v.y, v.z);
}

void Renderer::setVec4(Shader &shader, const glm::vec4 &v, const GLchar* name) {
	GLint loc = glGetUniformLocation(shader.program, name);
	glUniform4f(loc, v.x, v.y, v.z, v.w);
}

void Renderer::setMatrix(Shader &shader, const glm::mat4 &m, const GLchar* name) {
	GLint loc = glGetUniformLocation(shader.program, name);
	glUniformMatrix4fv(loc, 1, GL_FALSE, glm::value_ptr(m));
}