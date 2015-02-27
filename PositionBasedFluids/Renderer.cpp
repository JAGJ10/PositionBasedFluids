#include "Renderer.h"

using namespace std;

static const float PI = 3.14159265358979323846f;
static const int width = 1024;
static const int height = 512;
static const float zFar = 20.0f;
static const float zNear = 2.0f;
static const float aspectRatio = width / height;
static const glm::vec2 screenSize = glm::vec2(width, height);
static const glm::vec2 blurDirX = glm::vec2(1.0f / screenSize.x, 0.0f);
static const glm::vec2 blurDirY = glm::vec2(0.0f, 1.0f / screenSize.y);
static const glm::vec4 color = glm::vec4(.5f, 0.9f, 0.95f, 0.9f);
static float filterRadius = 5;
static const float radius = 0.1f;
static const float foamRadius = 0.01f;

Renderer::Renderer() :
	running(true),
	depth(Shader("depth.vert", "depth.frag")),
	blur(BlurShader("blur.vert", "blur.frag")),
	thickness(Shader("depth.vert", "thickness.frag")),
	fluidFinal(Shader("fluidFinal.vert", "fluidFinal.frag")),
	foamDepth(Shader("foamDepth.vert", "foamDepth.frag")),
	foamThickness(Shader("foamThickness.vert", "foamThickness.frag")),
	foamIntensity(Shader("foamIntensity.vert", "foamIntensity.frag")),
	sprayDepth(Shader("foamDepth.vert", "foamDepth.frag")),
	sprayThickness(Shader("foamThickness.vert", "foamThickness.frag")),
	sprayIntensity(Shader("foamIntensity.vert", "foamIntensity.frag")),
	bubbleDepth(Shader("foamDepth.vert", "foamDepth.frag")),
	bubbleThickness(Shader("foamThickness.vert", "foamThickness.frag")),
	bubbleIntensity(Shader("foamIntensity.vert", "foamIntensity.frag")),
	finalFS(Shader("final.vert", "final.frag")),
	system(ParticleSystem())
{
	initFramebuffers();
}

Renderer::~Renderer() {}

void Renderer::run(Camera &cam) {
	if (running) {
		for (int i = 0; i < 4; i++) {
			system.update();
		}
	}

	//Get particle positions
	fluidPositions = system.getFluidPositions();
	sprayPositions = system.getSprayPositions();
	foamPositions = system.getFoamPositions();
	bubblePositions = system.getBubblePositions();
	cout << "Spray: " << sprayPositions.size() << endl;
	cout << "Bubbles: " << bubblePositions.size() << endl;
	cout << "Foam: " << foamPositions.size() << endl << endl << endl;

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

	depth.shaderVAOPoints(fluidPositions);

	setMatrix(depth, mView, "mView");
	setMatrix(depth, projection, "projection");
	setFloat(depth, radius, "pointRadius");
	setFloat(depth, width / aspectRatio * (1.0f / tanf(cam.zoom * 0.5f)), "pointScale");

	glDisable(GL_BLEND);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
	glEnable(GL_POINT_SPRITE);
		
	glBindVertexArray(depth.vao);
		
	glDrawArrays(GL_POINTS, 0, (GLsizei)fluidPositions.size());

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

	thickness.shaderVAOPoints(fluidPositions);

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

	glDrawArrays(GL_POINTS, 0, (GLsizei)fluidPositions.size());

	glDisable(GL_VERTEX_PROGRAM_POINT_SIZE);
	glDisable(GL_POINT_SPRITE);
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_BLEND);

	//--------------------Particle fluidFinal-------------------------
	glUseProgram(fluidFinal.program);
	glBindFramebuffer(GL_FRAMEBUFFER, fluidFinal.fbo);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	fluidFinal.shaderVAOQuad();

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, blur.texH);
	depthMap = glGetUniformLocation(fluidFinal.program, "depthMap");
	glUniform1i(depthMap, 0);

	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, thickness.tex);
	GLint thicknessMap = glGetUniformLocation(fluidFinal.program, "thicknessMap");
	glUniform1i(thicknessMap, 1);

	setMatrix(fluidFinal, projection, "projection");
	setMatrix(fluidFinal, mView, "mView");
	setVec4(fluidFinal, color, "color");
	setVec2(fluidFinal, glm::vec2(1.0f / width, 1.0f / height), "invTexScale");

	glEnable(GL_DEPTH_TEST);
	glDepthMask(GL_TRUE);

	glBindVertexArray(fluidFinal.vao);

	glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

	glDisable(GL_DEPTH_TEST);

	//--------------------FOAM--------------------------
	renderSpray(projection, mView, cam);
	renderBubbles(projection, mView, cam);
	renderFoam(projection, mView, cam);

	//--------------------Final-------------------------
	glUseProgram(finalFS.program);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	finalFS.shaderVAOQuad();

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, fluidFinal.tex);
	GLint fluidMap = glGetUniformLocation(finalFS.program, "fluidMap");
	glUniform1i(fluidMap, 0);

	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, sprayIntensity.tex);
	GLint sprayMap = glGetUniformLocation(finalFS.program, "sprayMap");
	glUniform1i(sprayMap, 1);

	glActiveTexture(GL_TEXTURE2);
	glBindTexture(GL_TEXTURE_2D, bubbleIntensity.tex);
	GLint bubbleMap = glGetUniformLocation(finalFS.program, "bubbleMap");
	glUniform1i(bubbleMap, 2);

	glActiveTexture(GL_TEXTURE3);
	glBindTexture(GL_TEXTURE_2D, foamIntensity.tex);
	GLint foamMap = glGetUniformLocation(finalFS.program, "foamMap");
	glUniform1i(foamMap, 3);

	glBindVertexArray(finalFS.vao);

	glEnable(GL_DEPTH_TEST);
	glDepthMask(GL_TRUE);

	glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
}

void Renderer::initFramebuffers() {
	//Depth buffer
	depth.initFBO(depth.fbo);
	glBindFramebuffer(GL_FRAMEBUFFER, depth.fbo);
	depth.initTexture(width, height, GL_DEPTH_COMPONENT, GL_DEPTH_COMPONENT, depth.tex);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depth.tex, 0);
	
	//Thickness buffer
	thickness.initFBO(thickness.fbo);
	glBindFramebuffer(GL_FRAMEBUFFER, thickness.fbo);
	thickness.initTexture(width, height, GL_RGBA, GL_RGBA32F, thickness.tex);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, thickness.tex, 0);

	//Blur buffer
	blur.initFBO(blur.fboV);
	glBindFramebuffer(GL_FRAMEBUFFER, blur.fboV);
	blur.initTexture(width, height, GL_DEPTH_COMPONENT, GL_DEPTH_COMPONENT, blur.texV);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, blur.texV, 0);

	blur.initFBO(blur.fboH);
	glBindFramebuffer(GL_FRAMEBUFFER, blur.fboH);
	blur.initTexture(width, height, GL_DEPTH_COMPONENT, GL_DEPTH_COMPONENT, blur.texH);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, blur.texH, 0);

	//fluidFinal buffer
	fluidFinal.initFBO(fluidFinal.fbo);
	glBindFramebuffer(GL_FRAMEBUFFER, fluidFinal.fbo);
	fluidFinal.initTexture(width, height, GL_RGBA, GL_RGBA32F, fluidFinal.tex);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, fluidFinal.tex, 0);

	//Spray Depth buffer
	sprayDepth.initFBO(sprayDepth.fbo);
	glBindFramebuffer(GL_FRAMEBUFFER, sprayDepth.fbo);
	sprayDepth.initTexture(width, height, GL_RGBA, GL_RGBA32F, sprayDepth.tex);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, sprayDepth.tex, 0);
	sprayDepth.initTexture(width, height, GL_DEPTH_COMPONENT, GL_DEPTH_COMPONENT, sprayDepth.tex2);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, sprayDepth.tex2, 0);

	//Spray Thickness buffer
	sprayThickness.initFBO(sprayThickness.fbo);
	glBindFramebuffer(GL_FRAMEBUFFER, sprayThickness.fbo);
	sprayThickness.initTexture(width, height, GL_RGBA, GL_RGBA32F, sprayThickness.tex);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, sprayThickness.tex, 0);

	//Spray Intensity
	sprayIntensity.initFBO(sprayIntensity.fbo);
	glBindFramebuffer(GL_FRAMEBUFFER, sprayIntensity.fbo);
	sprayIntensity.initTexture(width, height, GL_RGBA, GL_RGBA32F, sprayIntensity.tex);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, sprayIntensity.tex, 0);

	//Bubble Depth buffer
	bubbleDepth.initFBO(bubbleDepth.fbo);
	glBindFramebuffer(GL_FRAMEBUFFER, bubbleDepth.fbo);
	bubbleDepth.initTexture(width, height, GL_RGBA, GL_RGBA32F, bubbleDepth.tex);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, bubbleDepth.tex, 0);
	bubbleDepth.initTexture(width, height, GL_DEPTH_COMPONENT, GL_DEPTH_COMPONENT, bubbleDepth.tex2);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, bubbleDepth.tex2, 0);

	//Bubble Thickness buffer
	bubbleThickness.initFBO(bubbleThickness.fbo);
	glBindFramebuffer(GL_FRAMEBUFFER, bubbleThickness.fbo);
	bubbleThickness.initTexture(width, height, GL_RGBA, GL_RGBA32F, bubbleThickness.tex);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, bubbleThickness.tex, 0);

	//Bubble Intensity
	bubbleIntensity.initFBO(bubbleIntensity.fbo);
	glBindFramebuffer(GL_FRAMEBUFFER, bubbleIntensity.fbo);
	bubbleIntensity.initTexture(width, height, GL_RGBA, GL_RGBA32F, bubbleIntensity.tex);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, bubbleIntensity.tex, 0);

	//Foam Depth buffer
	foamDepth.initFBO(foamDepth.fbo);
	glBindFramebuffer(GL_FRAMEBUFFER, foamDepth.fbo);
	foamDepth.initTexture(width, height, GL_RGBA, GL_RGBA32F, foamDepth.tex);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, foamDepth.tex, 0);
	foamDepth.initTexture(width, height, GL_DEPTH_COMPONENT, GL_DEPTH_COMPONENT, foamDepth.tex2);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, foamDepth.tex2, 0);

	//Foam Thickness buffer
	foamThickness.initFBO(foamThickness.fbo);
	glBindFramebuffer(GL_FRAMEBUFFER, foamThickness.fbo);
	foamThickness.initTexture(width, height, GL_RGBA, GL_RGBA32F, foamThickness.tex);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, foamThickness.tex, 0);

	//Foam Intensity
	foamIntensity.initFBO(foamIntensity.fbo);
	glBindFramebuffer(GL_FRAMEBUFFER, foamIntensity.fbo);
	foamIntensity.initTexture(width, height, GL_RGBA, GL_RGBA32F, foamIntensity.tex);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, foamIntensity.tex, 0);

	//Final buffer
	finalFS.initFBO(finalFS.fbo);
	glBindFramebuffer(GL_FRAMEBUFFER, finalFS.fbo);
	finalFS.initTexture(width, height, GL_RGBA, GL_RGBA32F, finalFS.tex);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, finalFS.tex, 0);
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

void Renderer::renderSpray(glm::mat4 &projection, glm::mat4 &mView, Camera &cam) {
	//--------------------Spray Depth-------------------------
	glUseProgram(sprayDepth.program);
	glBindFramebuffer(GL_FRAMEBUFFER, sprayDepth.fbo);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	sprayDepth.shaderVAOPointsFoam(sprayPositions);

	setMatrix(sprayDepth, projection, "projection");
	setMatrix(sprayDepth, mView, "mView");
	setFloat(sprayDepth, foamRadius, "pointRadius");
	setFloat(sprayDepth, width / aspectRatio * (1.0f / tanf(cam.zoom * 0.5f)), "pointScale");
	setFloat(sprayDepth, tanf(cam.zoom / 2), "fov");

	glEnable(GL_DEPTH_TEST);
	glDepthMask(GL_TRUE);
	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
	glEnable(GL_POINT_SPRITE);

	glBindVertexArray(sprayDepth.vao);

	glDrawArrays(GL_POINTS, 0, (GLsizei)sprayPositions.size());

	glDisable(GL_DEPTH_TEST);

	//--------------------Spray Thickness----------------------
	glUseProgram(sprayThickness.program);
	glBindFramebuffer(GL_FRAMEBUFFER, sprayThickness.fbo);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	sprayThickness.shaderVAOPointsFoam(sprayPositions);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, sprayDepth.tex);
	GLint sprayDepth = glGetUniformLocation(sprayThickness.program, "foamDepthMap");
	glUniform1i(sprayDepth, 0);

	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, depth.tex);
	GLint fluidDepth = glGetUniformLocation(sprayThickness.program, "fluidDepthMap");
	glUniform1i(fluidDepth, 1);

	setMatrix(sprayThickness, projection, "projection");
	setMatrix(sprayThickness, mView, "mView");
	setFloat(sprayThickness, foamRadius, "pointRadius");
	setFloat(sprayThickness, width / aspectRatio * (1.0f / tanf(cam.zoom * 0.5f)), "pointScale");
	setFloat(sprayThickness, tanf(cam.zoom / 2), "fov");
	setInt(sprayThickness, 0, "type");

	glEnable(GL_BLEND);
	glBlendFunc(GL_ONE, GL_ONE);
	glBlendEquation(GL_FUNC_ADD);
	glDepthMask(GL_FALSE);

	glBindVertexArray(sprayThickness.vao);

	glDrawArrays(GL_POINTS, 0, (GLsizei)sprayPositions.size());

	glDisable(GL_VERTEX_PROGRAM_POINT_SIZE);
	glDisable(GL_POINT_SPRITE);
	glDisable(GL_BLEND);

	//--------------------Spray Intensity----------------------
	glUseProgram(sprayIntensity.program);
	glBindFramebuffer(GL_FRAMEBUFFER, sprayIntensity.fbo);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	sprayIntensity.shaderVAOQuad();

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, sprayThickness.tex);
	GLint thickness = glGetUniformLocation(sprayIntensity.program, "thickness");
	glUniform1i(thickness, 0);

	glBindVertexArray(sprayIntensity.vao);

	glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
}

void Renderer::renderBubbles(glm::mat4 &projection, glm::mat4 &mView, Camera &cam) {
	//--------------------Bubble Depth-------------------------
	glUseProgram(bubbleDepth.program);
	glBindFramebuffer(GL_FRAMEBUFFER, bubbleDepth.fbo);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	bubbleDepth.shaderVAOPointsFoam(bubblePositions);

	setMatrix(bubbleDepth, projection, "projection");
	setMatrix(bubbleDepth, mView, "mView");
	setFloat(bubbleDepth, foamRadius, "pointRadius");
	setFloat(bubbleDepth, width / aspectRatio * (1.0f / tanf(cam.zoom * 0.5f)), "pointScale");
	setFloat(bubbleDepth, tanf(cam.zoom / 2), "fov");

	glEnable(GL_DEPTH_TEST);
	glDepthMask(GL_TRUE);
	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
	glEnable(GL_POINT_SPRITE);

	glBindVertexArray(bubbleDepth.vao);

	glDrawArrays(GL_POINTS, 0, (GLsizei)bubblePositions.size());

	glDisable(GL_DEPTH_TEST);

	//--------------------Bubble Thickness----------------------
	glUseProgram(bubbleThickness.program);
	glBindFramebuffer(GL_FRAMEBUFFER, bubbleThickness.fbo);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	bubbleThickness.shaderVAOPointsFoam(bubblePositions);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, bubbleDepth.tex);
	GLint bubbleDepth = glGetUniformLocation(bubbleThickness.program, "foamDepthMap");
	glUniform1i(bubbleDepth, 0);

	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, depth.tex);
	GLint fluidDepth = glGetUniformLocation(bubbleThickness.program, "fluidDepthMap");
	glUniform1i(fluidDepth, 1);

	setMatrix(bubbleThickness, projection, "projection");
	setMatrix(bubbleThickness, mView, "mView");
	setFloat(bubbleThickness, foamRadius / 2, "pointRadius");
	setFloat(bubbleThickness, width / aspectRatio * (1.0f / tanf(cam.zoom * 0.5f)), "pointScale");
	setFloat(bubbleThickness, tanf(cam.zoom / 2), "fov");
	setInt(bubbleThickness, 1, "type");

	glEnable(GL_BLEND);
	glBlendFunc(GL_ONE, GL_ONE);
	glBlendEquation(GL_FUNC_ADD);
	glDepthMask(GL_FALSE);

	glBindVertexArray(bubbleThickness.vao);

	glDrawArrays(GL_POINTS, 0, (GLsizei)bubblePositions.size());

	glDisable(GL_VERTEX_PROGRAM_POINT_SIZE);
	glDisable(GL_POINT_SPRITE);
	glDisable(GL_BLEND);

	//--------------------Bubble Intensity----------------------
	glUseProgram(bubbleIntensity.program);
	glBindFramebuffer(GL_FRAMEBUFFER, bubbleIntensity.fbo);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	bubbleIntensity.shaderVAOQuad();

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, bubbleThickness.tex);
	GLint thickness = glGetUniformLocation(bubbleIntensity.program, "thickness");
	glUniform1i(thickness, 0);

	glBindVertexArray(bubbleIntensity.vao);

	glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
}

void Renderer::renderFoam(glm::mat4 &projection, glm::mat4 &mView, Camera &cam) {
	//--------------------Foam Depth-------------------------
	glUseProgram(foamDepth.program);
	glBindFramebuffer(GL_FRAMEBUFFER, foamDepth.fbo);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	foamDepth.shaderVAOPointsFoam(foamPositions);

	setMatrix(foamDepth, projection, "projection");
	setMatrix(foamDepth, mView, "mView");
	setFloat(foamDepth, foamRadius, "pointRadius");
	setFloat(foamDepth, width / aspectRatio * (1.0f / tanf(cam.zoom * 0.5f)), "pointScale");
	setFloat(foamDepth, tanf(cam.zoom / 2), "fov");

	glEnable(GL_DEPTH_TEST);
	glDepthMask(GL_TRUE);
	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
	glEnable(GL_POINT_SPRITE);

	glBindVertexArray(foamDepth.vao);

	glDrawArrays(GL_POINTS, 0, (GLsizei)foamPositions.size());

	glDisable(GL_DEPTH_TEST);

	//--------------------Foam Thickness----------------------
	glUseProgram(foamThickness.program);
	glBindFramebuffer(GL_FRAMEBUFFER, foamThickness.fbo);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	foamThickness.shaderVAOPointsFoam(foamPositions);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, foamDepth.tex);
	GLint foamDepth = glGetUniformLocation(foamThickness.program, "foamDepthMap");
	glUniform1i(foamDepth, 0);

	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, depth.tex);
	GLint fluidDepth = glGetUniformLocation(foamThickness.program, "fluidDepthMap");
	glUniform1i(fluidDepth, 1);

	setMatrix(foamThickness, projection, "projection");
	setMatrix(foamThickness, mView, "mView");
	setFloat(foamThickness, foamRadius, "pointRadius");
	setFloat(foamThickness, width / aspectRatio * (1.0f / tanf(cam.zoom * 0.5f)), "pointScale");
	setFloat(foamThickness, tanf(cam.zoom / 2), "fov");
	setInt(foamThickness, 2, "type");

	glEnable(GL_BLEND);
	glBlendFunc(GL_ONE, GL_ONE);
	glBlendEquation(GL_FUNC_ADD);
	glDepthMask(GL_FALSE);

	glBindVertexArray(foamThickness.vao);

	glDrawArrays(GL_POINTS, 0, (GLsizei)foamPositions.size());

	glDisable(GL_VERTEX_PROGRAM_POINT_SIZE);
	glDisable(GL_POINT_SPRITE);
	glDisable(GL_BLEND);

	//--------------------Foam Intensity----------------------
	glUseProgram(foamIntensity.program);
	glBindFramebuffer(GL_FRAMEBUFFER, foamIntensity.fbo);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	foamIntensity.shaderVAOQuad();

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, foamThickness.tex);
	GLint thickness = glGetUniformLocation(foamIntensity.program, "thickness");
	glUniform1i(thickness, 0);

	glBindVertexArray(foamIntensity.vao);

	glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
}