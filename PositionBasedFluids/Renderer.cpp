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
static const glm::vec4 color = glm::vec4(.275f, 0.65f, 0.85f, 0.9f);
static float filterRadius = 3;
static const float radius = 0.05f;
static const float foamRadius = 0.005f;

Renderer::Renderer() :
	running(true),
	depth(Shader("depth.vert", "depth.frag")),
	blur(BlurShader("blur.vert", "blur.frag")),
	thickness(Shader("depth.vert", "thickness.frag")),
	fluidFinal(Shader("fluidFinal.vert", "fluidFinal.frag")),
	foamDepth(Shader("foamDepth.vert", "foamDepth.frag")),
	foamThickness(Shader("foamThickness.vert", "foamThickness.frag")),
	foamIntensity(Shader("foamIntensity.vert", "foamIntensity.frag")),
	foamRadiance(Shader("radiance.vert", "radiance.frag")),
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
	foamPositions = system.getFoamPositions();
	cout << "Foam: " << foamPositions.size() << endl;

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
	glBindTexture(GL_TEXTURE_2D, foamIntensity.tex);
	GLint foamIntensityMap = glGetUniformLocation(finalFS.program, "foamIntensityMap");
	glUniform1i(foamIntensityMap, 1);

	glActiveTexture(GL_TEXTURE2);
	glBindTexture(GL_TEXTURE_2D, foamRadiance.tex);
	GLint foamRadianceMap = glGetUniformLocation(finalFS.program, "foamRadianceMap");
	glUniform1i(foamRadianceMap, 2);

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

	//Foam Radiance
	foamRadiance.initFBO(foamRadiance.fbo);
	glBindFramebuffer(GL_FRAMEBUFFER, foamRadiance.fbo);
	foamRadiance.initTexture(width, height, GL_RGBA, GL_RGBA32F, foamRadiance.tex);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, foamRadiance.tex, 0);

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
	glBindTexture(GL_TEXTURE_2D, foamDepth.tex2);
	GLint foamD = glGetUniformLocation(foamThickness.program, "foamDepthMap");
	glUniform1i(foamD, 0);

	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, depth.tex);
	GLint fluidDepth = glGetUniformLocation(foamThickness.program, "fluidDepthMap");
	glUniform1i(fluidDepth, 1);

	setMatrix(foamThickness, projection, "projection");
	setMatrix(foamThickness, mView, "mView");
	setFloat(foamThickness, foamRadius, "pointRadius");
	setFloat(foamThickness, width / aspectRatio * (1.0f / tanf(cam.zoom * 0.5f)), "pointScale");
	setFloat(foamThickness, tanf(cam.zoom / 2), "fov");
	setVec2(foamThickness, screenSize, "screenSize");

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

	//--------------------Foam Radiance----------------------
	glUseProgram(foamRadiance.program);
	glBindFramebuffer(GL_FRAMEBUFFER, foamRadiance.fbo);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	foamRadiance.shaderVAOQuad();

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, foamDepth.tex2);
	foamD = glGetUniformLocation(foamRadiance.program, "foamDepthMap");
	glUniform1i(foamD, 0);

	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, depth.tex);
	fluidDepth = glGetUniformLocation(foamRadiance.program, "fluidDepthMap");
	glUniform1i(fluidDepth, 1);

	glActiveTexture(GL_TEXTURE2);
	glBindTexture(GL_TEXTURE_2D, foamDepth.tex);
	GLint foamNormalH = glGetUniformLocation(foamRadiance.program, "foamNormalHMap");
	glUniform1i(foamNormalH, 2);

	glActiveTexture(GL_TEXTURE3);
	glBindTexture(GL_TEXTURE_2D, foamIntensity.tex);
	GLint foamIntensity = glGetUniformLocation(foamRadiance.program, "foamIntensityMap");
	glUniform1i(foamIntensity, 3);

	setMatrix(foamRadiance, mView, "mView");

	glBindVertexArray(foamRadiance.vao);

	glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
}