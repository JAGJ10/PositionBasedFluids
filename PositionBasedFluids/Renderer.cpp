#include "Renderer.h"
#include <cuda_gl_interop.h>

using namespace std;

static const float zFar = 200.0f;
static const float zNear = 5.0f;
static const glm::vec4 color = glm::vec4(.275f, 0.65f, 0.85f, 0.9f);
static const float filterRadius = 3;
static const float radius = 0.05f;
static const float clothRadius = 0.03f;
static const float foamRadius = 0.01f;

Renderer::Renderer(int width, int height) :
	width(width),
	height(height),
	aspectRatio((float)width / (float)height),
	screenSize(glm::vec2(width, height)),
	gBuffer(GBuffer(width, height)),
	plane(Shader("plane.vert", "plane.frag")),
	cloth(Shader("clothMesh.vert", "clothMesh.frag")),
	depth(Shader("depth.vert", "depth.frag")),
	blur(Shader("blur.vert", "blur.frag")),
	thickness(Shader("depth.vert", "thickness.frag")),
	fluidFinal(Shader("fluidFinal.vert", "fluidFinal.frag")),
	foamDepth(Shader("foamDepth.vert", "foamDepth.frag")),
	foamThickness(Shader("foamThickness.vert", "foamThickness.frag")),
	foamIntensity(Shader("foamIntensity.vert", "foamIntensity.frag")),
	foamRadiance(Shader("radiance.vert", "radiance.frag")),
	finalFS(Shader("final.vert", "final.frag")),
	fsQuad(FullscreenQuad())
{
	blurDirX = glm::vec2(1.0f / screenSize.x, 0.0f);
	blurDirY = glm::vec2(0.0f, 1.0f / screenSize.y);

	//Floor
	glGenVertexArrays(1, &planeBuf.vao);

	glGenBuffers(1, &planeBuf.vbo);
	glBindBuffer(GL_ARRAY_BUFFER, planeBuf.vbo);
	//glBufferData(GL_ARRAY_BUFFER, sizeof(floorVertices), floorVertices, GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glGenBuffers(1, &planeBuf.ebo);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, planeBuf.ebo);
	//glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

Renderer::~Renderer() {
	cudaDeviceSynchronize();
	cudaGraphicsUnregisterResource(resources[0]);
	cudaGraphicsUnregisterResource(resources[1]);
	cudaGraphicsUnregisterResource(resources[2]);
	glDeleteBuffers(1, &positionVBO);
	glDeleteBuffers(1, &diffusePosVBO);
	glDeleteBuffers(1, &diffuseVelVBO);
}

void Renderer::initVBOS(int numParticles, int numDiffuse, vector<int> triangles) {
	glGenBuffers(1, &positionVBO);
	glBindBuffer(GL_ARRAY_BUFFER, positionVBO);
	glBufferData(GL_ARRAY_BUFFER, numParticles * 4 * sizeof(float), 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	cudaGraphicsGLRegisterBuffer(&resources[0], positionVBO, cudaGraphicsRegisterFlagsWriteDiscard);

	glGenBuffers(1, &diffusePosVBO);
	glBindBuffer(GL_ARRAY_BUFFER, diffusePosVBO);
	glBufferData(GL_ARRAY_BUFFER, numDiffuse * 4 * sizeof(float), 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	cudaGraphicsGLRegisterBuffer(&resources[1], diffusePosVBO, cudaGraphicsRegisterFlagsWriteDiscard);

	glGenBuffers(1, &diffuseVelVBO);
	glBindBuffer(GL_ARRAY_BUFFER, diffuseVelVBO);
	glBufferData(GL_ARRAY_BUFFER, numDiffuse * 3 * sizeof(float), 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	cudaGraphicsGLRegisterBuffer(&resources[2], diffuseVelVBO, cudaGraphicsRegisterFlagsWriteDiscard);

	glGenBuffers(1, &indicesVBO);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indicesVBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint) * triangles.size(), &triangles[0], GL_STATIC_DRAW);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

void Renderer::run(int numParticles, int numDiffuse, int numCloth, vector<int> triangles, Camera &cam) {
	//Set camera
	mView = cam.getMView();
	normalMatrix = glm::inverseTranspose(mView);
	projection = glm::perspective(cam.zoom, aspectRatio, zNear, zFar);
	//glm::mat4 projection = glm::infinitePerspective(cam.zoom, aspectRatio, zNear);

	gBuffer.bindDraw();

	//Clear buffer
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	//----------------------Infinite Plane---------------------
	renderPlane(planeBuf);

	//--------------------CLOTH-------------------------
	renderCloth(projection, mView, cam, numCloth, triangles);

	//--------------------WATER-------------------------
	renderWater(projection, mView, cam, numParticles - numCloth, numCloth);

	//--------------------FOAM--------------------------
	renderFoam(projection, mView, cam, numDiffuse);

	//--------------------Final - WATER & DIFFUSE-------------------------
	glUseProgram(finalFS.program);
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, gBuffer.fluid);
	GLint fluidMap = glGetUniformLocation(finalFS.program, "fluidMap");
	glUniform1i(fluidMap, 0);

	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, gBuffer.foamIntensity);
	GLint foamIntensityMap = glGetUniformLocation(finalFS.program, "foamIntensityMap");
	glUniform1i(foamIntensityMap, 1);

	glActiveTexture(GL_TEXTURE2);
	glBindTexture(GL_TEXTURE_2D, gBuffer.foamRadiance);
	GLint foamRadianceMap = glGetUniformLocation(finalFS.program, "foamRadianceMap");
	glUniform1i(foamRadianceMap, 2);

	finalFS.setUniformv2f("screenSize", screenSize);

	glEnable(GL_DEPTH_TEST);
	glDepthMask(GL_TRUE);

	fsQuad.render();
}

void Renderer::renderPlane(buffers &buf) {
	glUseProgram(plane.program);
	gBuffer.setDrawPlane();

	plane.setUniformmat4("mView", mView);
	plane.setUniformmat4("projection", projection);

	glBindVertexArray(buf.vao);
	glBindBuffer(GL_ARRAY_BUFFER, buf.vbo);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buf.ebo);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(0);
	glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
}

void Renderer::renderWater(glm::mat4 &projection, glm::mat4 &mView, Camera &cam, int numParticles, int numCloth) {
	//----------------------Particle Depth----------------------
	glUseProgram(depth.program);
	gBuffer.setDrawDepth();	

	glClear(GL_DEPTH_BUFFER_BIT);

	depth.bindPositionVAO(positionVBO, numCloth);
	
	depth.setUniformmat4("mView", mView);
	depth.setUniformmat4("projection", projection);
	depth.setUniformf("pointRadius", radius);
	depth.setUniformf("pointScale", width / aspectRatio * (1.0f / tanf(cam.zoom * 0.5f)));
	
	glDisable(GL_BLEND);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
	
	glDrawArrays(GL_POINTS, 0, (GLsizei)numParticles);

	glDisable(GL_DEPTH_TEST);
	glDisable(GL_VERTEX_PROGRAM_POINT_SIZE);

	//--------------------Particle Blur-------------------------
	glUseProgram(blur.program);

	//Vertical blur
	gBuffer.setDrawVerticalBlur();

	glClear(GL_COLOR_BUFFER_BIT);
	
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, gBuffer.depth);
	GLint depthMap = glGetUniformLocation(blur.program, "depthMap");
	glUniform1i(depthMap, 0);

	blur.setUniformmat4("projection", projection);
	blur.setUniformv2f("screenSize", screenSize);
	blur.setUniformv2f("blurDir", blurDirY);
	blur.setUniformf("filterRadius", filterRadius);
	blur.setUniformf("blurScale", 0.1f);
	//setFloat(blur, width / aspectRatio * (1.0f / (tanf(cam.zoom*0.5f))), "blurScale");

	fsQuad.render();

	//Horizontal blur
	gBuffer.setDrawHorizontalBlur();

	glClear(GL_COLOR_BUFFER_BIT);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, gBuffer.blurV);

	blur.setUniformi("depthMap", 0);
	blur.setUniformv2f("blurDir", blurDirY);

	fsQuad.render();

	//--------------------Particle Thickness-------------------------
	glUseProgram(thickness.program);
	gBuffer.setDrawThickness();

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	thickness.bindPositionVAO(positionVBO, numCloth);

	thickness.setUniformmat4("mView", mView);
	thickness.setUniformmat4("projection", projection);
	thickness.setUniformf("pointRadius", radius * 2.0f);
	thickness.setUniformf("pointScale", width / aspectRatio * (1.0f / tanf(cam.zoom * 0.5f)));

	glEnable(GL_BLEND);
	glBlendFunc(GL_ONE, GL_ONE);
	glBlendEquation(GL_FUNC_ADD);
	glDepthMask(GL_FALSE);
	//glEnable(GL_DEPTH_TEST);
	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);

	glDrawArrays(GL_POINTS, 0, (GLsizei)numParticles);

	glDisable(GL_VERTEX_PROGRAM_POINT_SIZE);
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_BLEND);

	//--------------------Particle fluidFinal-------------------------
	glUseProgram(fluidFinal.program);
	gBuffer.setDrawFluid();

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, gBuffer.blurH);
	depthMap = glGetUniformLocation(fluidFinal.program, "depthMap");
	glUniform1i(depthMap, 0);

	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, gBuffer.thickness);
	GLint thicknessMap = glGetUniformLocation(fluidFinal.program, "thicknessMap");
	glUniform1i(thicknessMap, 1);

	glActiveTexture(GL_TEXTURE2);
	glBindTexture(GL_TEXTURE_2D, gBuffer.plane);
	GLint sceneMap = glGetUniformLocation(fluidFinal.program, "sceneMap");
	glUniform1i(sceneMap, 2);

	fluidFinal.setUniformmat4("projection", projection);
	fluidFinal.setUniformmat4("mView", mView);
	fluidFinal.setUniformv4f("color", color);
	fluidFinal.setUniformv2f("invTexScale", glm::vec2(1.0f / width, 1.0f / height));

	glEnable(GL_DEPTH_TEST);
	glDepthMask(GL_TRUE);

	fsQuad.render();

	glDisable(GL_DEPTH_TEST);
}

void Renderer::renderFoam(glm::mat4 &projection, glm::mat4 &mView, Camera &cam, int numDiffuse) {
	//--------------------Foam Depth-------------------------
	glUseProgram(foamDepth.program);
	gBuffer.setDrawFoamDepth();

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	foamDepth.bindPositionVAO(diffusePosVBO, 0);

	foamDepth.setUniformmat4("mView", mView);
	foamDepth.setUniformmat4("projection", projection);
	foamDepth.setUniformf("pointRadius", foamRadius);
	foamDepth.setUniformf("pointScale", width / aspectRatio * (1.0f / tanf(cam.zoom * 0.5f)));
	foamDepth.setUniformf("fov", tanf(cam.zoom * 0.5f));

	glEnable(GL_DEPTH_TEST);
	//glDepthMask(GL_TRUE);
	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);

	glDrawArrays(GL_POINTS, 0, (GLsizei)numDiffuse);

	glDisable(GL_DEPTH_TEST);

	//--------------------Foam Thickness----------------------
	glUseProgram(foamThickness.program);
	gBuffer.setDrawFoamThickness();

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	foamThickness.bindPositionVAO(diffusePosVBO, 0);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, gBuffer.foamDepth);
	GLint foamD = glGetUniformLocation(foamThickness.program, "foamDepthMap");
	glUniform1i(foamD, 0);

	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, gBuffer.depth);
	GLint fluidDepth = glGetUniformLocation(foamThickness.program, "fluidDepthMap");
	glUniform1i(fluidDepth, 1);

	foamThickness.setUniformmat4("mView", mView);
	foamThickness.setUniformmat4("projection", projection);
	foamThickness.setUniformf("pointRadius", foamRadius);
	foamThickness.setUniformf("pointScale", width / aspectRatio * (1.0f / tanf(cam.zoom * 0.5f)));
	foamThickness.setUniformf("fov", tanf(cam.zoom * 0.5f));
	foamThickness.setUniformv2f("screenSize", screenSize);
	foamThickness.setUniformf("zNear", zNear);
	foamThickness.setUniformf("zFar", zFar);

	glEnable(GL_BLEND);
	glBlendFunc(GL_ONE, GL_ONE);
	glBlendEquation(GL_FUNC_ADD);
	glDepthMask(GL_FALSE);

	glDrawArrays(GL_POINTS, 0, (GLsizei)numDiffuse);

	glDisable(GL_VERTEX_PROGRAM_POINT_SIZE);
	glDisable(GL_BLEND);

	//--------------------Foam Intensity----------------------
	glUseProgram(foamIntensity.program);
	gBuffer.setDrawFoamIntensity();

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, gBuffer.foamThickness);
	GLint thickness = glGetUniformLocation(foamIntensity.program, "thickness");
	glUniform1i(thickness, 0);

	fsQuad.render();

	//--------------------Foam Radiance----------------------
	glUseProgram(foamRadiance.program);
	gBuffer.setDrawFoamRadiance();

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, gBuffer.foamDepth);
	foamD = glGetUniformLocation(foamRadiance.program, "foamDepthMap");
	glUniform1i(foamD, 0);

	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, gBuffer.depth);
	fluidDepth = glGetUniformLocation(foamRadiance.program, "fluidDepthMap");
	glUniform1i(fluidDepth, 1);

	glActiveTexture(GL_TEXTURE2);
	glBindTexture(GL_TEXTURE_2D, gBuffer.foamDepth); //FIXME MISSING TEXTURE FOR FOAM'S NORMAL MAP (normal map doesn't make sense anyway, try changing the AO to a regular AO?)
	GLint foamNormalH = glGetUniformLocation(foamRadiance.program, "foamNormalHMap");
	glUniform1i(foamNormalH, 2);

	glActiveTexture(GL_TEXTURE3);
	glBindTexture(GL_TEXTURE_2D, gBuffer.foamIntensity);
	GLint foamIntensity = glGetUniformLocation(foamRadiance.program, "foamIntensityMap");
	glUniform1i(foamIntensity, 3);

	foamRadiance.setUniformmat4("mView", mView);
	foamRadiance.setUniformmat4("projection", projection);
	foamRadiance.setUniformf("zNear", zNear);
	foamRadiance.setUniformf("zFar", zFar);

	fsQuad.render();
}

void Renderer::renderCloth(glm::mat4 &projection, glm::mat4 &mView, Camera &cam, int numCloth, std::vector<int> triangles) {
	glUseProgram(cloth.program);
	gBuffer.setDrawCloth();

	cloth.bindPositionVAO(positionVBO, 0);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indicesVBO);

	cloth.setUniformmat4("mView", mView);
	cloth.setUniformmat4("projection", projection);
	
	glDrawElements(GL_TRIANGLES, (GLsizei)triangles.size(), GL_UNSIGNED_INT, 0);
}