#include "Renderer.h"
#include <cuda_gl_interop.h>

using namespace std;

static const float zFar = 2000.0f;
static const float zNear = 5.0f;
static const glm::vec4 color = glm::vec4(.275f, 0.65f, 0.85f, 0.9f);
static const float filterRadius = 3;
static const float radius = 0.05f;
static const float clothRadius = 0.03f;
static const float foamRadius = 0.01f;
static const glm::vec4 lightDir = glm::vec4(1, 1, 1, 0);

Renderer::Renderer(int width, int height) :
	width(width),
	height(height),
	aspectRatio((float)width / (float)height),
	screenSize(glm::vec2(width, height)),
	gBuffer(GBuffer(width, height)),
	fluidBuffer(FluidBuffer(width, height)),
	dLightShadow(ShadowMap(width, height)),
	plane(Shader("plane.vert", "plane.frag")),
	geometry(Shader("gbuffer.vert", "gbuffer.frag")),
	finalPass(Shader("ubershader.vert", "ubershader.frag")),
	shadow(Shader("shadow.vert", "empty.frag")),
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
	dLightMView = glm::lookAt(glm::vec3(15.0f, 15.0f, 0.0f), glm::vec3(0), glm::vec3(0, 1, 0));
	dLightProjection = glm::ortho(-20.0f, 20.0f, -20.0f, 20.0f, zNear, zFar);

	blurDirX = glm::vec2(1.0f / screenSize.x, 0.0f);
	blurDirY = glm::vec2(0.0f, 1.0f / screenSize.y);

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

	//Floor
	glGenFramebuffers(1, &planeBuf.fbo);
	glBindFramebuffer(GL_FRAMEBUFFER, planeBuf.fbo);
	glGenVertexArrays(1, &planeBuf.vao);
	glGenTextures(1, &planeBuf.tex);

	glBindTexture(GL_TEXTURE_2D, planeBuf.tex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, width, height, 0, GL_RGBA, GL_FLOAT, nullptr);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, planeBuf.tex, 0);

	glGenBuffers(1, &planeBuf.vbo);
	glBindBuffer(GL_ARRAY_BUFFER, planeBuf.vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glGenBuffers(1, &planeBuf.ebo);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, planeBuf.ebo);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	glBindVertexArray(planeBuf.vao);
	glBindBuffer(GL_ARRAY_BUFFER, planeBuf.vbo);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, planeBuf.ebo);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(0);
	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

Renderer::~Renderer() {
	//cudaDeviceSynchronize();
	cudaGraphicsUnregisterResource(resources[0]);
	cudaGraphicsUnregisterResource(resources[1]);
	cudaGraphicsUnregisterResource(resources[2]);
	glDeleteVertexArrays(1, &positionVAO);
	glDeleteVertexArrays(1, &diffusePosVAO);
	glDeleteVertexArrays(1, &indicesVAO);
	glDeleteBuffers(1, &positionVBO);
	glDeleteBuffers(1, &diffusePosVBO);
	glDeleteBuffers(1, &diffuseVelVBO);
}

void Renderer::setMeshes(const std::vector<Mesh> &meshes) {
	this->meshes = meshes;
}

void Renderer::initVBOS(int numParticles, int numDiffuse, int numCloth, vector<int> triangles) {
	glGenVertexArrays(1, &positionVAO);
	glGenVertexArrays(1, &diffusePosVAO);
	glGenVertexArrays(1, &indicesVAO);

	glBindVertexArray(positionVAO);

	glGenBuffers(1, &positionVBO);
	glBindBuffer(GL_ARRAY_BUFFER, positionVBO);
	glBufferData(GL_ARRAY_BUFFER, numParticles * 4 * sizeof(float), 0, GL_DYNAMIC_DRAW);
	glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, (GLvoid*)(numCloth*sizeof(float4)));
	glEnableVertexAttribArray(0);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

	cudaGraphicsGLRegisterBuffer(&resources[0], positionVBO, cudaGraphicsRegisterFlagsWriteDiscard);

	glBindVertexArray(diffusePosVAO);

	glGenBuffers(1, &diffusePosVBO);
	glBindBuffer(GL_ARRAY_BUFFER, diffusePosVBO);
	glBufferData(GL_ARRAY_BUFFER, numDiffuse * 4 * sizeof(float), 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindBuffer(GL_ARRAY_BUFFER, diffusePosVBO);
	glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(0);
	glBindVertexArray(0);

	cudaGraphicsGLRegisterBuffer(&resources[1], diffusePosVBO, cudaGraphicsRegisterFlagsWriteDiscard);

	glGenBuffers(1, &diffuseVelVBO);
	glBindBuffer(GL_ARRAY_BUFFER, diffuseVelVBO);
	glBufferData(GL_ARRAY_BUFFER, numDiffuse * 3 * sizeof(float), 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	cudaGraphicsGLRegisterBuffer(&resources[2], diffuseVelVBO, cudaGraphicsRegisterFlagsWriteDiscard);
	if (numCloth != 0) {
		glGenBuffers(1, &indicesVBO);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indicesVBO);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint)* triangles.size(), &triangles[0], GL_STATIC_DRAW);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	}

	glBindVertexArray(0);
}

void Renderer::run(int numParticles, int numDiffuse, int numCloth, vector<int> triangles, Camera &cam) {
	//Set camera
	mView = cam.getMView();
	normalMatrix = glm::mat3(glm::inverseTranspose(mView));
	projection = glm::perspective(cam.zoom, aspectRatio, zNear, zFar);
	//glm::mat4 projection = glm::infinitePerspective(cam.zoom, aspectRatio, zNear);
	//Clear buffer
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	//----------------------Infinite Plane---------------------
	renderPlane(planeBuf);

	//Render geometry to gBuffer
	geometryPass();

	compositePass();
	//return;
	//--------------------SHADOWS-----------------------
	//shadowPass(cam, numParticles);
	fluidBuffer.bindDraw();
	//--------------------CLOTH-------------------------
	//renderCloth(projection, mView, cam, numCloth, triangles);

	//--------------------WATER-------------------------
	renderWater(projection, mView, cam, numParticles - numCloth, numCloth);
	return;
	//--------------------FOAM--------------------------
	//renderFoam(projection, mView, cam, numDiffuse);

	//--------------------Final - WATER & DIFFUSE-------------------------
	glUseProgram(finalFS.program);
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, fluidBuffer.fluid);
	/*glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, fluidBuffer.foamIntensity);
	glActiveTexture(GL_TEXTURE2);
	glBindTexture(GL_TEXTURE_2D, fluidBuffer.foamRadiance);*/

	finalFS.setUniformi("fluidMap", 0);
	//finalFS.setUniformi("foamIntensityMap", 1);
	//finalFS.setUniformi("foamRadianceMap", 2);

	finalFS.setUniformv2f("screenSize", screenSize);

	glEnable(GL_DEPTH_TEST);
	glDepthMask(GL_TRUE);

	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	fsQuad.render();

	GLenum err = glGetError();
	if (err != 0) cout << err << endl;
}

void Renderer::renderPlane(buffers &buf) {
	glUseProgram(plane.program);
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, planeBuf.fbo);

	plane.setUniformmat4("mView", mView);
	plane.setUniformmat4("projection", projection);

	glDepthMask(GL_TRUE);
	glEnable(GL_DEPTH_TEST);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	
	glBindVertexArray(buf.vao);
	glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
}

void Renderer::geometryPass() {
	glUseProgram(geometry.program);
	gBuffer.bindDraw();
	gBuffer.setDrawBuffers();

	geometry.setUniformmat4("mView", mView);
	geometry.setUniformmat4("projection", projection);
	geometry.setUniformmat3("mNormal", normalMatrix);

	glDepthMask(GL_TRUE);
	glEnable(GL_DEPTH_TEST);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	for (auto &i : meshes) {
		geometry.setUniformv3f("diffuse", glm::vec3(0.5f));
		geometry.setUniformf("specular", 0.0f);
		i.render();
	}

	glDisable(GL_DEPTH_TEST);
	glDepthMask(GL_FALSE);
}

void Renderer::compositePass() {
	//Composition pass (directional light + light buffer)
	glUseProgram(finalPass.program);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	finalPass.setUniformmat4("inverseMView", glm::inverse(mView));
	finalPass.setUniformv3f("l", glm::vec3(mView * lightDir));

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, gBuffer.position);
	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, gBuffer.normal);
	glActiveTexture(GL_TEXTURE2);
	glBindTexture(GL_TEXTURE_2D, gBuffer.color);

	finalPass.setUniformi("positionMap", 0);
	finalPass.setUniformi("normalMap", 1);
	finalPass.setUniformi("colorMap", 2);

	glClear(GL_COLOR_BUFFER_BIT);

	fsQuad.render();
}

void Renderer::shadowPass(Camera &cam, int numParticles) {
	//glViewport(0, 0, 2048, 2048);
	glUseProgram(depth.program);
	dLightShadow.bindDraw();
	glDrawBuffer(GL_NONE);
	glReadBuffer(GL_NONE);

	depth.setUniformmat4("projection", dLightProjection);
	depth.setUniformmat4("mView", dLightMView);
	depth.setUniformf("pointRadius", radius);
	depth.setUniformf("pointScale", width / aspectRatio * (1.0f / tanf(cam.zoom * 0.5f)));

	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
	glDepthMask(GL_TRUE);
	glEnable(GL_DEPTH_TEST);

	glClear(GL_DEPTH_BUFFER_BIT);

	glBindVertexArray(positionVAO);
	glDrawArrays(GL_POINTS, 0, (GLsizei)numParticles);

	glDisable(GL_VERTEX_PROGRAM_POINT_SIZE);
	glDisable(GL_DEPTH_TEST);
	glDepthMask(GL_FALSE);

	dLightShadow.unbindDraw();
	//glViewport(0, 0, width, height);
}

void Renderer::renderWater(glm::mat4 &projection, glm::mat4 &mView, Camera &cam, int numParticles, int numCloth) {
	//----------------------Particle Depth----------------------
	glUseProgram(depth.program);
	fluidBuffer.setDrawDepth();	
	
	depth.setUniformmat4("mView", mView);
	depth.setUniformmat4("projection", projection);
	depth.setUniformf("pointRadius", radius);
	depth.setUniformf("pointScale", width / aspectRatio * (1.0f / tanf(cam.zoom * 0.5f)));

	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
	glDepthMask(GL_TRUE);
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);

	glClear(GL_DEPTH_BUFFER_BIT);
	
	glBindVertexArray(positionVAO);
	glDrawArrays(GL_POINTS, 0, (GLsizei)numParticles);

	//--------------------Particle Thickness-------------------------
	glUseProgram(thickness.program);
	fluidBuffer.setDrawThickness();

	thickness.setUniformmat4("mView", mView);
	thickness.setUniformmat4("projection", projection);
	thickness.setUniformf("pointRadius", radius * 2.0f);
	thickness.setUniformf("pointScale", width / aspectRatio * (1.0f / tanf(cam.zoom * 0.5f)));

	glDepthMask(GL_FALSE);
	glDisable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);
	glBlendFunc(GL_ONE, GL_ONE);
	glBlendEquation(GL_FUNC_ADD);

	glClear(GL_COLOR_BUFFER_BIT);

	glBindVertexArray(positionVAO);
	glDrawArrays(GL_POINTS, 0, (GLsizei)numParticles);

	glDisable(GL_VERTEX_PROGRAM_POINT_SIZE);
	glDisable(GL_BLEND);

	//--------------------Particle Blur-------------------------
	glUseProgram(blur.program);

	//Vertical blur
	fluidBuffer.setDrawVerticalBlur();
	
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, fluidBuffer.depth);
	
	blur.setUniformi("depthMap", 0);
	blur.setUniformmat4("projection", projection);
	blur.setUniformv2f("screenSize", screenSize);
	blur.setUniformv2f("blurDir", blurDirY);
	blur.setUniformf("filterRadius", filterRadius);
	blur.setUniformf("blurScale", 0.1f);
	//setFloat(blur, width / aspectRatio * (1.0f / (tanf(cam.zoom*0.5f))), "blurScale");

	glClear(GL_COLOR_BUFFER_BIT);

	fsQuad.render();

	//Horizontal blur
	fluidBuffer.setDrawHorizontalBlur();

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, fluidBuffer.blurV);

	blur.setUniformi("depthMap", 0);
	blur.setUniformv2f("blurDir", blurDirY);

	glClear(GL_COLOR_BUFFER_BIT);

	fsQuad.render();

	//--------------------Particle fluidFinal-------------------------
	glUseProgram(fluidFinal.program);
	//fluidBuffer.setDrawFluid();
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, fluidBuffer.blurH);
	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, fluidBuffer.thickness);
	glActiveTexture(GL_TEXTURE2);
	glBindTexture(GL_TEXTURE_2D, planeBuf.tex);
	//glActiveTexture(GL_TEXTURE3);
	//glBindTexture(GL_TEXTURE_2D, dLightShadow.depth);

	fluidFinal.setUniformi("depthMap", 0);
	fluidFinal.setUniformi("thicknessMap", 1);
	fluidFinal.setUniformi("sceneMap", 2);
	//fluidFinal.setUniformi("shadowMap", 3);

	fluidFinal.setUniformmat4("projection", projection);
	fluidFinal.setUniformmat4("inverseProjection", glm::inverse(projection));
	fluidFinal.setUniformmat4("mView", mView);
	fluidFinal.setUniformmat4("inverseMView", glm::inverse(mView));
	fluidFinal.setUniformv4f("color", color);
	fluidFinal.setUniformv2f("invTexScale", glm::vec2(1.0f / width, 1.0f / height));
	fluidFinal.setUniformmat4("shadowMapMVP", dLightProjection * dLightMView);
	fluidFinal.setUniformi("shadowMapWidth", 2048);
	fluidFinal.setUniformi("shadowMapHeight", 2048);

	glClear(GL_COLOR_BUFFER_BIT);

	fsQuad.render();
}

void Renderer::renderFoam(glm::mat4 &projection, glm::mat4 &mView, Camera &cam, int numDiffuse) {
	//--------------------Foam Depth-------------------------
	glUseProgram(foamDepth.program);
	fluidBuffer.setDrawFoamDepth();

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	foamDepth.setUniformmat4("mView", mView);
	foamDepth.setUniformmat4("projection", projection);
	foamDepth.setUniformf("pointRadius", foamRadius);
	foamDepth.setUniformf("pointScale", width / aspectRatio * (1.0f / tanf(cam.zoom * 0.5f)));
	foamDepth.setUniformf("fov", tanf(cam.zoom * 0.5f));

	glEnable(GL_DEPTH_TEST);
	//glDepthMask(GL_TRUE);
	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);

	glBindVertexArray(diffusePosVAO);
	glDrawArrays(GL_POINTS, 0, (GLsizei)numDiffuse);

	glDisable(GL_DEPTH_TEST);

	//--------------------Foam Thickness----------------------
	glUseProgram(foamThickness.program);
	fluidBuffer.setDrawFoamThickness();

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, fluidBuffer.foamDepth);
	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, fluidBuffer.depth);

	foamThickness.setUniformi("foamDepthMap", 0);
	foamThickness.setUniformi("fluidDepthMap", 1);

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

	glBindVertexArray(diffusePosVAO);
	glDrawArrays(GL_POINTS, 0, (GLsizei)numDiffuse);

	glDisable(GL_VERTEX_PROGRAM_POINT_SIZE);
	glDisable(GL_BLEND);

	//--------------------Foam Intensity----------------------
	glUseProgram(foamIntensity.program);
	fluidBuffer.setDrawFoamIntensity();

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, fluidBuffer.foamThickness);

	foamIntensity.setUniformi("thickness", 0);

	fsQuad.render();

	//--------------------Foam Radiance----------------------
	glUseProgram(foamRadiance.program);
	fluidBuffer.setDrawFoamRadiance();

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, fluidBuffer.foamDepth);
	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, fluidBuffer.depth);
	glActiveTexture(GL_TEXTURE2);
	glBindTexture(GL_TEXTURE_2D, fluidBuffer.foamDepth); //FIXME MISSING TEXTURE FOR FOAM'S NORMAL MAP (normal map doesn't make sense anyway, try changing the AO to a regular AO?)
	glActiveTexture(GL_TEXTURE3);
	glBindTexture(GL_TEXTURE_2D, fluidBuffer.foamIntensity);

	foamRadiance.setUniformi("foamDepthMap", 0);
	foamRadiance.setUniformi("fluidDepthMap", 1);
	foamRadiance.setUniformi("foamNormalHMap", 2);
	foamRadiance.setUniformi("foamIntensityMap", 3);

	foamRadiance.setUniformmat4("mView", mView);
	foamRadiance.setUniformmat4("projection", projection);
	foamRadiance.setUniformf("zNear", zNear);
	foamRadiance.setUniformf("zFar", zFar);

	fsQuad.render();
}

void Renderer::renderCloth(glm::mat4 &projection, glm::mat4 &mView, Camera &cam, int numCloth, std::vector<int> triangles) {
	glUseProgram(cloth.program);
	fluidBuffer.setDrawCloth();

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indicesVBO);

	cloth.setUniformmat4("mView", mView);
	cloth.setUniformmat4("projection", projection);
	
	glDrawElements(GL_TRIANGLES, (GLsizei)triangles.size(), GL_UNSIGNED_INT, 0);
}