#define GLEW_DYNAMIC
#include <GL/glew.h>
#include "Renderer.h"
#include "Shader.hpp"
#include <GLFW/glfw3.h>

using namespace std;

static const int width = 1024;
static const int height = 512;
static const float zFar = 200;
static const float zNear = 1;
static const float aspectRatio;
static const glm::vec2 screenSize;
static const glm::vec3 color;
static const int useThickness;
static const int filterRadius;

Renderer::Renderer() :
	depth{ Shader("depth.vert", "depth.frag") },
	normals{ Shader("normal.vert", "normal.frag") },
	blur{ Shader("blur.vert", "blur.frag") },
	thickness{ Shader("depth.vert", "thickness.frag") },
	composite{ Shader("composite.vert", "composite.frag") }
	{}

Renderer::~Renderer() {}

void Renderer::run() {
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 4);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

	GLFWwindow* window = glfwCreateWindow(512, 512, "Position Based Fluids", nullptr, nullptr);
	glfwMakeContextCurrent(window);

	glewExperimental = GL_TRUE;
	glewInit();

	// Define the viewport dimensions
	glViewport(0, 0, 512, 512);

	Shader simple("simple.vert", "simple.frag");

	while (!glfwWindowShouldClose(window)) {
		// Check and call events
		glfwPollEvents();

		// Render
		// Clear the colorbuffer
		glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);
		glUseProgram(simple.program);
		simple.shaderVAOQuad();
		glBindVertexArray(simple.vao);
		glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
		glBindVertexArray(0);

		// Swap the buffers
		glfwSwapBuffers(window);
	}
	
	glfwTerminate();
}

void Renderer::initShaders() {

}

void Renderer::initFramebuffers() {

}