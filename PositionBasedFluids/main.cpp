#define GLEW_DYNAMIC
#include <GL/glew.h>
#include "Renderer.h"
#include <GLFW/glfw3.h>

#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>

static const int width = 512;
static const int height = 512;

int main() {
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);

	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 4);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

	GLFWwindow* window = glfwCreateWindow(width, height, "Position Based Fluids", nullptr, nullptr);
	glfwMakeContextCurrent(window);
	
	glewExperimental = GL_TRUE;
	glewInit();
	glGetError();
	
	// Define the viewport dimensions
	glViewport(0, 0, width, height);
	
	Renderer render = Renderer();
	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
	
	while (!glfwWindowShouldClose(window)) {
		// Check and call events
		glfwPollEvents();

		// Render
		// Clear the colorbuffer
		glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		render.run();

		// Swap the buffers
		glfwSwapBuffers(window);
	}

	glfwTerminate();

	return 0;
}