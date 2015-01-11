#define GLEW_DYNAMIC
#include <GL/glew.h>
/*#include <SFML/Window.hpp>
#include <SFML/OpenGL.hpp>*/
#include "Renderer.h"
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

Renderer::Renderer() {}

Renderer::~Renderer() {}

void Renderer::run() {
	/*sf::Window window(sf::VideoMode(1024, 512), "Position Based Fluids");
	glewExperimental = GL_TRUE;
	glewInit();

	bool running = true;
	while (running) {
		glClearColor(1, 1, 1, 1);
		sf::Event event;
		while (window.pollEvent(event)) {
			if (event.type == sf::Event::Closed)
				running = false;
			else if (event.type == sf::Event::Resized)
				glViewport(0, 0, event.size.width, event.size.height);
		}

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		window.display();
	}*/

	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

	GLFWwindow* window = glfwCreateWindow(800, 600, "LearnOpenGL", nullptr, nullptr);
	glfwMakeContextCurrent(window);

	glewExperimental = GL_TRUE;
	glewInit();

	while (!glfwWindowShouldClose(window))
	{
		// Check and call events
		glfwPollEvents();

		// Render
		// Clear the colorbuffer
		glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);

		// Swap the buffers
		glfwSwapBuffers(window);
	}

	glfwTerminate();
}

void Renderer::initShaders() {

}

void Renderer::initFramebuffers() {

}