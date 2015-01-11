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

Renderer::Renderer() {}

Renderer::~Renderer() {}

void Renderer::run() {
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

	GLFWwindow* window = glfwCreateWindow(512, 512, "Position Based Fluids", nullptr, nullptr);
	glfwMakeContextCurrent(window);

	glewExperimental = GL_TRUE;
	glewInit();

	// Define the viewport dimensions
	glViewport(0, 0, 512, 512);

	Shader simple("E:/Main Data/My Documents/GitHub/PositionBasedFluids/PositionBasedFluids/simple.vert", "E:/Main Data/My Documents/GitHub/PositionBasedFluids/PositionBasedFluids/simple.frag");
	GLfloat vertices[] = {
		// Positions	// Colors
		0.5f, -0.5f, 1.0f, 0.0f, 0.0f,	// Bottom Right
		-0.5f, -0.5f, 0.0f, 1.0f, 0.0f,	// Bottom Left
		0.0f, 0.5f, 0.0f, 0.0f, 1.0f	// Top 
	};

	GLuint VBO, VAO;
	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO);
	// Bind our Vertex Array Object first, then bind and set our buffers and pointers.
	glBindVertexArray(VAO);

	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

	// Position attribute
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat), (GLvoid*)0);
	glEnableVertexAttribArray(0);
	// Color attribute
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat), (GLvoid*)(2 * sizeof(GLfloat)));
	glEnableVertexAttribArray(1);

	glBindVertexArray(0);

	while (!glfwWindowShouldClose(window)) {
		// Check and call events
		glfwPollEvents();

		// Render
		// Clear the colorbuffer
		glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);
		simple.Use();
		glBindVertexArray(VAO);
		glDrawArrays(GL_TRIANGLES, 0, 3);
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