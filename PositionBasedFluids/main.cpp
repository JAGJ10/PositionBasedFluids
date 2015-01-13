#define GLEW_DYNAMIC
#include <GL/glew.h>
#include "Renderer.h"
#include <GLFW/glfw3.h>

#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>

static const int width = 512;
static const int height = 512;
GLfloat lastX = (width / 2), lastY = (height / 2);
double deltaTime = 0.0f;
double lastFrame = 0.0f;
Camera cam = Camera();

//void keyHandler(GLFWwindow* window, int key, int scancode, int action, int mode);
//void mouseMovementHandler(GLFWwindow* window, double xpos, double ypos);
void handleInput(GLFWwindow* window);

int main() {
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);

	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 4);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

	GLFWwindow* window = glfwCreateWindow(width, height, "Position Based Fluids", nullptr, nullptr);
	glfwMakeContextCurrent(window);

	//Set callbacks for keyboard and mouse
	//glfwSetKeyCallback(window, keyHandler);
	//glfwSetCursorPosCallback(window, mouseMovementHandler);
	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);
	
	glewExperimental = GL_TRUE;
	glewInit();
	glGetError();
	
	// Define the viewport dimensions
	glViewport(0, 0, width, height);
	
	Renderer render = Renderer();
	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
	
	while (!glfwWindowShouldClose(window)) {
		//Set frame times
		double currentFrame = glfwGetTime();
		deltaTime = currentFrame - lastFrame;
		lastFrame = currentFrame;

		// Check and call events
		glfwPollEvents();
		handleInput(window);

		render.run(cam);

		// Swap the buffers
		glfwSwapBuffers(window);

		glfwSetCursorPos(window, width / 2, height / 2);
	}

	glfwTerminate();

	return 0;
}

void handleInput(GLFWwindow* window) {
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, GL_TRUE);

	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
		cam.wasdMovement(FORWARD, deltaTime);

	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
		cam.wasdMovement(BACKWARD, deltaTime);

	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
		cam.wasdMovement(RIGHT, deltaTime);

	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
		cam.wasdMovement(LEFT, deltaTime);

	if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
		cam.wasdMovement(UP, deltaTime);

	double xpos, ypos;
	glfwGetCursorPos(window, &xpos, &ypos);
	cam.mouseMovement((width/2) -  xpos, (height/2) - ypos, deltaTime);
}