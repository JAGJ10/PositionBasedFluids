#ifndef CAMERA_H
#define CAMERA_H

#include "common.h"
#include <GL/glew.h>

enum Movement {
	FORWARD,
	BACKWARD,
	LEFT,
	RIGHT,
	UP,
	DOWN
};

class Camera {
public:
	glm::vec3 eye;
	glm::vec3 front;
	glm::vec3 up;
	glm::vec3 right;
	
	GLfloat yaw;
	GLfloat pitch;

	float speed;
	float mouseSens;
	GLfloat zoom;

	Camera() : eye(glm::vec3(-5.0f, 9.0f, 13.0f)),
		front(glm::normalize(glm::vec3(cos(glm::radians(-90.0f)), 0.0f, sin(glm::radians(-90.0f))))),
		up(glm::vec3(0.0f, 1.0f, 0.0f)),
		right(glm::cross(up, (eye - front))),
		yaw(-90.0f),
		pitch(0.0f),
		speed(1.0f),
		mouseSens(4.0f),
		zoom(45.0f)
	{}

	glm::mat4 getMView() {
		return glm::lookAt(eye, glm::vec3(4.0f, 1.0f, 2.0f), up);
		//return glm::lookAt(eye, eye + front, up);
	}

	void wasdMovement(Movement dir, float deltaTime) {
		float velocity = speed * deltaTime;
		switch (dir) {
		case FORWARD:
			eye += front * velocity;
			break;
		case BACKWARD:
			eye -= front * velocity;
			break;
		case LEFT:
			eye -= glm::normalize(glm::cross(front, up)) * velocity;
			break;
		case RIGHT:
			eye += glm::normalize(glm::cross(front, up)) * velocity;
			break;
		case UP:
			eye.y += 2 * (float)velocity;
			break;
		case DOWN:
			eye.y -= 2 * (float)velocity;
			break;
		}
	}

	void mouseMovement(float xoffset, float yoffset, float deltaTime) {
		yaw += (GLfloat)(mouseSens * deltaTime * xoffset);
		pitch += (GLfloat)(mouseSens * deltaTime * yoffset);
		
		if (pitch > 89.0f) pitch = 89.0f;
		if (pitch < -89.0f)	pitch = -89.0f;

		front.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
		front.y = sin(glm::radians(pitch));
		front.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
		front = glm::normalize(front);
	}
};

#endif