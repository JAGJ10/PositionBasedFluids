#define GLEW_STATIC
#include <GL/glew.h>
#include <SFML/Window.hpp>
#include <SFML/OpenGL.hpp>
#include "common.h"
#include "ParticleSystem.h"

#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>

int main() {
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
	ParticleSystem system = ParticleSystem(0.05f);

	sf::Window window(sf::VideoMode(1024, 512), "Position Based Fluids");
	glewExperimental = GL_TRUE;
	glewInit();

	bool running = true;
	while (running) {
		sf::Event event;
		while (window.pollEvent(event)) {
			if (event.type == sf::Event::Closed)
				running = false;
			else if (event.type == sf::Event::Resized)
				glViewport(0, 0, event.size.width, event.size.height);
		}
		
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		window.display();
	}

	return 0;
}