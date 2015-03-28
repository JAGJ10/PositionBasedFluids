#ifndef SCENE_H
#define SCENE_H

#include "common.h"

class Scene {
public:
	Scene(std::string name) : name(name) {}
	virtual void Init() = 0;
	
private:
	std::string name;

};

class DamBreak : Scene {
public:
	
};

class FluidCloth : Scene {
public:

};

class Lighthouse : Scene {
public:

};

#endif