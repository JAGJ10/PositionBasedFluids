#version 450 core

layout(location = 0) in vec3 fragPos;
layout(location = 1) in vec3 fragNormal;

uniform vec3 diffuse;
uniform float specular;

layout(location = 0) out vec4 position;
layout(location = 1) out vec4 normal;
layout(location = 2) out vec4 color;

void main() {
	position = vec4(fragPos, 1.0);
	normal = vec4(fragNormal, specular);
	//normal = vec4(fragNormal * 0.5 + 0.5, specular);
	color = vec4(diffuse, 1.0);
}