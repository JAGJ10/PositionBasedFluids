#version 420 core

in vec3 fragPos;

out vec4 fragColor;

void main() {
	fragColor = vec4(fragPos, 1.0);
}