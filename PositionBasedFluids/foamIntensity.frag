#version 400 core

in vec2 coord;

uniform sampler2D thickness;

out float intensity;

void main() {
	float p = texture(thickness, coord).x;

	float pexp = pow(p, 1.25);
	intensity = pexp / (2 + pexp);
}