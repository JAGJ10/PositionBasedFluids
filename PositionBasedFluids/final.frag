#version 400 core

in vec2 coord;

uniform sampler2D fluidMap;
uniform sampler2D foamMap;

out vec4 fragColor;

void main() {
	float foam = texture(foamMap, coord).x;

	vec4 fluid = texture(fluidMap, coord).xyzw;

	if (foam >= 0.4) {
		fragColor = vec4(foam);
		return;
	}

	fragColor = vec4(fluid);
}