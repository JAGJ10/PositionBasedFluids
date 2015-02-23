#version 400 core

in vec2 coord;

uniform sampler2D fluidMap;
uniform sampler2D foamMap;

out vec4 fragColor;

void main() {
	vec3 foam = texture(foamMap, coord).xyz;

	vec4 fluid = texture(fluidMap, coord).xyzw;

	if (foam.y == 0) {
		fragColor = vec4(foam, 1);
		return;
	}

	fragColor = fluid;
}