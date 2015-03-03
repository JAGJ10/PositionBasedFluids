#version 400 core

in vec2 coord;

uniform sampler2D fluidMap;
uniform sampler2D foamIntensityMap;
uniform sampler2D foamRadianceMap;

out vec4 fragColor;

void main() {
	float foamIntensity = texture(foamIntensityMap, coord).x;
	float foamRadiance = texture(foamRadianceMap, coord).x;
	float hPass = texture(foamRadianceMap, coord).y;
	float i = texture(foamRadianceMap, coord).z;
	vec4 fluid = texture(fluidMap, coord).xyzw;

	float squiggly = clamp(foamRadiance * (vec3(1, 1, 1) - vec3(0, 0, 0.2)), 0, 1);

	//float ifinal = clamp(foamIntensity, 0, 1);

	fragColor = (1 - foamIntensity) * fluid + (foamIntensity * (1 - squiggly));
}