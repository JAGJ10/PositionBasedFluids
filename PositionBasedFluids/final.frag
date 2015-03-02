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
	vec4 fluid = texture(fluidMap, coord).xyzw;

	float sum = 0.0;
	sum += texture(foamRadianceMap, vec2(coord.x - 4.0*hPass, coord.y)).x * 0.05;
	sum += texture(foamRadianceMap, vec2(coord.x - 3.0*hPass, coord.y)).x * 0.09;
	sum += texture(foamRadianceMap, vec2(coord.x - 2.0*hPass, coord.y)).x * 0.12;
	sum += texture(foamRadianceMap, vec2(coord.x - hPass, coord.y)).x * 0.15;
	sum += texture(foamRadianceMap, vec2(coord.x, coord.y)).x * 0.16;
	sum += texture(foamRadianceMap, vec2(coord.x + hPass, coord.y)).x * 0.15;
	sum += texture(foamRadianceMap, vec2(coord.x + 2.0*hPass, coord.y)).x * 0.12;
	sum += texture(foamRadianceMap, vec2(coord.x + 3.0*hPass, coord.y)).x * 0.09;
	sum += texture(foamRadianceMap, vec2(coord.x + 4.0*hPass, coord.y)).x * 0.05;

	float squiggly = clamp(sum * (vec3(1, 1, 1) - vec3(0, 0, 0.2)), 0, 1);

	//float ifinal = clamp(foamIntensity, 0, 1);

	fragColor = (1 - foamIntensity) * fluid + (foamIntensity * (.8 - squiggly));
}