#version 400 core

in vec2 coord;

uniform sampler2D fluidMap;
uniform sampler2D foamIntensityMap;
uniform sampler2D foamRadianceMap;
uniform sampler2D clothMap;
uniform vec2 screenSize;

out vec4 fragColor;

const float PI = 3.14159265358979323846f;

void main() {
	vec4 cloth = texture(clothMap, coord);
	if (cloth.x > 0) {
		fragColor = cloth;
		return;
	}
	float foamIntensity = texture(foamIntensityMap, coord).x;
	float foamRadiance = texture(foamRadianceMap, coord).x;
	float hPass = texture(foamRadianceMap, coord).y;
	float i = texture(foamRadianceMap, coord).z;
	vec4 fluid = texture(fluidMap, coord).xyzw;

	float sum = 0;
	hPass *= 3 / 2;

	for (float x = -hPass; x < hPass; x+=1) {
		for (float y = -hPass; y < hPass; y+=1) {
			vec2 cxy = vec2(x / screenSize.x, y / screenSize.y);
			sum += texture(foamRadianceMap, coord + cxy).x * exp(-pow(length(vec2(x, y)), 2) / (pow(hPass, 2) * 2)) * (1 / (2 * PI * pow(hPass, 2)));
		}
	}

	float squiggly = clamp(sum * (vec3(1, 1, 1) - vec3(0, 0.2, 0.6)), 0, 1);
	fragColor = (1 - foamIntensity) * fluid + (foamIntensity * (1 - squiggly));
}