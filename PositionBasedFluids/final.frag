#version 450 core

in vec2 coord;

uniform sampler2D fluidMap;
uniform sampler2D foamIntensityMap;
uniform sampler2D foamRadianceMap;
uniform sampler2D foamDepthMap;
uniform vec2 screenSize;

uniform mat4 projection;

out vec4 fragColor;

const float PI = 3.14159265358979323846f;

void main() {
	float foamIntensity = texture(foamIntensityMap, coord).x;
	if (foamIntensity == 0) discard;
	float foamRadiance = texture(foamRadianceMap, coord).x;
	float hPass = texture(foamRadianceMap, coord).y;
	float i = texture(foamRadianceMap, coord).z;
	//vec4 fluid = texture(fluidMap, coord).xyzw;
	float foamDepth = texture(foamDepthMap, coord).x;

	float sum = 0;
	float totalWeight = 0;
	hPass *= 3 / 2;

	for (float x = -hPass; x < hPass; x+=1) {
		for (float y = -hPass; y < hPass; y+=1) {
			vec2 cxy = vec2(x / screenSize.x, y / screenSize.y);
			float weight = exp(-pow(length(vec2(x, y)), 2) / (pow(hPass, 2) * 2)) * (1 / (2 * PI * pow(hPass, 2)));
			sum += texture(foamRadianceMap, coord + cxy).x * weight;
			totalWeight += weight;
		}
	}
	sum /= totalWeight;

	vec4 squiggly = vec4(clamp(sum * (vec3(1, 1, 1) - vec3(0, 0.2, 0.6)), 0, 1), 0);
	fragColor = (1 - foamIntensity) + (foamIntensity * (0.8 - squiggly));
	//fragColor = fluid;
	vec4 clipPos = projection*vec4(0.0, 0.0, foamDepth, 1.0);
	clipPos.z /= clipPos.w;
	gl_FragDepth = foamDepth;//clipPos.z*0.5+0.5;
}