#version 400 core

in vec2 coord;

uniform sampler2D foamDepthMap;
uniform sampler2D fluidDepthMap;
uniform sampler2D foamNormalHMap;

out float squiggly;

const float PI = 3.14159265358979323846f;

float rand(vec2 co) {
    return fract(sin(dot(co.xy, vec2(12.9898,78.233))) * 43758.5453);
}

void main() {
	float hfrag = texture(foamNormalHMap, coord).w;

	float foamDepth = texture(foamDepthMap, coord).x;
	float fluidDepth = texture(fluidDepthMap, coord).x;

	float omega = 0;
	float omegaBottom = 0;

	for (float p = 0; p < 3; p+=1) {
		float hpass = hfrag * (1 + 7 * p);
		float v = clamp(0.75 * PI * pow(hpass, 3) * 0.5, 16, 512);

		for (float i = 0; i < v; i+=1) {
			vec2 s = vec2(rand(vec2(10 * v, 10 * v)), rand(vec2(20 * v, 20 * v)));
			if (length(s) > 1) continue;

			vec2 sampleCoord = coord + (s * hpass);
			float sampleDepth = texture(foamDepthMap, sampleCoord).x;

			float lambda = pow(1 - length(s), 2);
			float delta = pow(max(1 - (abs(foamDepth - sampleDepth) / 5), 0), 2);

			float k = ((sampleDepth > foamDepth || sampleDepth > fluidDepth) && (delta > 0.0 && delta < 1.0)) ? 1.0 : 0.0;
	
			omega += lambda * delta * k;
			omegaBottom += lambda;
		}
	}

	omega /= omegaBottom;
	squiggly = clamp(pow(omega * 1, 1.5) + -0.05, 0, 1);
}