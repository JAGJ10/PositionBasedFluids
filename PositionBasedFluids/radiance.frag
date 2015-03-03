#version 400 core

in vec2 coord;

uniform vec4 mView;
uniform sampler2D foamDepthMap;
uniform sampler2D fluidDepthMap;
uniform sampler2D foamNormalHMap;
uniform sampler2D foamIntensityMap;

out vec3 squiggly;

const float PI = 3.14159265358979323846f;
const vec3 lightDir = vec3(0, 1, 0);

float rand(vec2 co) {
    return fract(sin(dot(co.xy, vec2(12.9898,78.233))) * 43758.5453);
}

void main() {
	float hfrag = texture(foamNormalHMap, coord).w;
	float foamDepth = texture(foamDepthMap, coord).x;
	float fluidDepth = texture(fluidDepthMap, coord).x;

	float omega = 0;
	float omegaBottom = 0;
	float hpass;

	vec3 nfrag = texture(foamNormalHMap, coord).xyz;
	vec3 l = (mView * vec4(lightDir, 0.0)).xyz;
	
	float irr = 0;
	float irrBottom = 0;

	for (float p = 0; p < 3; p+=1) {
		hpass = hfrag * (1 + 7 * p);
		float v = clamp(0.75 * PI * pow(hpass, 3) * 0.5, 16, 512);
		
		for (float i = 0; i < v; i+=1) {
			vec2 s = vec2(rand(vec2(10 * v, 10 * v)), rand(vec2(20 * v, 20 * v)));
			if (length(s) > 1) continue;
			s.x /= 1024;
			s.y /= 512;
			
			float sampleFoamDepth = texture(foamDepthMap, coord + s*hpass).x;
			float sampleFluidDepth = texture(fluidDepthMap, coord + s*hpass).x;
			float sampleIntensity = texture(foamIntensityMap, coord + s*hpass).x;

			float lambda = pow(1 - length(s), 2);
			float delta = pow(max(1 - (abs(foamDepth - sampleDepth) / 5), 0), 2);

			float k = ((sampleFoamDepth > foamDepth || sampleFluidDepth > fluidDepth) && (delta > 0.0 && delta < 1.0)) ? 1.0 : 0.0;
	
			omega += lambda * delta * k * sampleIntensity;
			omegaBottom += lambda;
		}
	}

	omega /= omegaBottom;
	squiggly.x = clamp(pow(omega * 1, 1.5) + -0.05, 0, 1);
	squiggly.y = hpass;
	squiggly.z = irr / irrBottom;
	
}