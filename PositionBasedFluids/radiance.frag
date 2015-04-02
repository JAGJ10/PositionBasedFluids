#version 400 core

in vec2 coord;

uniform mat4 mView;
uniform mat4 projection;
uniform mat4 inverseProjection;
uniform float zNear;
uniform float zFar;
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
	
	float irr = abs(dot(l, nfrag));

	vec4 eyeCoord = vec4(((vec3(coord, foamDepth) * 2.0f) - 1.0f), 1);
	eyeCoord = inverseProjection * eyeCoord;
	eyeCoord /= eyeCoord.w;

	for (float p = 0; p < 1; p+=1) {
		hpass = hfrag * (1 + 7 * p);
		float v = clamp(0.75 * PI * pow(hpass, 3) * 0.5, 16, 256);
		
		for (float i = 0; i < v; i+=1) {
			vec3 s = vec3(rand(vec2(10 * v, 10 * v)), rand(vec2(20 * v, 20 * v)), rand(vec2(30 * v, 30 * v)));
			s = (s * 2.0f) - 1.0f;
			if (length(s) > 1) continue;
			float lambda = pow(1 - length(s), 2);
			s += eyeCoord.xyz;
			vec4 sproj = projection * vec4(s, 1);
			sproj /= sproj.w;
			sproj = sproj * 0.5f + 0.5f;
			
			float sampleFoamDepth = texture(foamDepthMap, sproj.xy).x;
			float sampleFluidDepth = texture(fluidDepthMap, sproj.xy).x;
			float sampleIntensity = texture(foamIntensityMap, sproj.xy).x;

			float delta = pow(max(1 - (abs(sampleFoamDepth - sproj.z) / 5), 0), 2);

			float k = ((sproj.z > sampleFoamDepth || sproj.z > sampleFluidDepth) && (delta > 0.0 && delta < 1.0)) ? 1.0 : 0.0;
	
			omega += lambda * delta * k * sampleIntensity;
			omegaBottom += lambda;
		}
	}

	omega /= omegaBottom;
	squiggly.x = clamp(pow(omega * 1, 1.5) + -0.05, 0, 1);
	squiggly.y = hpass;
	squiggly.z = irr;
}