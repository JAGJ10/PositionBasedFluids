#version 400 core

in vec2 coord;

uniform sampler2D depthMap;
uniform vec2 screenSize;
uniform mat4 projection;
uniform vec2 blurDir;
uniform float filterRadius;
uniform float blurScale;

const float blurDepthFalloff = 4;

void main() {
	float depth = texture(depthMap, coord).x;
	if (depth == 1.0f) {
		gl_FragDepth = 1;
		return;
	}

	if (depth == 0.0f) {
		gl_FragDepth = 0;
		return;
	}
	
	float sum = 0.0f;
	float wsum = 0.0f;
	
	for (float x = -filterRadius; x <= filterRadius; x += 1.0f) {
		float s = texture(depthMap, coord + x*blurDir).x;

		float r = x * blurScale;
		float w = exp(-r*r);
		
		float r2 = (s - depth) * blurDepthFalloff;
		float g = exp(-r2*r2);
		
		sum += s * w * g;
		wsum += w * g;
	}

	if (wsum > 0.0f) {
		sum /= wsum;
	}
	
	gl_FragDepth = sum;
}