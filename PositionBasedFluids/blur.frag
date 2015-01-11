#version 400 core

in vec2 coord;

uniform sampler2D depthMap;
uniform vec2 screenSize;
uniform mat4 projection;
uniform vec2 blurDir;
uniform int useThickness;
uniform float filterRadius;

const float blurScale = .1;
const float blurDepthFalloff = 2;

void main() {
	float depth = texture(depthMap, coord).x;
	if (depth == 1f) {
		discard;
	}
	
	if (useThickness == 1) {
		depth /= 20;
	}
	
	float sum = 0.0f;
	float wsum = 0.0f;
	
	for (float x = -filterRadius; x <= filterRadius; x += 1.0f) {
		float s = texture(depthMap, coord + x*blurDir).x;
		if (useThickness == 1) {
			s /= 20;
		}

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