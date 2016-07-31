#version 450 core

in vec2 coord;

uniform sampler2D depthMap;
uniform vec2 screenSize;
uniform mat4 projection;
uniform vec2 blurDir;
uniform float filterRadius;
uniform float blurScale;

out float fragColor;
float sqr(float x) { return x*x; }
const float blurDepthFalloff = 5.5f;

void main() {
	float depth = texture(depthMap, coord).x;

	//float blurDepthFalloff = 5.5;
	float maxBlurRadius = 5.0;
	//discontinuities between different tap counts are visible. to avoid this we 
	//use fractional contributions between #taps = ceil(radius) and floor(radius) 
	float radius = min(maxBlurRadius, blurScale * (0.05 / -depth));
	float radiusInv = 1.0/radius;
	float taps = ceil(radius);
	float frac = taps - radius;

	float sum = 0.0;
    float wsum = 0.0;
	float count = 0.0;

    for(float y=-taps; y <= taps; y += 1.0) {
        for(float x=-taps; x <= taps; x += 1.0) {
			vec2 offset = vec2(x, y);

            float s = texture(depthMap, coord + offset).x;

			if (s < -10000.0*0.5)
				continue;

            // spatial domain
            float r1 = length(vec2(x, y))*radiusInv;
			float w = exp(-(r1*r1));

            // range domain (based on depth difference)
            float r2 = (s - depth) * blurDepthFalloff;
            float g = exp(-(r2*r2));

			//fractional radius contributions
			float wBoundary = step(radius, max(abs(x), abs(y)));
			float wFrac = 1.0 - wBoundary*frac;

			sum += s * w * g * wFrac;
			wsum += w * g * wFrac;
			count += g * wFrac;
        }
    }

    if (wsum > 0.0) {
        sum /= wsum;
    }

	float blend = count/sqr(2.0*radius+1.0);
	fragColor = mix(depth, sum, blend);
}