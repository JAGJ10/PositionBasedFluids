#version 400 core

in vec3 pos;
in float radius;

uniform mat4 mView;
uniform mat4 projection;
uniform vec2 screenSize;

out float thickness;

void main() {
	//calculate normal
	vec3 normal;
	normal.xy = gl_PointCoord * 2.0 - 1.0;
	float r2 = dot(normal.xy, normal.xy);
	
	if (r2 > 1.0f) {
		discard;
	}
	
	float nz = sqrt(1 - r2);
	
    thickness = nz * 1.25 * 2.0f * exp(-r2 * 2.0f);
    //thickness = 1 - r2;
    //thickness = 1;
    //float dist = length(gl_PointCoord.xy-vec2(0.5f,0.5f));
	//float sigma = 3.0f;
	//float mu = 0.0f;
	//thickness = 0.02f * exp(-(dist-mu)*(dist-mu)/(2*sigma));
}