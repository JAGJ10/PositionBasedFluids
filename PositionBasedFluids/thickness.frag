#version 400 core

in vec3 pos;

uniform mat4 mView;
uniform mat4 projection;

out float thickness;

void main() {
	//calculate normal
	vec3 normal;
	normal.xy = gl_PointCoord * 2.0 - 1.0;
	float r2 = dot(normal.xy, normal.xy);
	
	if (r2 > 1.0f) {
		discard;
	}
	
	normal.z = sqrt(1 - r2);
	
	thickness = normal.z * 0.005f;
}