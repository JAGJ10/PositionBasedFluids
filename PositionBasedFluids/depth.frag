#version 400 core

in vec3 pos;

uniform mat4 mView;
uniform mat4 projection;
uniform vec2 screenSize;

void main() {
	//calculate normal
	vec3 normal;
	normal.xy = gl_PointCoord * 2.0 - 1.0;
	float r2 = dot(normal.xy, normal.xy);
	
	if (r2 > 1.0) {
		discard;
	}
	
	normal.z = sqrt(1.0 - r2);

	//calculate depth
	vec4 pixelPos = vec4(pos + normal * 1.25f, 1.0);
	vec4 clipSpacePos = projection * pixelPos;
	
	gl_FragDepth = clipSpacePos.z / clipSpacePos.w * 0.5f + 0.5f;
}