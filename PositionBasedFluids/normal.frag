#version 400 core

in vec2 coord;

uniform sampler2D depthMap;
uniform mat4 mView;
uniform mat4 projection;
uniform float zNear;
uniform float zFar;

out vec4 normal;

float linearizeDepth(float depth) {	
	return (2.0 * zNear) / (zFar + zNear - depth * (zFar - zNear));
}

vec3 uvToEye(vec2 p, float z) {
	vec2 pos = p * 2.0f - 1.0f;
	vec4 clipPos = vec4(pos, z, 1.0f);
	vec4 viewPos = inverse(projection) * clipPos;
	return viewPos.xyz / viewPos.w;
}

void main() {
	float depth = texture(depthMap, coord).x;
	//depth = linearizeDepth(depth);
	if (depth >= .99f) {
		//discard;
	}
	
	vec3 pos = uvToEye(coord, depth);
	
	normal = vec4(normalize(cross(dFdx(pos.xyz), dFdy(pos.xyz))), 1.0f);
}