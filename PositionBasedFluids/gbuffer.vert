#version 420 core

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;

uniform mat4 mView;
uniform mat4 projection;
uniform mat3 mNormal;

out vec3 fragPos;
out vec3 fragNormal;

void main() {
    gl_Position = projection * mView * vec4(position, 1.0);
	fragPos = (mView * vec4(position, 1.0)).xyz;
	fragNormal = normalize(mNormal * normal);
}