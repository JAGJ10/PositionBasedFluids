#version 420 core

in vec3 position;

uniform mat4 mView;
uniform mat4 projection;

out vec3 fragPos;

void main() {
	gl_Position = projection * mView * vec4(position, 1.0);
	fragPos = (mView * vec4(position, 1.0)).xyz;
}