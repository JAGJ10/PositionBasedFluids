#version 420 core

in vec2 vertexPos;

out vec2 coord;

void main() {
    coord = vertexPos * 0.5f + 0.5f;
    gl_Position = vec4(vertexPos, 0.0, 1.0);
}