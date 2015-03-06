#version 400 core
in vec3 coord;

uniform samplerCube skybox;

out vec4 color;

void main() {    
    color = texture(skybox, coord);
}