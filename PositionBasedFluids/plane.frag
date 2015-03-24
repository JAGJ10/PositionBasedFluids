#version 400 core
in vec3 coord;

out vec3 color;

void main() {
	vec3 col1 = vec3(0.8);
	vec3 col2 = vec3(0.9);
    color = mix(col1, col2, 0.5 * mod(floor(coord.x) + floor(coord.y) + floor(coord.z), 2));
}