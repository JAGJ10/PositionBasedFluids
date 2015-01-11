#version 400 core

in vec3 vertexPos;

uniform mat4 projection;
uniform mat4 mView;
uniform vec2 screenSize;

out vec3 pos;
out float radius;

void main() {
	vec4 viewPos = mView * vec4(vertexPos, 1.0);
    float dist = length(viewPos);
    pos = viewPos.xyz;
    gl_Position = projection * viewPos;
    //radius = 10 * screenSize.y / (8.0 * gl_Position.z);
    //radius = 5.0f / (-pos.z * 4.0f * (1.0f / screenSize.y));
    radius = 1.25 * (600.0 / dist);
    
    gl_PointSize = radius;
}