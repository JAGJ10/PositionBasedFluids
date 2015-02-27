#version 400 core

in vec4 vertexPos;

uniform mat4 projection;
uniform mat4 mView;
uniform float pointRadius;
uniform float pointScale;
uniform float fov;

out vec3 pos;
out float hfrag;

void main() {
	vec4 viewPos = mView * vec4(vertexPos.xyz, 1.0);
    gl_Position = projection * viewPos;
	pos = viewPos.xyz;
	float ri = pointRadius / ((int(vertexPos.w) % 5) + 1);
	gl_PointSize = pointScale * (ri / gl_Position.w);
	hfrag = ri / (fov * abs(gl_Position.w));
}