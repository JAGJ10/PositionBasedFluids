#version 400 core

in vec4 vertexPos;

uniform mat4 projection;
uniform mat4 mView;
uniform vec2 screenSize;
uniform float pointRadius;
uniform float pointScale;

out vec3 pos;
out float lifetime;
//out float type;

void main() {
	float ri = pointRadius;

	float trl = vertexPos.w;

//	if (trl < 1500) {
//		trl -= 1000;
//		type = 1;
//	} else if (trl < 2500) {
//		trl -= 2000;
//		type = 2;
//	}

	ri /= ((int(trl) % 5) + 1);

	vec4 viewPos = mView * vec4(vertexPos.xyz, 1.0);
    gl_Position = projection * viewPos;
	pos = viewPos.xyz;
	gl_PointSize = pointScale * (ri / gl_Position.w);

	lifetime = trl;
}