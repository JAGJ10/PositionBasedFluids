#version 450 core

in vec3 pos;

uniform mat4 mView;
uniform mat4 projection;
uniform float pointRadius;

layout(location = 0) out vec4 position;
layout(location = 1) out vec4 normal;
layout(location = 2) out vec4 color;

const vec3 lightDir = vec3(1, 1, 1);
const vec3 diffuseCol = vec3(0.0f, 0.0f, 0.8f);

void main() {
	//calculate normal
	vec3 n;
	n.xy = gl_PointCoord * 2.0 - 1.0;
	float r2 = dot(n.xy, n.xy);
	
	if (r2 > 1.0) {
		discard;
	}
	
	n.z = sqrt(1.0 - r2);

	//calculate depth
	vec4 pixelPos = vec4(pos + n * pointRadius, 1.0);
	vec4 clipSpacePos = projection * pixelPos;
	
	gl_FragDepth = (clipSpacePos.z / clipSpacePos.w) * 0.5f + 0.5f;
	//fragColor = (clipSpacePos.z / clipSpacePos.w) * 0.5f + 0.5f;
	//fragColor = pixelPos.z;

	float diffuse = max(0.0, dot(n, lightDir));
	color = vec4(diffuseCol, 1.0f);
	normal = vec4(n, 0.0f);
	position = vec4(pos, 1.0f);
}