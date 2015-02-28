#version 400 core

in vec3 pos;
in float ri;
in float hfrag;

uniform mat4 mView;
uniform mat4 projection;
uniform float pointRadius;

out vec4 hn;

void main() {
	//calculate normal
	vec3 normal;
	normal.xy = gl_PointCoord * 2.0 - 1.0;
	float r2 = dot(normal.xy, normal.xy);
	
	if (r2 > 1.0) {
		discard;
	}

	normal.z = sqrt(1.0 - r2);

	//calculate depth
	vec4 pixelPos = vec4(pos + normal * ri, 1.0);
	vec4 clipSpacePos = projection * pixelPos;

	hn = vec4(normal, hfrag);

	gl_FragDepth = (clipSpacePos.z / clipSpacePos.w) * 0.5f + 0.5f;
}