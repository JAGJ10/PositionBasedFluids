#version 400 core

in vec3 pos;
in float lifetime;
in float type;

uniform sampler2D foamDepthMap;
uniform sampler2D fluidDepthMap;
uniform mat4 mView;
uniform mat4 projection;
uniform vec2 screenSize;

out float fThickness;

float linearizeDepth(float depth) {
	return (2.0 * 2) / (20 + 2 - depth * (20 - 2));
}

void main() {
	//calculate normal
	vec3 normal;
	normal.xy = gl_PointCoord * 2.0 - 1.0;
	float r2 = dot(normal.xy, normal.xy);
	
	if (r2 > 1.0f) {
		discard;
	}
	
	float r = sqrt(r2);

	if (r <= 1) {
		if (type == 1) {
			fThickness = 1 - pow(r, 1.5);
		} else if (type == 2) {
			fThickness = 1 - (1 - pow(r, 2));
		} else {
			fThickness = 1 - pow(r, 2.25);
		}
	} else {
		fThickness = 0;
		return;
	}

	fThickness *= pow(1 - pow(lifetime, 2), 0.4);

	vec2 coord = vec2(gl_FragCoord.x / screenSize.x, gl_FragCoord.y / screenSize.y);

	float zFluid = texture(fluidDepthMap, coord).x;
	zFluid = linearizeDepth(zFluid);
	float zFoam = texture(foamDepthMap, coord).x;
	zFoam = linearizeDepth(zFoam);

	if (zFoam > zFluid) {
		if ((zFoam - zFluid) / .01 <= 1) {
			fThickness *= pow(1 - pow((zFoam - zFluid) / .01, 1), 4);
		} else {
			fThickness = 0;
		}
	}
}