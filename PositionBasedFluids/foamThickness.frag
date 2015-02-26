#version 400 core

in vec3 pos;
in float lifetime;

//uniform sampler2D fluidDepth;
//uniform sampler2D foamDepth;
uniform mat4 mView;
uniform mat4 projection;
uniform int type;

out float fThickness;

void main() {
	//calculate normal
	vec3 normal;
	normal.xy = gl_PointCoord * 2.0 - 1.0;
	float r2 = dot(normal.xy, normal.xy);
	
	if (r2 > 1.0f) {
		discard;
	}
	
	float r = sqrt(r2);
	
	if (r2 <= 1) {
		if (type == 0) {
			fThickness = 1 - pow(r, 1.5);
		} else if (type == 1) {
			fThickness = 1 - (1 - pow(r, 2));
		} else {
			fThickness = 1 - pow(r, 2.25);
		}
	} else {
		fThickness = 0;
		return;
	}

	fThickness *= pow(1 - pow(lifetime, 2), 0.4);

	//float zFluid = texture(fluidDepth, )
}