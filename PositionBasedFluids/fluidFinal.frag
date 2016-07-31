#version 400 core

in vec2 coord;

uniform vec4 color;
uniform sampler2D depthMap;
uniform sampler2D thicknessMap;
uniform sampler2D sceneMap;
uniform sampler2DShadow shadowMap;
uniform mat4 projection;
uniform mat4 inverseProjection;
uniform mat4 mView;
uniform mat4 inverseMView;
uniform vec2 invTexScale;
uniform mat4 shadowMapMVP;
uniform int shadowMapWidth;
uniform int shadowMapHeight;

out vec4 fragColor;

const vec4 lightDir = vec4(-1, 1, 1, 0);
const float shininess = 300.0;
const float fresPower = 5.0f;
const float fresScale = 0.9;
const float fresBias = 0.1;

float linearizeDepth(float depth) {
	float f = 100.0;
	float n = 1.0;
	return (2 * n) / (f + n - depth * (f - n));
}

float getShadowFactor(vec3 position) {
	//vec4 p = inverseMView * vec4(position, 1.0);
	vec4 p = vec4(position, 1.0);
	p = shadowMapMVP * p;
	p /= p.w;
	p.xyz = p.xyz * 0.5 + 0.5;

	float factor = 0;

	vec2 offset = vec2(1.0 / float(shadowMapWidth), 1.0 / float(shadowMapHeight));

	for (int y = -1; y <= 1; y++) {
		for (int x = -1; x <= 1; x++) {
			vec3 uvc = vec3(p.xy + (vec2(x,y) * offset), p.z + 0.005);
			factor += texture(shadowMap, uvc);
        }
    }

	return (0.5 + (factor / 18));
}

vec3 uvToEye(vec2 p, float z) {
	vec2 pos = p * 2.0f - 1.0f;
	vec4 clipPos = vec4(pos, z, 1.0f);
	vec4 viewPos = inverse(projection) * clipPos;
	return viewPos.xyz / viewPos.w;
}

void main() {
	vec4 scene = texture(sceneMap, coord);
    float depth = texture(depthMap, coord).x;
	if (depth == 0.0f) {
		discard;
	}
	
	// reconstruct eye space pos from depth
	vec3 eyePos = uvToEye(coord, depth);

	// finite difference approx for normals, can't take dFdx because
	// the one-sided difference is incorrect at shape boundaries
	vec3 zl = eyePos - uvToEye(coord - vec2(invTexScale.x, 0.0), texture(depthMap, coord - vec2(invTexScale.x, 0.0)).x);
	vec3 zr = uvToEye(coord + vec2(invTexScale.x, 0.0), texture(depthMap, coord + vec2(invTexScale.x, 0.0)).x) - eyePos;
	vec3 zt = uvToEye(coord + vec2(0.0, invTexScale.y), texture(depthMap, coord + vec2(0.0, invTexScale.y)).x) - eyePos;
	vec3 zb = eyePos - uvToEye(coord - vec2(0.0, invTexScale.y), texture(depthMap, coord - vec2(0.0, invTexScale.y)).x);
	
	vec3 dx = zl;
	vec3 dy = zt;

	if (abs(zr.z) < abs(zl.z))
		dx = zr;

	if (abs(zb.z) < abs(zt.z))
		dy = zb;

	vec3 normal = normalize(cross(dx, dy));
    
	vec4 worldPos = inverseMView * vec4(eyePos, 1.0);
	//float shadowFactor = getShadowFactor(worldPos.xyz);

    //Phong specular
	vec3 l = (mView * lightDir).xyz;
    vec3 viewDir = -normalize(eyePos);
    vec3 halfVec = normalize(viewDir + l);
    float specular = pow(max(0.0f, dot(normal, halfVec)), shininess);	

	vec2 texScale = vec2(0.75, 1.0);
	float refractScale = 1.33 * 0.025;
	refractScale *= smoothstep(0.1, 0.4, worldPos.y);
	vec2 refractCoord = coord + normal.xy*refractScale*texScale;

	float thickness = max(texture(thicknessMap, refractCoord).x, 0.3);
	vec3 transmission = exp(-(vec3(1.0)-color.xyz)*thickness);

	vec3 refract = texture(sceneMap, refractCoord).xyz*transmission;
    
	float attenuation = max(smoothstep(0.95, 1.0, abs(dot(l, -lightDir.xyz))), 0.05);

	float ln = dot(l, normal)*attenuation;

    //Fresnel
    float fresnel = fresBias + fresScale * pow(1.0f - max(dot(normal, viewDir), 0.0), fresPower);

	//Diffuse light
	vec3 diffuse = color.xyz * mix(vec3(0.29, 0.379, 0.59), vec3(1.0), (ln*0.5 + 0.5)) * (1 - color.w);

	vec3 skyColor = vec3(0.1, 0.2, 0.4)*1.2;
	vec3 groundColor = vec3(0.1, 0.1, 0.2);

	vec3 rEye = reflect(viewDir, normal).xyz;
	vec3 rWorld = (inverseMView*vec4(rEye, 0.0)).xyz;

	vec3 reflect = vec3(0.75f) + mix(groundColor, skyColor, smoothstep(0.15, 0.25, rWorld.y));
    
    //Compositing everything
    vec3 finalColor = diffuse + (mix(refract, reflect, fresnel) + specular) * color.w;

	fragColor = vec4(finalColor, 1.0);
	//fragColor = vec4(finalColor * shadowFactor, 1.0);
	//fragColor = vec4(shadowFactor);
	//fragColor = vec4(depth);
	//fragColor = vec4(thickness);
	//fragColor = scene;
	//fragColor = vec4(normal, 1.0);

	vec4 clipPos = projection*vec4(0.0, 0.0, depth, 1.0);
	clipPos.z /= clipPos.w;
	gl_FragDepth = depth;//clipPos.z*0.5+0.5;
}