#version 400 core

in vec2 coord;

uniform vec3 color;
uniform sampler2D depthMap;
uniform sampler2D thicknessMap;
uniform sampler2D normalMap;
uniform vec2 screenSize;
uniform mat4 projection;
uniform mat4 mView;
uniform float zNear;
uniform float zFar;

out vec4 fragColor;

const vec3 lightDir = vec3(.5, .5, .5);
const float shininess = 100.0;
const vec3 specularColor = vec3(1.0, 1.0, 1.0);
const float fresPower = 2.0f;
const float fresScale = 0.4;
const float fresBias = 0.1;
const vec3 thicknessRefraction = vec3(0.02, 0.03, 0.06);

float linearizeDepth(float depth) {	
	return (2.0 * zNear) / (zFar + zNear - depth * (zFar - zNear));
}

vec3 uvToEye(vec2 p, float z) {
	vec2 pos = p * 2.0f - 1.0f;
	vec4 clipPos = vec4(pos, z, 1.0f);
	vec4 viewPos = inverse(projection) * clipPos;
	return viewPos.xyz / viewPos.w;
}

void main() {
    float depth = texture(depthMap, coord).x;
    depth = linearizeDepth(depth);
    if (depth >= .99f) {
     	discard;
     }
    
    vec3 pos = uvToEye(coord, depth);
    
	vec3 normal = texture(normalMap, coord).xyz;
	
	//Color from absorption
    float thickness = texture(thicknessMap, coord).x;

    vec3 cBeer = vec3(exp(-1*thickness), exp(-.1*thickness), exp(-.001*thickness));
    vec3 absorbColor = cBeer;
    
    //Diffuse light
    float diffuse = max(0.0f, dot(normal, lightDir) * 0.5f + 0.5f);
    
    //Phong specular
    vec3 viewDir = normalize(-pos);
    vec3 halfVec = normalize(viewDir + lightDir);
    float specular = pow(max(0.0f, dot(normal, halfVec)), shininess);	
    
    //Fresnel
    //float fresnel = fresBias + fresScale * pow(1.0f - max(0.0f, dot(normal, viewDir)), fresPower);
    
    float alpha = thickness;
    
    //Compositing everything
    vec4 finalColor = vec4(absorbColor + diffuse*specularColor.xyz*specular, alpha);
    
	//fragColor = vec4(diffuse * cBeer + specular, alpha);
	//fragColor = vec4(depth);
	//fragColor = vec4(diffuse * color, 1.0f);
	fragColor = finalColor;
	//fragColor = vec4(fresnel * vec3(1), 1);
   	//fragColor = vec4(cBeer, 1.0f);
   	//fragColor = vec4(normal, 1.0f);
   	//fragColor = vec4(thickness);

   	gl_FragDepth = depth;
}