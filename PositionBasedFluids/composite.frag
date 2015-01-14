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
const float shininess = 150.0;
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
    if (depth == 1f) {
     	discard;
     }
    
    vec3 pos = uvToEye(coord, depth);
    
	vec3 normal = texture(normalMap, coord).xyz;
	
	//Color from absorption
    float thickness = texture(thicknessMap, coord).x / 10;
    vec3 cBeer = vec3(exp(-1*thickness), exp(-.1*thickness), exp(-.001*thickness));
    vec3 absorbColor = cBeer;
    
    //Diffuse light
    float diffuse = dot(normal, lightDir) * 0.5f + 0.5f;
    
    //Phong specular
    vec3 viewDir = normalize(-pos);
    vec3 halfVec = normalize(viewDir + lightDir);
    float specular = pow(max(0.0f, dot(normal, halfVec)), shininess);
    
    // specular light
	//float k = max(dot(viewDir, reflect(-lightDir, normal)), 0);
	//k = pow(k, 8);
	
	// Schlick's approximation for the fresnel term
	//float cos_theta = dot(viewDir, normal);
	//float fresnel = 0.75 + (1 - 0.75) * pow(1 - cos_theta, 5);
	//k *= fresnel;
	
	//vec3 specular = k * min(cBeer + 0.5, 1);
    
    //Fresnel
    float fresnel = fresBias + fresScale * pow(1.0f - max(0.0f, dot(normal, viewDir)), fresPower);
    
    float alpha = 1 - exp(-.25*thickness);
    
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