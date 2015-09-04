#version 420 core

in vec2 coord;

uniform mat4 inverseMView;

uniform sampler2D positionMap;
uniform sampler2D normalMap;
uniform sampler2D colorMap;

uniform vec3 l;

out vec4 fragColor;

const float specularPower = 16.0f;
const vec3 lightColor = vec3(0.5);

float linearizeDepth(float depth) {
	float f = 2000.0;
	float n = 10.0;
	return (2 * n) / (f + n - depth * (f - n));
}

void main() {
    vec3 n = normalize(texture(normalMap, coord).xyz);
	float s = texture(normalMap, coord).w;
	vec3 pos = texture(positionMap, coord).xyz;
	vec3 color = texture(colorMap, coord).xyz;

	vec3 ambient = color * 0.1;

	vec3 v = -normalize(pos);
	vec3 h = normalize(v + l);

	float ndotl = dot(n, l);
	vec3 diffuse = max(0.0f, ndotl) * color;

	vec3 specular = vec3(0);
	if (ndotl >= 0) specular = pow(max(0.0f, dot(n, h)), specularPower) * vec3(s);

	vec3 finalColor = (diffuse + specular);

	fragColor = vec4(finalColor, 1);
}