#ifndef GBUFFER_H
#define GBUFFER_H

#include "Shader.h"

class GBuffer {
public:
	//GLuint position, normal, color, depth, light, effect1, effect2;
	GLuint cloth, depth, thickness, blurH, blurV, fluid, foamDepth, foamThickness, foamIntensity, foamRadiance;

	GBuffer(int widthIn, int heightIn);
	~GBuffer();

	GLuint getFBO() const;
	int getWidth() const;
	int getHeight() const;

	void setDrawPlane();
	void setDrawCloth();
	void setDrawDepth();
	void setDrawThickness();
	void setDrawHorizontalBlur();
	void setDrawVerticalBlur();
	void setDrawFluid();
	void setDrawFoamDepth();
	void setDrawFoamThickness();
	void setDrawFoamIntensity();
	void setDrawFoamRadiance();

	void bind();
	void bindDraw();
	void bindRead();
	void unbind();
	void unbindDraw();
	void unbindRead();

private:
	GLuint fbo;

	int width, height;
};

#endif