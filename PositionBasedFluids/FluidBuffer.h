#ifndef FluidBuffer_H
#define FluidBuffer_H

#include "Shader.h"

class FluidBuffer {
public:
	GLuint cloth, depth, thickness, blurH, blurV, fluid, foamDepth, foamThickness, foamIntensity, foamRadiance;

	FluidBuffer(int widthIn, int heightIn);
	~FluidBuffer();

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