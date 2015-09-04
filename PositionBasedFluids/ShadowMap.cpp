#include "ShadowMap.h"

ShadowMap::ShadowMap(int width, int height) : width(width), height(height) {
	//Create FBO
	glGenFramebuffers(1, &fbo);
	glBindFramebuffer(GL_FRAMEBUFFER, fbo);

	//Create depth texture
	glGenTextures(1, &depth);
	glBindTexture(GL_TEXTURE_2D, depth);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_R_TO_TEXTURE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL);
	glTexParameteri(GL_TEXTURE_2D, GL_DEPTH_TEXTURE_MODE, GL_INTENSITY);

	glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depth, 0);
	
	glDrawBuffer(GL_NONE);
	glReadBuffer(GL_NONE);

	//Unbind
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

ShadowMap::~ShadowMap() {
	if (fbo != 0) {
		glDeleteFramebuffers(1, &fbo);
		glDeleteTextures(1, &depth);
	}
}

void ShadowMap::bind() {
	glBindFramebuffer(GL_FRAMEBUFFER, fbo);
}

void ShadowMap::bindDraw() {
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo);
}

void ShadowMap::bindRead() {
	glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo);
}

void ShadowMap::unbind() {
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void ShadowMap::unbindDraw() {
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
}

void ShadowMap::unbindRead() {
	glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
}