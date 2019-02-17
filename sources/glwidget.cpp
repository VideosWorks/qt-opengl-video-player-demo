/****************************************************************************
**
** Copyright (C) 2016 The Qt Company Ltd.
** Contact: https://www.qt.io/licensing/
**
** This file is part of the examples of the Qt Toolkit.
**
** $QT_BEGIN_LICENSE:BSD$
** Commercial License Usage
** Licensees holding valid commercial Qt licenses may use this file in
** accordance with the commercial license agreement provided with the
** Software or, alternatively, in accordance with the terms contained in
** a written agreement between you and The Qt Company. For licensing terms
** and conditions see https://www.qt.io/terms-conditions. For further
** information use the contact form at https://www.qt.io/contact-us.
**
** BSD License Usage
** Alternatively, you may use this file under the terms of the BSD license
** as follows:
**
** "Redistribution and use in source and binary forms, with or without
** modification, are permitted provided that the following conditions are
** met:
**   * Redistributions of source code must retain the above copyright
**     notice, this list of conditions and the following disclaimer.
**   * Redistributions in binary form must reproduce the above copyright
**     notice, this list of conditions and the following disclaimer in
**     the documentation and/or other materials provided with the
**     distribution.
**   * Neither the name of The Qt Company Ltd nor the names of its
**     contributors may be used to endorse or promote products derived
**     from this software without specific prior written permission.
**
**
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
** OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
** LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
** DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
** THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
** (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
** OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."
**
** $QT_END_LICENSE$
**
****************************************************************************/

#include "glwidget.h"

#include <cuda_runtime.h>

#include <QtCore/qglobal.h>
#include <QOpenGLTexture>
#include <QMouseEvent>
#include <QGuiApplication>

#include "prothighperformancetimer.h"

GLWidget::GLWidget(QWidget *parent)
    : QOpenGLWidget(parent),
      program(0)
{
	setAttribute(Qt::WA_StaticContents);
	setAttribute(Qt::WA_NoSystemBackground);
	setAttribute(Qt::WA_OpaquePaintEvent);
	setAttribute(Qt::WA_NativeWindow);

	connect(this, &QOpenGLWidget::aboutToCompose, this, &GLWidget::onAboutToCompose);
	connect(this, &QOpenGLWidget::frameSwapped, this, &GLWidget::onFrameSwapped);
	connect(this, &QOpenGLWidget::aboutToResize, this, &GLWidget::onAboutToResize);
	connect(this, &QOpenGLWidget::resized, this, &GLWidget::onResized);

	startTimer(300);

	mVideoWidth = 1920;
	mVideoHeight = 1080;
	resize(1920, 1080);
	
	mSignalConnect = false;

	connect(this, &GLWidget::OnContextWanted, this, &GLWidget::DoContextWanted, Qt::QueuedConnection);

	HANDLE id = QThread::currentThreadId();
}

GLWidget::~GLWidget()
{
    makeCurrent();

    doneCurrent();
}

QPaintEngine * GLWidget::paintEngine() const
{
	return nullptr;
}

void GLWidget::paintEvent(QPaintEvent *)
{
}

void GLWidget::initializeGL()
{
    initializeOpenGLFunctions();

	QOpenGLShader *vshader = new QOpenGLShader(QOpenGLShader::Vertex, this);
	const char *vsrc =
		"attribute vec4 vertexIn; \
             attribute vec4 textureIn; \
             varying vec4 textureOut;  \
             void main(void)           \
             {                         \
                 gl_Position = vertexIn; \
                 textureOut = textureIn; \
             }";
	bool result = vshader->compileSourceCode(vsrc);

	QOpenGLShader *fshader = new QOpenGLShader(QOpenGLShader::Fragment, this);
	const char *fsrc =
		"varying mediump vec4 textureOut;\n"
		"uniform sampler2D textureY;\n"
		"uniform sampler2D textureUV;\n"
		"void main(void)\n"
		"{\n"
		"vec3 yuv; \n"
		"vec3 rgb; \n"
		"yuv.x = texture2D(textureY, textureOut.st).r - 0.0625; \n"
		"yuv.y = texture2D(textureUV, textureOut.st).r - 0.5; \n"
		"yuv.z = texture2D(textureUV, textureOut.st).g - 0.5; \n"
		"rgb = mat3( 1,       1,         1, \n"
		"0,       -0.39465,  2.03211, \n"
		"1.13983, -0.58060,  0) * yuv; \n"
		"gl_FragColor = vec4(rgb, 1); \n"
		"}\n";
	result = fshader->compileSourceCode(fsrc);

	program.addShader(vshader);
	program.addShader(fshader);
	program.link();

	GLfloat points[]{
		-1.0f, 1.0f,
		 1.0f, 1.0f,
		 1.0f, -1.0f,
		-1.0f, -1.0f,

		0.0f,0.0f,
		1.0f,0.0f,
		1.0f,1.0f,
		0.0f,1.0f
	};

	vbo.create();
	vbo.bind();
	vbo.allocate(points, sizeof(points));

	glGenTextures(1, &mYTexture);
	glBindTexture(GL_TEXTURE_2D, mYTexture);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glBindTexture(GL_TEXTURE_2D, 0);

	glGenTextures(1, &idUV);
	glBindTexture(GL_TEXTURE_2D, idUV);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	InitializePOB();
}

void GLWidget::InitializePOB()
{
	cudaSetDevice(0);

	glGenBuffers(1, &mY_PBO);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, mY_PBO);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, mVideoWidth * mVideoHeight, 0, GL_STREAM_DRAW);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
	CHECK(cuGraphicsGLRegisterBuffer(&mY_Resource, mY_PBO, CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD));
	CHECK(cuGraphicsMapResources(1, &mY_Resource, 0));
	size_t nSize = 0;
	CHECK(cuGraphicsResourceGetMappedPointer(&mY_dpBackBuffer, &nSize, mY_Resource));

	glGenBuffers(1, &mUV_PBO);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, mUV_PBO);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, mVideoWidth * mVideoHeight / 2, 0, GL_STREAM_DRAW);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
	CHECK(cuGraphicsGLRegisterBuffer(&mUV_Resource, mUV_PBO, CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD));
	CHECK(cuGraphicsMapResources(1, &mUV_Resource, 0));
	CHECK(cuGraphicsResourceGetMappedPointer(&mUV_dpBackBuffer, &nSize, mUV_Resource));
}

void GLWidget::timerEvent(QTimerEvent * pEvent)
{
	//update();
}

void GLWidget::Render()
{
	makeCurrent();
	glViewport(0, 0, width(), height());
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glDisable(GL_DEPTH_TEST);

	program.bind();
	program.enableAttributeArray("vertexIn");
	program.enableAttributeArray("textureIn");
	program.setAttributeBuffer("vertexIn", GL_FLOAT, 0, 2, 2 * sizeof(GLfloat));
	program.setAttributeBuffer("textureIn", GL_FLOAT, 2 * 4 * sizeof(GLfloat), 2, 2 * sizeof(GLfloat));

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, mYTexture);
	//glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, w, h, 0, GL_RED, GL_UNSIGNED_BYTE, (data.data()));

	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, idUV);
	//glTexImage2D(GL_TEXTURE_2D, 0, GL_RG, w >> 1, h >> 1, 0, GL_RG, GL_UNSIGNED_BYTE, data.data() + w * h);

	program.setUniformValue("textureY", 0);
	program.setUniformValue("textureUV", 1);

	glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	program.disableAttributeArray("vertexIn");
	program.disableAttributeArray("textureIn");
	program.release();

	doneCurrent();
}

void GLWidget::Clear()
{ 
	QMutexLocker locker(&m_renderMutex);
	makeCurrent();
	CHECK(cuGraphicsUnmapResources(1, &mY_Resource, 0));
	CHECK(cuGraphicsUnregisterResource(mY_Resource));
	glDeleteBuffers(1, &mY_PBO);

	CHECK(cuGraphicsUnmapResources(1, &mUV_Resource, 0));
	CHECK(cuGraphicsUnregisterResource(mUV_Resource));
	glDeleteBuffers(1, &mUV_PBO);
	doneCurrent();
}

void GLWidget::DoVideoFrameReceived(AVFrame *pFrame)
{
	//https://devblogs.nvidia.com/cuda-pro-tip-always-set-current-device-avoid-multithreading-bugs/
	cudaSetDevice(0);
	mDecodeThread = QThread::currentThread();
	m_grabMutex.lock();
	//将QOpenGLContext设置为当前线程才能够就行gl函数调用
	emit OnContextWanted();
	m_grabCond.wait(&m_grabMutex);
	m_grabMutex.unlock();

	QMutexLocker lock(&m_renderMutex);
	QOpenGLContext *ctx = context();
	Q_ASSERT(context()->thread() == QThread::currentThread());
	makeCurrent();
	{
		CUdeviceptr dpFrame = (CUdeviceptr)pFrame->data[0];

		CUDA_MEMCPY2D m = { 0 };
		m.srcMemoryType = CU_MEMORYTYPE_DEVICE;
		m.srcDevice = dpFrame;
		m.srcPitch = pFrame->linesize[0];
		m.dstMemoryType = CU_MEMORYTYPE_DEVICE;
		m.dstDevice = mY_dpBackBuffer;
		m.dstPitch = mVideoWidth;
		m.WidthInBytes = mVideoWidth;
		m.Height = mVideoHeight;
		CHECK(cuMemcpy2DAsync(&m, 0));

		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, mY_PBO);
		glBindTexture(GL_TEXTURE_2D, mYTexture);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, mVideoWidth, mVideoHeight, 0, GL_RED, GL_UNSIGNED_BYTE, 0);
		glBindTexture(GL_TEXTURE_2D, 0);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
	}
	{
		CUdeviceptr dpFrame = (CUdeviceptr)pFrame->data[0];
		CUDA_MEMCPY2D m = { 0 };
		m.srcMemoryType = CU_MEMORYTYPE_DEVICE;
		m.srcDevice = dpFrame;
		m.srcPitch = pFrame->linesize[0];
		m.dstMemoryType = CU_MEMORYTYPE_DEVICE;
		m.dstDevice = mUV_dpBackBuffer;
		m.dstPitch = mVideoWidth;
		m.WidthInBytes = mVideoWidth;
		m.Height = mVideoHeight / 2;
		m.srcY = mVideoHeight/* + 8*/;
		CHECK(cuMemcpy2DAsync(&m, 0));

		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, mUV_PBO);
		glBindTexture(GL_TEXTURE_2D, idUV);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RG, mVideoWidth / 2, mVideoHeight / 2, 0, GL_RG, GL_UNSIGNED_BYTE, 0);
		glBindTexture(GL_TEXTURE_2D, 0);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
	}
	doneCurrent();

	Render();

	context()->moveToThread(qGuiApp->thread());
	
	update();
}

void GLWidget::onAboutToCompose()
{
	// We are on the gui thread here. Composition is about to
	// begin. Wait until the render thread finishes.
	m_renderMutex.lock();
}

void GLWidget::onFrameSwapped()
{
	m_renderMutex.unlock();
}

void GLWidget::onAboutToResize()
{
	m_renderMutex.lock();
}

void GLWidget::onResized()
{
	m_renderMutex.unlock();
}

void GLWidget::DoContextWanted()
{
	m_renderMutex.lock();
	QMutexLocker lock(&m_grabMutex);
	QOpenGLContext *ctx = context();
	context()->moveToThread(mDecodeThread);
	m_grabCond.wakeAll();
	m_renderMutex.unlock();
}
