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

#ifndef GLWIDGET_H
#define GLWIDGET_H

#include <QtCore/QMutex>
#include <QtCore/QWaitCondition>
#include <QtCore/QThread>

#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include <QOpenGLExtraFunctions>
#include <QOpenGLBuffer>

#include <QOpenGLShaderProgram>
#include <QFile>
#include <QtCore/QThread>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavformat/avio.h>
#include <libswscale/swscale.h>
#include <libavdevice/avdevice.h>
#include <libavutil/imgutils.h>
#include <libavutil/file.h>
#include <libavutil/hwcontext.h>
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>
}
#include <cuda.h>
#include <cudaGL.h>

QT_FORWARD_DECLARE_CLASS(QOpenGLTexture)

/*
	单个OpenGL Context，工作线程等待知道context切换到这个线程后才能继续执行
*/
class GLWidget : public QOpenGLWidget, protected QOpenGLExtraFunctions
{
    Q_OBJECT

public:
    explicit GLWidget(QWidget *parent = 0);
    ~GLWidget();

	virtual QPaintEngine* paintEngine() const override;
protected:
	void paintEvent(QPaintEvent *) override;

signals:
    void clicked();

	void OnContextWanted();

protected:

    void initializeGL() override;
	
	void InitializePOB();

	void timerEvent(QTimerEvent *pEvent);

	void Render();

	void Clear();

protected slots:
	void DoVideoFrameReceived(AVFrame *pFrame);

	void onAboutToCompose();
	void onFrameSwapped();
	void onAboutToResize();
	void onResized();

	void DoContextWanted();

	void CHECK(CUresult result)
	{
		if (result != CUDA_SUCCESS)
		{
			char msg[256];
			const char* e = msg;
			cuGetErrorName(result, &e);
			printf("WAK Error %d %s\n", result, e);
			exit(1);
		}
	}
private:
	QOpenGLShaderProgram program;
	GLuint mYTexture, idUV;
	QOpenGLBuffer vbo;
	QFile file;
	int mVideoWidth;
	int mVideoHeight;

	QMutex m_renderMutex;
	QMutex m_grabMutex;
	QWaitCondition m_grabCond;

	bool mSignalConnect;

	QThread *mDecodeThread;

	GLuint mY_PBO;
	CUgraphicsResource mY_Resource;
	CUdeviceptr mY_dpBackBuffer;

	GLuint mUV_PBO;
	CUgraphicsResource mUV_Resource;
	CUdeviceptr mUV_dpBackBuffer;
};

#endif
