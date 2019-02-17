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

#include <QApplication>
#include <QSurfaceFormat>

#include "sharedglwidget.h"
#include "glwidget.h"
#include "protvideodecodethread.h"

#include <cuda_runtime.h>

//https://forum.qt.io/topic/56889/what-exactly-is-a-qoffscreensurface/4
//If you want to render in multiple threads you have two choices :
	//1.Have one context, synchronize the threads and switch the context to be current in only one of them at any given time.
	//	In practice this is useless as you are rendering in only one thread while the others wait for their turn,
	//	thus the whole effort of threading is wasted because the rendering is serialized anyway.
	
	//2.Create multiple OpenGL contexts and make them share resources.
	//  This way you can make multiple threads, each with its own context, 
	//	render at the same time.Each background thread would use a QOffscreenSurface and the main thread would use a window.

int main(int argc, char *argv[])
{
    Q_INIT_RESOURCE(textures);

	//cudaDeviceSynchronize();

	QSurfaceFormat format = QSurfaceFormat::defaultFormat();
	format.setSwapBehavior(QSurfaceFormat::DoubleBuffer);
#ifdef _DEBUG
	format.setOption(QSurfaceFormat::DebugContext);
#endif //_DEBUG
	format.setProfile(QSurfaceFormat::CoreProfile);
	format.setSwapInterval(1);
	format.setDepthBufferSize(24);
	//QSurfaceFormat::setDefaultFormat(format);

	//QApplication::setAttribute(Qt::AA_ShareOpenGLContexts);
    QApplication app(argc, argv);

	QList<QThread*> threadList;
	for (int i = 0; i < 2; i++) {
		SharedGLWidget *widget = new SharedGLWidget;
		widget->show();

	//	SharedGLWidget *widget1 = new SharedGLWidget;
	//	widget1->show();
	//	widget1->move(100, 100);
		protVideoDecodeThread *thread = new protVideoDecodeThread();
		QObject::connect(thread, SIGNAL(OnVideoFrameReceived(AVFrame*)), widget, SLOT(DoVideoFrameReceived(AVFrame*)), Qt::DirectConnection);
	//	QObject::connect(thread, SIGNAL(OnVideoFrameReceived(AVFrame*)), widget1, SLOT(DoVideoFrameReceived(AVFrame*)), Qt::DirectConnection);

		//thread->Initialize("channel0");
		thread->Initialize("channel3.avi");
//		thread->Initialize("D:\\video\\bbb_sunflower_1080p_30fps_normal_h264.mp4");
		threadList.append(thread);
	}
	foreach(QThread *t, threadList) {
		t->start();
	}
    return app.exec();
}
