
#ifndef SHAREDGLWIDGET_H
#define SHAREDGLWIDGET_H

#include <QtCore/QMutex>
#include <QtCore/QWaitCondition>
#include <QtCore/QThread>

#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include <QOpenGLBuffer>

#include <QOpenGLShaderProgram>
#include <QtCore/QThread>

#include <QtGui/QOffScreenSurface>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavformat/avio.h>
#include <libswscale/swscale.h>
#include <libavdevice/avdevice.h>
#include <libavutil/imgutils.h>
#include <libavutil/file.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>
}

#include <cuda_runtime.h>
#include <cudaGL.h>

QT_FORWARD_DECLARE_CLASS(QOpenGLTexture)

/*
	使用QOffscreenSurface进行离屏渲染。有2个context，可以共享纹理
*/
class SharedGLWidget : public QOpenGLWidget, protected QOpenGLFunctions
{
    Q_OBJECT

public:
    explicit SharedGLWidget(QWidget *parent = 0);
    ~SharedGLWidget();

	QPaintEngine* paintEngine() const;

signals:
	void OnContextWanted();

protected:
    void initializeGL() override;
	
	void paintGL() override;

	void resizeGL(int w, int h) override;

	void InitializePOB();

	void Clear();

protected slots:
	void DoVideoFrameReceived(AVFrame *pFrame);

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
	GLuint mYTexture, mUVTexture;
	QOpenGLBuffer vbo;
	int mVideoWidth;
	int mVideoHeight;

	QMutex m_renderMutex;
	QMutex m_grabMutex;
	QWaitCondition m_grabCond;

	QThread *mDecodeThread;

	GLuint mY_PBO;
	CUgraphicsResource mY_Resource;
	CUdeviceptr mY_dpBackBuffer;

	GLuint mUV_PBO;
	CUgraphicsResource mUV_Resource;
	CUdeviceptr mUV_dpBackBuffer;

	QOffscreenSurface *mOffscreenSurface;
	QOpenGLContext *mSharedGLContext;
};

#endif
