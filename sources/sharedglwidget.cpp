#include "sharedglwidget.h"

#include <QtCore/qglobal.h>
#include <QOpenGLTexture>
#include <QMouseEvent>
#include <QGuiApplication>

#include "prothighperformancetimer.h"

SharedGLWidget::SharedGLWidget(QWidget *parent)
    : QOpenGLWidget(parent)
{
	setAttribute(Qt::WA_StaticContents);
	setAttribute(Qt::WA_NoSystemBackground);
	setAttribute(Qt::WA_OpaquePaintEvent);
	setAttribute(Qt::WA_NativeWindow);

	mVideoWidth = 1920;
	mVideoHeight = 1080;
	resize(1280, 720);

	connect(this, &SharedGLWidget::OnContextWanted, this, &SharedGLWidget::DoContextWanted, Qt::QueuedConnection);

	//调用show()函数来初始化QOpenGLcontext
	show();

	mOffscreenSurface = new QOffscreenSurface();
	mOffscreenSurface->create();

	mSharedGLContext = new QOpenGLContext();
	mSharedGLContext->setScreen(mOffscreenSurface->screen());
	mSharedGLContext->setShareContext(context());
	//mSharedGLContext->setFormat(format());
	if (!mSharedGLContext->create()) {
		qDebug() << "Failed to create OpenGLContext.";
	}

}
 
SharedGLWidget::~SharedGLWidget()
{
	if (mSharedGLContext) {
		delete mSharedGLContext;
		mSharedGLContext = NULL;
	}
	if (mOffscreenSurface) {
		delete mOffscreenSurface;
		mOffscreenSurface = NULL;
	}
}

QPaintEngine * SharedGLWidget::paintEngine() const
{
	return nullptr;
}

void SharedGLWidget::initializeGL()
{
	QMutexLocker locker(&m_renderMutex);

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

	glGenTextures(1, &mUVTexture);
	glBindTexture(GL_TEXTURE_2D, mUVTexture);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	InitializePOB();
}

void SharedGLWidget::paintGL()
{
	QMutexLocker locker(&m_renderMutex);
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
	glBindTexture(GL_TEXTURE_2D, mUVTexture);
	//glTexImage2D(GL_TEXTURE_2D, 0, GL_RG, w >> 1, h >> 1, 0, GL_RG, GL_UNSIGNED_BYTE, data.data() + w * h);

	program.setUniformValue("textureY", 0);
	program.setUniformValue("textureUV", 1);

	glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	program.disableAttributeArray("vertexIn");
	program.disableAttributeArray("textureIn");
	program.release();
}

void SharedGLWidget::resizeGL(int w, int h)
{
}

void SharedGLWidget::InitializePOB()
{
	cudaDeviceSynchronize();
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

void SharedGLWidget::Clear()
{ 
	QMutexLocker locker(&m_renderMutex);
	makeCurrent();
	glDeleteTextures(1, &mYTexture);

	CHECK(cuGraphicsUnmapResources(1, &mY_Resource, 0));
	CHECK(cuGraphicsUnregisterResource(mY_Resource));
	glDeleteBuffers(1, &mY_PBO);

	CHECK(cuGraphicsUnmapResources(1, &mUV_Resource, 0));
	CHECK(cuGraphicsUnregisterResource(mUV_Resource));
	glDeleteBuffers(1, &mUV_PBO);
	doneCurrent();
}

void SharedGLWidget::DoVideoFrameReceived(AVFrame *pFrame)
{
	//https://devblogs.nvidia.com/cuda-pro-tip-always-set-current-device-avoid-multithreading-bugs/
	cudaSetDevice(0);
	mDecodeThread = QThread::currentThread();
	if (mSharedGLContext->thread() != mDecodeThread) {
		m_grabMutex.lock();
		//将QOpenGLContext设置为当前线程才能够就行gl函数调用
		emit OnContextWanted();
		m_grabCond.wait(&m_grabMutex);
		m_grabMutex.unlock();
	}
	QMutexLocker locker(&m_renderMutex);
	bool r = mSharedGLContext->makeCurrent(mOffscreenSurface);
	
	QOpenGLContext *ctx = QOpenGLContext::currentContext();
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
		glBindTexture(GL_TEXTURE_2D, mUVTexture);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RG, mVideoWidth / 2, mVideoHeight / 2, 0, GL_RG, GL_UNSIGNED_BYTE, 0);
		glBindTexture(GL_TEXTURE_2D, 0);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
	}
	mSharedGLContext->doneCurrent();
	
	update();
}

void SharedGLWidget::DoContextWanted()
{
	QMutexLocker lock(&m_grabMutex);
	mSharedGLContext->moveToThread(mDecodeThread);
	m_grabCond.wakeAll();
}
