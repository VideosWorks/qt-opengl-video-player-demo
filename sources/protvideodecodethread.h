/****************************************************************************
**
** Copyright (C) 2010-2018 Kazo Vision. (http://www.kazovision.com)
** All rights reserved.
**
****************************************************************************/

#ifndef protvideodecodethread_h
#define protvideodecodethread_h

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


#include <QtCore/QThread>
#include <QtCore/QFile>
#include <QMutex>

/*! \brief videoprocess::protVideoDecodeThread
	\author luofeng (huai_f@live.cn)
	\version 0.01
	\date 2018.02.27

	��Ƶ�ļ������߳�
*/
class protVideoDecodeThread : public QThread
{
	Q_OBJECT

public:
	protVideoDecodeThread();

	~protVideoDecodeThread();

	bool Initialize(const QString &pVideoFilePath);

	bool OpenVideoCodec();

	void Close();

	void CHECK(CUresult result);

protected:
	virtual void run() override;


private:
	static enum AVPixelFormat GetHardwareFormat(AVCodecContext *pCtx, const enum AVPixelFormat *pPixelFmts);

signals:
	void OnVideoFrameReceived(AVFrame *pFrame);

private:
	AVFormatContext *mFormatContext;
	AVCodecContext *mVideoDecodeCtx;
	AVBufferRef *mHardwareDeviceCtx;
	static enum AVPixelFormat mHardwarePixelFmt;

	AVPacket *mVideoPacket;	

	QFile mVideoFile;

	QMutex mMutex;
	bool mQuit;

	bool mInitialize;
	bool mPlayOneFrame;
	int mCurrentFrameIndex;

	float mFrameInterval;//������һ֡��ʱ����
	float mAverageFrameInterval;//ƽ��ÿһ֡��ʱ����������ʱ���֡����ת��
	float mPlaybackSpeed;//�����ٶ�
	float mFrameRate;

	int mActuralFrameIndexDiff;
};

#endif //protvideodecodethread_h