/****************************************************************************
**
** Copyright (C) 2010-2018 Kazo Vision. (http://www.kazovision.com)
** All rights reserved.
**
****************************************************************************/

#include "protvideodecodethread.h"

#include <QtCore/QElapsedTimer>
#include "prothighperformancetimer.h"

enum AVPixelFormat protVideoDecodeThread::mHardwarePixelFmt = AV_PIX_FMT_NONE;

protVideoDecodeThread::protVideoDecodeThread()
	 :QThread()
{
	mFormatContext = NULL;
	mVideoDecodeCtx = NULL;
	mHardwareDeviceCtx = NULL;
	mVideoPacket = NULL;
	mCurrentFrameIndex = 0;
	//打开解码器需要较长的时间，因此提前打开
	//if (! OpenVideoCodec()) {
	//
	//}

	mQuit = false;
	mInitialize = false;
	mPlayOneFrame = false;
	mFrameRate = 30.0f;
	mFrameInterval = mAverageFrameInterval = 1000 / mFrameRate;
	mPlaybackSpeed = 1.0f;
	mActuralFrameIndexDiff = 0;
}

protVideoDecodeThread::~protVideoDecodeThread()
{
	mMutex.lock();
	mQuit = true;
	mInitialize = false;
	mPlayOneFrame = false;
	mMutex.unlock();
	wait();

	avcodec_close(mVideoDecodeCtx);
	avcodec_free_context(&mVideoDecodeCtx);
	mVideoDecodeCtx = NULL;
	av_buffer_unref(&mHardwareDeviceCtx);
	Close();
}

bool protVideoDecodeThread::Initialize(const QString &pFilePath)
{
	QByteArray videofilepathdata = pFilePath.toLocal8Bit();
	const char* videofilepath = videofilepathdata.data();
	if (avformat_open_input(&mFormatContext, videofilepath, NULL, NULL) != 0) {
	//	logger()->error("Open video file '%1' failed.", pFilePath);
		return false;
	}
	if (avformat_find_stream_info(mFormatContext, NULL) < 0) {
	//	logger()->error("Couldn't find stream information.");
		return false;
	}
	av_dump_format(mFormatContext, 0, videofilepath, 0);
	int videostreamindex = av_find_best_stream(mFormatContext, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);
	if (videostreamindex == -1) {
		//logger()->error("Couldn't find video stream index.");
		return false;
	}
	enum AVHWDeviceType hwtype = av_hwdevice_find_type_by_name("cuda");
	if (hwtype == AV_HWDEVICE_TYPE_NONE) {
		while ((hwtype = av_hwdevice_iterate_types(hwtype)) != AV_HWDEVICE_TYPE_NONE)
			printf("%s ", av_hwdevice_get_type_name(hwtype));
		printf("\n");
		return false;
	}
	AVCodec *videocodec = avcodec_find_decoder_by_name("h264_cuvid");
	//AVCodec *videocodec = avcodec_find_decoder(AV_CODEC_ID_H264);
	if (!videocodec) {
		return false;
	}
	for (int i = 0;; i++) {
		const AVCodecHWConfig *config = avcodec_get_hw_config(videocodec, i);
		if (!config) {
			return false;
		}
		if (config->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX &&
			config->device_type == hwtype) {
			mHardwarePixelFmt = config->pix_fmt;
			break;
		}
	}
	mVideoDecodeCtx = avcodec_alloc_context3(videocodec);
	if (!mVideoDecodeCtx) {
		return false;
	}
	mVideoDecodeCtx->get_format = GetHardwareFormat;

	//refcounted_frames=1表示frame的分配与释放由自己控制，为0则有ffmpeg自己控制
	//av_opt_set_int(mVideoDecodeCtx, "refcounted_frames", 1, 0);

	int err = 0;
	if ((err = av_hwdevice_ctx_create(&mHardwareDeviceCtx, hwtype, NULL, NULL, 0)) < 0) {
		return false;
	}

	mVideoDecodeCtx->hw_device_ctx = av_buffer_ref(mHardwareDeviceCtx);

	avcodec_parameters_to_context(mVideoDecodeCtx, mFormatContext->streams[videostreamindex]->codecpar);
	
	if (avcodec_open2(mVideoDecodeCtx, videocodec, NULL) < 0) {
		return false;
	}

	mVideoPacket = (AVPacket*)av_malloc(sizeof(AVPacket));
	
	return true;
}

bool protVideoDecodeThread::OpenVideoCodec()
{
	//enum AVHWDeviceType hwtype = av_hwdevice_find_type_by_name("cuda");
	//if (hwtype == AV_HWDEVICE_TYPE_NONE) {
	//	while ((hwtype = av_hwdevice_iterate_types(hwtype)) != AV_HWDEVICE_TYPE_NONE)
	//		printf("%s ", av_hwdevice_get_type_name(hwtype));
	//	printf("\n");
	//	return false;
	//}
	//AVCodec *videocodec = avcodec_find_decoder_by_name("h264_cuvid");
	//if (!videocodec) {
	//	return false;
	//}
	//for (int i = 0;; i++) {
	//	const AVCodecHWConfig *config = avcodec_get_hw_config(videocodec, i);
	//	if (!config) {
	//		return false;
	//	}
	//	if (config->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX &&
	//		config->device_type == hwtype) {
	//		mHardwarePixelFmt = config->pix_fmt;
	//		break;
	//	}
	//}
	//mVideoDecodeCtx = avcodec_alloc_context3(videocodec);
	//if (!mVideoDecodeCtx) {
	//	return false;
	//}
	//mVideoDecodeCtx->get_format = GetHardwareFormat;

	////refcounted_frames=1表示frame的分配与释放由自己控制，为0则有ffmpeg自己控制
	////av_opt_set_int(mVideoDecodeCtx, "refcounted_frames", 1, 0);

	//int err = 0;
	////create hwdevicectx
	//if ((err = av_hwdevice_ctx_create(&mHardwareDeviceCtx, hwtype, NULL, NULL, 0)) < 0) {
	//	return false;
	//}

	//mVideoDecodeCtx->hw_device_ctx = av_buffer_ref(mHardwareDeviceCtx);

	//if (avcodec_open2(mVideoDecodeCtx, videocodec, NULL) < 0) {
	//	return false;
	//}
	return true;
}

void protVideoDecodeThread::Close()
{
	QMutexLocker locker(&mMutex);
	mInitialize = false;
	mPlayOneFrame = false;
	locker.unlock();
	msleep(50);
	av_packet_free(&mVideoPacket);
	mVideoPacket = NULL;
	avformat_close_input(&mFormatContext);
	avformat_free_context(mFormatContext);
	mFormatContext = NULL;
	mCurrentFrameIndex = 0;
}

void protVideoDecodeThread::run()
{
	int ret = 0;
	AVFrame *frame = av_frame_alloc();

	protHighPerformanceTimer htimer;
	htimer.Start();
	while (true) {
		while (av_read_frame(mFormatContext, mVideoPacket) >= 0) {
			htimer.Start();
			if (avcodec_send_packet(mVideoDecodeCtx, mVideoPacket) < 0) {
				msleep(1);
				mCurrentFrameIndex++;
				continue;
			}
			ret = avcodec_receive_frame(mVideoDecodeCtx, frame);
			if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
				msleep(1);
				continue;
			}
			else if (ret < 0) {
				msleep(1);
				continue;
			}
			htimer.Start();
			emit OnVideoFrameReceived(frame);
			htimer.Stop();

			av_frame_unref(frame);
			av_packet_unref(mVideoPacket);
			msleep(30);
		}
		break;
	}
	av_frame_free(&frame);
}

enum AVPixelFormat protVideoDecodeThread::GetHardwareFormat(AVCodecContext *pCtx, const enum AVPixelFormat *pPixelFmts)
{
	const enum AVPixelFormat *p;

	for (p = pPixelFmts; *p != -1; p++) {
		if (*p == mHardwarePixelFmt)
			return *p;
	}

	fprintf(stderr, "Failed to get HW surface format.\n");
	return AV_PIX_FMT_NONE;
}