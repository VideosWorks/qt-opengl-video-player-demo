/****************************************************************************
**
** Copyright (C) 2010-2018 Kazo Vision. (http://www.kazovision.com)
** All rights reserved.
**
****************************************************************************/

#ifndef prothighperformancetimer_h
#define prothighperformancetimer_h

#include <QtCore/QObject>
#include <QtCore/QDebug>

#ifdef Q_OS_WIN32
	#include <d3d11.h>
#else
	#define LARGE_INTEGER int
#endif //Q_OS_WIN32


/*! \brief common::protHighPerformanceTimer
	\author luofeng (huai_f@live.cn)
	\version 0.01
	\date 2018.02.27
	
	精确计算程序运行时间
*/
class protHighPerformanceTimer : public QObject
{
public:
	protHighPerformanceTimer()
	{
#ifdef Q_OS_WIN32
		QueryPerformanceFrequency(&mFrequency);
		QueryPerformanceCounter(&mStartTime);
#else
		//TODO:
#endif //Q_OS_WIN32
	}

	~protHighPerformanceTimer()
	{

	}

	inline void Start()
	{
#ifdef Q_OS_WIN32
		QueryPerformanceCounter(&mStartTime);
#else
		//TODO:
#endif //Q_OS_WIN32
	}

	inline void Stop(const QString &pMessage = QString()) 
	{
#ifdef Q_OS_WIN32
		QueryPerformanceCounter(&mStopTime);
		qDebug() <<(mStopTime.QuadPart - mStartTime.QuadPart) * 1000000LL / mFrequency.QuadPart << " us";
		//qDebug() << pMessage.toStdString().data() << (mStopTime.QuadPart - mStartTime.QuadPart) * 1000000LL / mFrequency.QuadPart << "us";
#else
		//TODO:
#endif //Q_OS_WIN32
	}

private:
	LARGE_INTEGER mFrequency;
	LARGE_INTEGER mStartTime;
	LARGE_INTEGER mStopTime;

};

#endif //highperformancetimer_h
