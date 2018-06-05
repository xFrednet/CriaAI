#pragma once

#include "../Common.hpp"
#include "Window.h"
#include <mutex>

#define CR_SCREENCAP_CHANNEL_COUNT     4

#ifndef CR_SCREENCAP_THREAD_BREAK_ON_FAIL
#	define CR_SCREENCAP_THREAD_BREAK_ON_FAIL     true
#endif

namespace cria_ai { namespace os {

	class CRScreenCapturer 
	{
	public:
		static CRScreenCapturer* CreateInstance(CRWindowPtr target, crresult* result = nullptr);
		
		/*
		 * * These functions are API specific.
	     */
	protected:
		CRWindowPtr m_Target;

		std::mutex       m_FrameLock;
		CR_FLOAT_BITMAP* m_Frame;
		CR_RECT          m_FrameSize;
		bool             m_ContinueCapture;

		std::thread      m_CaptureThread;

		CRScreenCapturer();
		virtual crresult init() = 0;
		virtual crresult newTarget(CRWindowPtr& target) = 0;
	public:
		virtual ~CRScreenCapturer();

		virtual crresult setTarget(CRWindowPtr& target);
		virtual crresult grabFrame() = 0;
		
		/*
		 * Thread
		 */
		crresult runCaptureThread();
		crresult stopCaptureThread();
		bool isCaptureThreadRunnning() const;

		/**
		 * \brief This retruns the latest captured frame this can only be done
		 * once per frame to support multi threading. The requested bitmap has
		 * to be deleted by the requester using "CRDeleteFBmpNormal".
		 * 
		 * \return This returns a bitmap that has to be requested has to be 
		 * deleted by the requester using "CRDeleteFBmpNormal"
		 */
		CR_FLOAT_BITMAP* getFrame();
	};

}}