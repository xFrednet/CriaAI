#pragma once

#include "../ScreenCapturer.h"

#include "../../Common.hpp"

#ifdef CRIA_OS_WIN

#include "WinOSContext.h"
#include "WinWindow.h"

namespace cria_ai { namespace os { namespace win {
	
	/*
	 * This capturer is currently only able to capture areas on the primary monitor.
	 */
	class CRWinScreenCapturer : public CRScreenCapturer
	{
	private:
		HWND m_SrcHwnd;

		HBITMAP m_WinBmp;
		BITMAPINFO m_WinBmpInfo;
		byte* m_BmpIntBuffer;

	public:
		CRWinScreenCapturer();
		crresult init(CRWindowPtr target) override;
		
		~CRWinScreenCapturer();

		crresult newTarget(CRWindowPtr target) override;
		crresult grabFrame() override;
	};

}}}
#endif //CRIA_OS_WIN