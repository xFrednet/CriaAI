#pragma once

#include "../../Common.hpp"
#include "../ScreenCapturer.h"

#include <windows.h>

namespace cria_ai { namespace api { namespace win {
	
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
		crresult init(const CR_RECT& cArea, uint8 displayNo) override;
		
		~CRWinScreenCapturer();

		crresult updateRectangle(CR_RECT area) override;
		crresult grabFrame() override;
	};

}}}