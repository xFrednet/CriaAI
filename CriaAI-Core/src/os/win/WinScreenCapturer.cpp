#include "WinScreenCapturer.h"

#ifdef CRIA_OS_WIN

#if (CR_SCREENCAP_CHANNEL_COUNT != 4)
#	error The window screen capturer does not support a different channel count than 4.
#endif

namespace cria_ai { namespace os { namespace win {
	
	CRWinScreenCapturer::CRWinScreenCapturer()
		: CRScreenCapturer(),
		m_SrcHwnd(nullptr),
		m_WinBmp(nullptr),
		m_WinBmpInfo(),
		m_BmpIntBuffer(nullptr)
	{
	}
	crresult CRWinScreenCapturer::init(CRWindowPtr& target)
	{
		m_SrcHwnd = GetDesktopWindow();
		if (!m_SrcHwnd)
			return CRRES_ERR_WIN_UNKNOWN;

		/*
		 * Return
		 */
		return setTarget(target);
	}

	CRWinScreenCapturer::~CRWinScreenCapturer()
	{
		if (m_WinBmp)
			DeleteObject(m_WinBmp);

		if (m_BmpIntBuffer)
			free(m_BmpIntBuffer);
	}

	crresult CRWinScreenCapturer::newTarget(CRWindowPtr& target)
	{
		if (!m_SrcHwnd)
			return CRRES_ERR_MISSING_INFORMATION;

		/*
		 * Deleting the old winBmp and keeping it null until the resizing is complete.
		 */
		if (m_WinBmp)
		{
			DeleteObject(m_WinBmp);
			m_WinBmp = nullptr;
		}

		/*
		* retrieving the client area
		*/
		CR_RECT area = target->getClientArea();

		/*
		 * creating a new windows bitmap
		 */
		HBITMAP winBmp;
		{
			HDC srcDC = GetDC(m_SrcHwnd);
			if (!srcDC)
				return CRRES_ERR_WIN_FAILED_TO_RETRIVE_DC;
			winBmp = CreateCompatibleBitmap(srcDC, area.Width, area.Height);
			if (!winBmp)
				return CRRES_ERR_WIN_FAILED_TO_CREATE_HBMP;
			ReleaseDC(m_SrcHwnd, srcDC);
		}

		/*
		 * updating the windows bitmap info
		 */
		{
			memset(&m_WinBmpInfo, 0, sizeof(BITMAPINFOHEADER));
			m_WinBmpInfo.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
			m_WinBmpInfo.bmiHeader.biWidth = area.Width;
			m_WinBmpInfo.bmiHeader.biHeight = -(int)area.Height;
			m_WinBmpInfo.bmiHeader.biPlanes = 1;
			m_WinBmpInfo.bmiHeader.biBitCount = CR_SCREENCAP_CHANNEL_COUNT * 8;
		}

		/*
		 * Recreating the bitmap int buffer
		 */
		{
			if (m_BmpIntBuffer)
				free(m_BmpIntBuffer);

			size_t size = area.Width * area.Height * CR_SCREENCAP_CHANNEL_COUNT;
			m_BmpIntBuffer = (byte*)malloc(size);
			if (!m_BmpIntBuffer)
				return CRRES_ERR_MALLOC_FAILED;
			memset(m_BmpIntBuffer, 0, size);
		}

		/*
		 * Recreating the last frame bitmap
		 */
		CR_FLOAT_BITMAP* newFrameBmp = CRCreateFBmp(area.Width, area.Height, CR_SCREENCAP_CHANNEL_COUNT);
		CR_FLOAT_BITMAP* oldFrameBmp = m_LastFrame;
		m_LastFrame = newFrameBmp;
		CRDeleteFBmp(oldFrameBmp);
		if (!newFrameBmp)
			return CRRES_ERR_UTILS_FAILED_TO_CREATE_FBMP;

		/*
		 * Return
		 */
		m_WinBmp = winBmp;
		return CRRES_OK;
	}

	crresult CRWinScreenCapturer::grabFrame()
	{
		if (!m_WinBmp)
			return CRRES_ERR_TIMING_THREADED_YAY_MULTI;

		/*
		* Windows stuff :/
		*/
		HDC srcDC = GetDC(m_SrcHwnd);
		HDC dstDC = CreateCompatibleDC(srcDC);
		HGDIOBJ oldDstDcObj = SelectObject(dstDC, m_WinBmp);
		if (!srcDC || !dstDC)
		{
			if (!srcDC)
				return CRRES_ERR_WIN_FAILED_TO_RETRIVE_DC;
			
			return CRRES_ERR_WIN_FAILED_TO_CREATE_DC;
		}

		/*
		 * Getting the Data
		 */
		CR_RECT area = m_Target->getClientArea();
		if (!BitBlt(dstDC, 0, 0, area.Width, area.Height, srcDC, area.X, area.Y, SRCCOPY) || /* copy data to the win bmp */
			GetDIBits(dstDC, m_WinBmp, 0, area.Height, m_BmpIntBuffer, &m_WinBmpInfo, DIB_RGB_COLORS) != (int)area.Height) /* copy from win bmp to the int buffer*/
			return CRRES_ERR_WIN_UNKNOWN;
		for (uint pxNo = 0; pxNo < area.Width * area.Height * CR_SCREENCAP_CHANNEL_COUNT; pxNo += 4)
		{
			m_LastFrame->Data[pxNo + 0] = ((float)m_BmpIntBuffer[pxNo + 2]) / 255.0f; /* R */
			m_LastFrame->Data[pxNo + 1] = ((float)m_BmpIntBuffer[pxNo + 1]) / 255.0f; /* G */
			m_LastFrame->Data[pxNo + 2] = ((float)m_BmpIntBuffer[pxNo + 0]) / 255.0f; /* B */
			m_LastFrame->Data[pxNo + 3] = 1.0f; /* A */
		}

		/*
		 * Finishing windows stuff
		 */
		SelectObject(dstDC, oldDstDcObj);
		DeleteDC(dstDC);
		ReleaseDC(m_SrcHwnd, srcDC);

		/*
		* Returning de la output
		*/
		return CRRES_SUCCESS;
	}

}}}

#endif // CRIA_OS_WIN
