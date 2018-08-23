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
	crresult CRWinScreenCapturer::init()
	{
		m_SrcHwnd = GetDesktopWindow();
		if (!m_SrcHwnd)
			return CRRES_ERR_WIN_UNKNOWN;

		/*
		 * Return
		 */
		return CRRES_OK;
	}

	CRWinScreenCapturer::~CRWinScreenCapturer()
	{
		m_FrameLock.lock();
		
		if (m_WinBmp)
			DeleteObject(m_WinBmp);

		if (m_BmpIntBuffer)
			free(m_BmpIntBuffer);
		
		m_FrameLock.unlock();
	}

	crresult CRWinScreenCapturer::newTarget(CRWindowPtr& target)
	{
		if (!m_SrcHwnd)
			return CRRES_ERR_MISSING_INFORMATION;

		m_FrameLock.lock();

		/*
		 * creating a new windows bitmap
		 */
		{
			if (m_WinBmp)
			{
				DeleteObject(m_WinBmp);
				m_WinBmp = nullptr;
			}
			HDC srcDC = GetDC(m_SrcHwnd);
			if (!srcDC)
				return CRRES_ERR_WIN_FAILED_TO_RETRIVE_DC;
			m_WinBmp = CreateCompatibleBitmap(srcDC, m_FrameSize.Width, m_FrameSize.Height);
			if (!m_WinBmp)
				return CRRES_ERR_WIN_FAILED_TO_CREATE_HBMP;
			ReleaseDC(m_SrcHwnd, srcDC);
		}

		/*
		 * updating the windows bitmap info
		 */
		{
			memset(&m_WinBmpInfo, 0, sizeof(BITMAPINFOHEADER));
			m_WinBmpInfo.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
			m_WinBmpInfo.bmiHeader.biWidth = m_FrameSize.Width;
			m_WinBmpInfo.bmiHeader.biHeight = -(int)m_FrameSize.Height;
			m_WinBmpInfo.bmiHeader.biPlanes = 1;
			m_WinBmpInfo.bmiHeader.biBitCount = CR_SCREENCAP_CHANNEL_COUNT * 8;
		}

		/*
		 * Recreating the bitmap int buffer
		 */
		{
			if (m_BmpIntBuffer)
				free(m_BmpIntBuffer);

			size_t size = m_FrameSize.Width * m_FrameSize.Height * CR_SCREENCAP_CHANNEL_COUNT * 4;
			m_BmpIntBuffer = (byte*)malloc(size);
			if (!m_BmpIntBuffer)
				return CRRES_ERR_MALLOC_FAILED;
			memset(m_BmpIntBuffer, 0, size);
		}

		/*
		 * Return
		 */
		m_FrameLock.unlock();
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
		area.Size = m_FrameSize.Size;
		if (!BitBlt(dstDC, 0, 0, area.Width, area.Height, srcDC, area.X, area.Y, SRCCOPY) || /* copy data to the win bmp */
			GetDIBits(dstDC, m_WinBmp, 0, area.Height, m_BmpIntBuffer, &m_WinBmpInfo, DIB_RGB_COLORS) != (int)area.Height) /* copy from win bmp to the int buffer*/
			return CRRES_ERR_WIN_UNKNOWN;

		/*
		 * Finishing windows stuff
		 */
		SelectObject(dstDC, oldDstDcObj);
		DeleteDC(dstDC);
		ReleaseDC(m_SrcHwnd, srcDC);

		/*
		 * Frame
		 */
		std::lock_guard<std::mutex> lock(m_FrameLock);
		if (!m_Frame) {
			m_Frame = CRFBmpCreateNormal(m_FrameSize.Width, m_FrameSize.Height, CR_SCREENCAP_CHANNEL_COUNT);
			if (!m_Frame)
				return CRRES_ERR_UTILS_FAILED_TO_CREATE_FBMP;
		}

		/*
		 * Converting data
		 */
		for (uint pxNo = 0; pxNo < area.Width * area.Height * CR_SCREENCAP_CHANNEL_COUNT; pxNo += 4)
		{
			m_Frame->Data[pxNo + 0] = ((float)m_BmpIntBuffer[pxNo + 2]) / 255.0f; /* R */
			m_Frame->Data[pxNo + 1] = ((float)m_BmpIntBuffer[pxNo + 1]) / 255.0f; /* G */
			m_Frame->Data[pxNo + 2] = ((float)m_BmpIntBuffer[pxNo + 0]) / 255.0f; /* B */
			m_Frame->Data[pxNo + 3] = 1.0f; /* A */
		}

		/*
		* Returning de la output
		*/
		return CRRES_SUCCESS;
	}

}}}

#endif // CRIA_OS_WIN
