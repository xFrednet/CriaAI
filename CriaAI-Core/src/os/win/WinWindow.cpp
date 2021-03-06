/******************************************************************************
* Cria  - The worst artificial intelligence on the market.                    *
*         <https://github.com/xFrednet/CriaAI>                                *
*                                                                             *
* =========================================================================== *
* Copyright (C) 2017, 2018, xFrednet <xFrednet@gmail.com>                     *
*                                                                             *
* This software is provided 'as-is', without any express or implied warranty. *
* In no event will the authors be held liable for any damages arising from    *
* the use of this software.                                                   *
*                                                                             *
* Permission is hereby granted, free of charge, to anyone to use this         *
* software for any purpose, including the rights to use, copy, modify,        *
* merge, publish, distribute, sublicense, and/or sell copies of this          *
* software, subject to the following conditions:                              *
*                                                                             *
*   1.  The origin of this software must not be misrepresented; you           *
*       must not claim that you wrote the original software. If you           *
*       use this software in a product, an acknowledgment in the              *
*       product documentation would be greatly appreciated but is not         *
*       required                                                              *
*                                                                             *
*   2.  Altered source versions should be plainly marked as such, and         *
*       must not be misrepresented as being the original software.            *
*                                                                             *
*   3.  This code should not be used for any military or malicious            *
*       purposes.                                                             *
*                                                                             *
*   4.  This notice may not be removed or altered from any source             *
*       distribution.                                                         *
*                                                                             *
******************************************************************************/
#include "WinWindow.h"

namespace cria_ai { namespace os { namespace win {
	CRWinWindow::CRWinWindow(const String& title)
		: CRWindow(title),
		m_Hwnd(nullptr)
	{
	}

	crresult CRWinWindow::init(const String& title)
	{
		/*
		 * Getting the HWND
		 */
		if (title.empty()) 
			m_Hwnd = GetDesktopWindow(); 
		else 
			m_Hwnd = FindWindow(nullptr, title.c_str());
		
		/*
		 * Error check
		 */
		if (!m_Hwnd)
			return CRRES_ERR_OS_WINDOW_TITLE_NOT_FOUND;

		/*
		 * Return
		 */
		return CRRES_OK_OS;
	}

	bool CRWinWindow::isFocussed() const
	{
		return (GetForegroundWindow() == m_Hwnd);
	}

	CR_RECT CRWinWindow::getClientArea() const
	{
		WINDOWINFO winInfo;
		if (!GetWindowInfo(m_Hwnd, &winInfo))
			return CR_RECT{0, 0, 0, 0};
		RECT winCArea = winInfo.rcClient;

		/*
		* Translating the area
		*/
		CR_RECT cArea;
		cArea.X      = (int)winCArea.left;
		cArea.Y      = (int)winCArea.top;
		cArea.Width  = (uint)(winCArea.right - winCArea.left);
		cArea.Height = (uint)(winCArea.bottom - winCArea.top);

		return cArea;
	}

	CR_VEC2I CRWinWindow::getCorrectResizeSize(uint width, uint height) const
	{
		LONG hwndStyle = GetWindowLong(m_Hwnd, GWL_STYLE);

		RECT size = {0, 0, (LONG)width, (LONG)height};
		AdjustWindowRect(&size, hwndStyle, false);

		return CR_VEC2I(size.right - size.left, size.bottom - size.top);
	}
	crresult CRWinWindow::setPos(int x, int y)
	{
		if (!SetWindowPos(m_Hwnd, nullptr, (int)x, (int)y, 0, 0, 
			SWP_NOSIZE | SWP_NOACTIVATE | SWP_NOZORDER))
			return CRRES_ERR_OS_WINDOW_RESIZE_FAILED;

		return CRRES_OK;
	}
	crresult CRWinWindow::setSize(uint width, uint height)
	{
		CR_VEC2I size = getCorrectResizeSize(width, height);

		if (!SetWindowPos(m_Hwnd, nullptr, 0, 0, size.X, size.Y,
			SWP_NOMOVE | SWP_NOACTIVATE | SWP_NOZORDER))
			return CRRES_ERR_OS_WINDOW_RESIZE_FAILED;

		return CRRES_OK;
	}
	crresult CRWinWindow::setClientArea(const CR_RECT& bounds)
	{
		CR_VEC2I size = getCorrectResizeSize(bounds.Width, bounds.Height);

		if (!SetWindowPos(m_Hwnd, nullptr, bounds.X, bounds.Y, size.X, size.Y,
			SWP_NOACTIVATE | SWP_NOZORDER))
			return CRRES_ERR_OS_WINDOW_RESIZE_FAILED;

		return CRRES_OK;
	}


	HWND CRWinWindow::getHWND()
	{
		return m_Hwnd;
	}
}}}