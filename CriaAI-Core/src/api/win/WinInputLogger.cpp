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
#include "WinInputLogger.h"

#ifdef CRIA_OS_WIN

#define getInstance()                  ((cria_ai::api::win::CRWinInputLogger*)s_Instance)

namespace cria_ai { namespace api { namespace win {
	LRESULT CRWinInputLogger::HandleKeyboardHook(UINT message, WPARAM wp, LPARAM lp)
	{
		if (s_Instance && message == HC_ACTION) {

			KBDLLHOOKSTRUCT st_hook = *((KBDLLHOOKSTRUCT*)lp);

			switch (wp) {
				case WM_KEYUP:
				case WM_SYSKEYUP:
					getInstance()->processKey((CR_KEY_ID)st_hook.vkCode, false);
					break;
				case WM_KEYDOWN:
				case WM_SYSKEYDOWN:
					getInstance()->processKey((CR_KEY_ID)st_hook.vkCode, true);
					break;
				default:
					break;
			}
		}
		return CallNextHookEx(nullptr, message, wp, lp);
	}

	LRESULT CRWinInputLogger::HandleMouseHook(UINT message, WPARAM wp, LPARAM lp)
	{
		if (s_Instance && message == HC_ACTION) 
		{
			MSLLHOOKSTRUCT hook = *((MSLLHOOKSTRUCT*)lp);

			switch (wp)
			{
				case WM_MOUSEWHEEL:
					getInstance()->processMWheel(((short)HIWORD(hook.mouseData)) / WHEEL_DELTA);
					break;

				/*
				 * Mouse movement
				 */
				case WM_MOUSEMOVE:
					getInstance()->processNewMousePos(CR_VEC2I(hook.pt.x, hook.pt.y));
					break;
				
				/*
				 * Mouse button events
				 */
				//left
				case WM_LBUTTONDOWN:
				case WM_LBUTTONUP:
					getInstance()->processMButton(CR_MBUTTON_LEFT, (wp == WM_LBUTTONDOWN));
					break;
				//middle
				case WM_MBUTTONDOWN:
				case WM_MBUTTONUP:
					getInstance()->processMButton(CR_MBUTTON_MIDDLE, (wp == WM_MBUTTONDOWN));
					break;
				//right
				case WM_RBUTTONDOWN:
				case WM_RBUTTONUP:
					getInstance()->processMButton(CR_MBUTTON_RIGHT, (wp == WM_RBUTTONDOWN));
					break;

				//other buttons
				case WM_XBUTTONDOWN:
					getInstance()->processMButton((CR_MBUTTON_ID)(short)HIWORD(hook.mouseData), true);
					break;
				case WM_XBUTTONUP:
					getInstance()->processMButton((CR_MBUTTON_ID)(short)HIWORD(hook.mouseData), false);
					break;
				default:
					std::cout << wp << std::endl;
					break;
			}
		}

		return CallNextHookEx(nullptr, message, wp, lp);
	}

	CRWinInputLogger::CRWinInputLogger()
		: m_KeyboardHook(nullptr),
		m_KeyLayout(nullptr),
		m_MouseHook(nullptr),
		m_OldMousePos(win::GetMousePos())
	{
	}
	CRWinInputLogger::~CRWinInputLogger()
	{
		UnhookWindowsHookEx(m_KeyboardHook);
		UnhookWindowsHookEx(m_MouseHook);
	}

	crresult CRWinInputLogger::init()
	{
		/*
		 * If a 64-bit application installs a global hook on 64-bit Windows, the 64-bit hook is injected into each 64-bit process, while all 32-bit processes use a callback to the hooking application. 
		 * ~ https://msdn.microsoft.com/en-us/library/windows/desktop/ms644990(v=vs.85).aspx
		 * So this should work theoretically
		 */
		m_KeyboardHook = SetWindowsHookEx(WH_KEYBOARD_LL, (HOOKPROC)CRWinInputLogger::HandleKeyboardHook, nullptr, 0);
		if (!m_KeyboardHook)
			return CRRES_ERR_WIN_FAILED_TO_INSTALL_HOCK;
		m_KeyLayout = GetKeyboardLayout(0);
		if (!m_KeyLayout)
			return CRRES_ERR_WIN_COULD_NOT_GET_KEY_LAYOUT;

		m_MouseHook = SetWindowsHookEx(WH_MOUSE_LL, (HOOKPROC)CRWinInputLogger::HandleMouseHook, nullptr, 0);
		if (!m_MouseHook)
			return CRRES_ERR_WIN_FAILED_TO_INSTALL_HOCK;

		return CRRES_OK_WIN;
	}

	void CRWinInputLogger::update()
	{
		MSG msg;
		while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
			TranslateMessage(&msg);
			DispatchMessage(&msg);
		}
	}

	void CRWinInputLogger::processNewMousePos(CR_VEC2I newPos)
	{
		processMMove(newPos, newPos.X - m_OldMousePos.X, newPos.Y - m_OldMousePos.Y);

		m_OldMousePos = newPos;
	}

	void CRWinInputLogger::newTargetWindow(const String& title)
	{
		m_ClientArea = win::GetClientArea(title);
	}

}}}

#endif //CRIA_OS_WIN