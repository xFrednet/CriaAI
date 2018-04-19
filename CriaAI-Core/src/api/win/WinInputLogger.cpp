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
					getInstance()->keyEvent(st_hook.vkCode, false);
					break;
				case WM_KEYDOWN:
				case WM_SYSKEYDOWN:
					getInstance()->keyEvent(st_hook.vkCode, true);
					break;
				default:
					break;
			}
		}
		return CallNextHookEx(nullptr, message, wp, lp);
	}

	LRESULT CRWinInputLogger::HandleMouseHook(UINT message, WPARAM wp, LPARAM lp)
	{
		return CallNextHookEx(nullptr, message, wp, lp);
	}

	void CRWinInputLogger::keyEvent(uint32 keyID, bool isDown)
	{
		callKeyCBs((CR_KEY_ID)keyID, isDown);
	}

	CRWinInputLogger::CRWinInputLogger()
		: m_KeyboardHook(nullptr),
		m_KeyLayout(nullptr),
		m_MouseHook(nullptr)
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
		if (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
			TranslateMessage(&msg);
			DispatchMessage(&msg);
		}
	}
}}}

#endif //CRIA_OS_WIN