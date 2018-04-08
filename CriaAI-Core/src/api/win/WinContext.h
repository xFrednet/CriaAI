#pragma once

#include "../../Common.hpp"

#include <windows.h>

namespace cria_ai { namespace api { namespace win {
	
	inline HWND FindHWND(const String& title)
	{
		return FindWindow(nullptr, title.c_str());
	}

	inline CR_RECT GetClientArea(HWND hwnd)
	{
		/*
		* Finding the window
		*/
		if (!hwnd)
			return CR_RECT{0, 0, 0, 0};

		/*
		* Retrieving area
		*/
		WINDOWINFO winInfo;
		if (!GetWindowInfo(hwnd, &winInfo))
			return CR_RECT{0, 0, 0, 0};
		RECT winCArea = winInfo.rcClient;

		/*
		 * Translating the area
		 */
		CR_RECT cArea;
		cArea.X = (uint)winCArea.left;
		cArea.Y = (uint)winCArea.top;
		cArea.Width = (uint)(winCArea.right - winCArea.left);
		cArea.Height = (uint)(winCArea.bottom - winCArea.top);

		return cArea;
	}
	inline CR_RECT GetClientArea(const String& title)
	{
		return GetClientArea(FindHWND(title));
	}

	inline CR_RECT GetVirtualScreenClientArea()
	{
		CR_RECT vArea;

		vArea.X      = GetSystemMetrics(SM_XVIRTUALSCREEN);
		vArea.Y      = GetSystemMetrics(SM_YVIRTUALSCREEN);
		vArea.Width  = GetSystemMetrics(SM_CXVIRTUALSCREEN);
		vArea.Height = GetSystemMetrics(SM_CYVIRTUALSCREEN);

		return vArea;
	}
}}}
