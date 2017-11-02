#include "WinWindow.hpp"
#include <iostream>
#include <map>

#define WM_SETWINDOWCLASS              WM_USER + 1

#if defined(_WIN32) || defined(_WIN64)
namespace bmp_renderer { namespace api {
	LRESULT CALLBACK WindowProc(HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam)
	{
		static std::map<HWND, WinWindow*> s_WindowClasses;

		switch (message) {
		case WM_SETWINDOWCLASS:
			s_WindowClasses[hwnd] = (WinWindow*)wParam;
			std::cout << "WM_SETWINDOWCLASS: hwnd:" << hwnd << " Window: " << ((WinWindow*)wParam) << std::endl;
			break;
		case WM_PAINT:
			if (s_WindowClasses[hwnd])
			{
				PAINTSTRUCT ps;
				HDC hDC = BeginPaint(hwnd, &ps);

				HDC memDC = CreateCompatibleDC(hDC);
				HGDIOBJ memDcOldObject = SelectObject(memDC, s_WindowClasses[hwnd]->m_HBitmap);

				BitBlt(hDC, 0, 0, s_WindowClasses[hwnd]->m_BmpWidth, s_WindowClasses[hwnd]->m_BmpHeight, memDC, 0, 0, SRCCOPY);

				SelectObject(memDC, memDcOldObject);
				DeleteDC(memDC);
				EndPaint(hwnd, &ps);

				std::cout << "Redraw" << std::endl;
			}
				
			break;
		case WM_CLOSE:
			if (!s_WindowClasses[hwnd])
				return DefWindowProc(hwnd, message, wParam, lParam);

			switch (s_WindowClasses[hwnd]->getOnExitAction()) {
			case WINDOW_ON_EXIT_DESTROY:
				return DefWindowProc(hwnd, message, wParam, lParam);
			case WINDOW_ON_EXIT_HIDE:
				ShowWindow(hwnd, SW_HIDE);
				return 0;
			case WINDOW_ON_EXIT_MINIMIZE:
				PostMessage(hwnd, WM_SYSCOMMAND, SC_MINIMIZE, 0);
				return 0;
			case WINDOW_ON_EXIT_DO_NOTHING:
				//I ain't doin nothin
				return 0;
			default:
				return DefWindowProc(hwnd, message, wParam, lParam);
			}
			break;
		case WM_DESTROY:
			std::cout << "WM_DESTROY" << std::endl;
			if (s_WindowClasses[hwnd])
			{
				s_WindowClasses[hwnd]->m_Hwnd = nullptr;
				s_WindowClasses.erase(hwnd);
			}
			
			PostQuitMessage(0);
			break;
		default:
			break;
		}

		return DefWindowProc(hwnd, message, wParam, lParam);
	}
	
	WinWindow::WinWindow(const char* name, unsigned width, unsigned height, WINDOW_ON_EXIT_ACTION onExit)
		: Window(onExit),
		m_Hwnd(nullptr),
		m_HBitmap(nullptr),
		m_BmpWidth(width),
		m_BmpHeight(height)
	{
		//init HWND
		{
			WNDCLASSEX wc;
			DWORD style = WS_OVERLAPPEDWINDOW;

			memset(&wc, 0, sizeof(WNDCLASSEX));

			wc.cbSize = sizeof(WNDCLASSEX);
			wc.style = CS_HREDRAW | CS_VREDRAW;
			wc.lpszClassName = name;
			wc.lpfnWndProc = WindowProc;

			RECT size = { 0, 0, LONG(width), LONG(height) };
			AdjustWindowRect(&size, style, false);

			RegisterClassEx(&wc);

			m_Hwnd = CreateWindowEx(NULL,
				name,
				name,
				style,
				100,
				100,
				size.right - size.left,
				size.bottom - size.top,
				NULL, NULL, NULL, NULL);

			PostMessage(getHandle(), WM_SETWINDOWCLASS, (WPARAM)this, 0);
		}
		//init HBitmap
		{
			memset(&m_BmpInfo, 0, sizeof(BITMAPINFO));
			m_BmpInfo.bmiHeader.biSize = sizeof(BITMAPINFO);
			m_BmpInfo.bmiHeader.biWidth = width;
			m_BmpInfo.bmiHeader.biHeight = height;
			m_BmpInfo.bmiHeader.biPlanes = 1;
			m_BmpInfo.bmiHeader.biBitCount = 32;
			m_BmpInfo.bmiHeader.biCompression = BI_RGB;

			HDC hDC = GetDC(getHandle());
			RGBQUAD* pPixels = nullptr;
			m_HBitmap = CreateDIBSection(hDC, &m_BmpInfo, DIB_RGB_COLORS, (void**)&pPixels, NULL, 0);

			Color setColor(0xff, 0, 0xff, 0xaa);
			for (unsigned pixelNr = 0; pixelNr < width * height; pixelNr++) {
				memcpy(&pPixels[pixelNr], &setColor, sizeof(RGBQUAD));
			}
			ReleaseDC(getHandle(), hDC);
		}
		
		ShowWindow(getHandle(), SW_SHOW);
	}
	WinWindow::~WinWindow()
	{
		destroy();
		if (m_HBitmap) {
			DeleteObject(m_HBitmap);
			m_HBitmap = nullptr;
		}
	}

	bool WinWindow::update()
	{
		if (!isValid())
			return false;

		MSG msg;
		if (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
			TranslateMessage(&msg);
			DispatchMessage(&msg);

			if (msg.message == WM_QUIT) {
				std::cout << "WM_QUIT m_Hwnd: " << m_Hwnd << std::endl;
				m_Hwnd = nullptr;
				return false;
			}
		}
		return true;
	}

	void WinWindow::loadBitmap(const Bitmap bitmap)
	{
		HDC hDC = GetDC(m_Hwnd);

		SetDIBits(hDC, m_HBitmap, 0,
			((bitmap->HEIGHT < m_BmpHeight) ? bitmap->HEIGHT : m_BmpHeight),
			bitmap->Data, &m_BmpInfo, DIB_RGB_COLORS);

		ReleaseDC(m_Hwnd, hDC);
		RedrawWindow(m_Hwnd, nullptr, nullptr, RDW_INVALIDATE);
	}

	void WinWindow::setVisibility(bool visible)
	{
		ShowWindow(m_Hwnd, (visible) ? SW_SHOW : SW_HIDE);
	}
	bool WinWindow::getVisibility() const
	{
		return IsWindowVisible(m_Hwnd);
	}

	void WinWindow::destroy()
	{
		DestroyWindow(m_Hwnd);
		m_Hwnd = nullptr;
	}

	bool WinWindow::isValid() const
	{
		return (m_Hwnd) ? true : false;
	}
}}
#endif