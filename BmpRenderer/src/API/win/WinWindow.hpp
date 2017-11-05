#pragma once

#include "API/Window.hpp"

#if defined(_WIN32) || defined(_WIN64)

#include <windows.h>

namespace bmp_renderer { namespace api {
	
	LRESULT CALLBACK WindowProc(HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam);

	class WinWindow : public Window
	{
	private:
		friend LRESULT CALLBACK WindowProc(HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam);

		HWND m_Hwnd;

		BITMAPINFO m_BmpInfo;
		HBITMAP m_HBitmap;
		unsigned m_BmpWidth;
		unsigned m_BmpHeight;

	public:
		WinWindow(const char* name, unsigned width, unsigned height, WINDOW_ON_EXIT_ACTION onExit);
		~WinWindow();

		bool update();
		void loadBitmap(const Bitmap bitmap);

		void setVisibility(bool visible);
		bool getVisibility() const;

		void destroy();
		bool isValid() const;

		inline HWND getHandle() { return m_Hwnd; }
		inline HBITMAP getBitmapHandle() { return m_HBitmap; }
	};

}}
#endif