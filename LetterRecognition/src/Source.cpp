#pragma once

#if 1
#include <Cria.hpp>

#include <windows.h>

#include <iostream>
#include <thread>

bmp_renderer::Bitmap* getScreen(HWND srcHwnd) {
	
	/*
	 * Getting the client size
	 */
	RECT clientFrame;
	GetClientRect(srcHwnd, &clientFrame);
	int width  = clientFrame.right  - clientFrame.left; 
	int height = clientFrame.bottom - clientFrame.top; 

	std::cout << "RECT: T(" << clientFrame.top << "), B(" << clientFrame.bottom << ") \n";
	std::cout << "RECT: L(" << clientFrame.left << "), R(" << clientFrame.right << ") \n";
	std::cout << "width(" << width << ")" << ", height(" << height << ") \n";
	/*
	 * Create output
	 */
	bmp_renderer::Bitmap* bmp = bmp_renderer::CreateBmp(width, height);
	
	/*
	 * Windows stuff :/
	 */
	HDC srcDC = GetDC(srcHwnd);
	HDC memDC = CreateCompatibleDC(srcDC);
	HBITMAP winBmp = CreateCompatibleBitmap(srcDC, width, height);
	HGDIOBJ oldMemDcObj = SelectObject(memDC, winBmp);

	BitBlt(memDC, 0, 0, width, height, srcDC, 0, 0, SRCCOPY | CAPTUREBLT);
	
	/*
	 * Getting the data
	 */
	{
		BITMAPINFO bmpOutInfo;
		memset(&bmpOutInfo, 0, sizeof(bmpOutInfo));
		bmpOutInfo.bmiHeader.biSize     = sizeof(BITMAPINFOHEADER);
		bmpOutInfo.bmiHeader.biWidth    = width;
		bmpOutInfo.bmiHeader.biHeight   = -height;
		bmpOutInfo.bmiHeader.biPlanes   = 1;
		bmpOutInfo.bmiHeader.biBitCount = 32;

		GetDIBits(memDC, winBmp, 0, height, bmp->Data, &bmpOutInfo, DIB_RGB_COLORS);
	}
	
	/*
	 * Finishing windows stuff :)
	 */
	SelectObject(memDC, oldMemDcObj);
	DeleteDC(memDC);
	DeleteObject(winBmp);

	/*
	 * Returning de la output 
	 */
	return bmp;
}

void sleep(int sleepTime)
{
	for (int time = sleepTime; time > 0; time--)
	{
		std::this_thread::sleep_for(std::chrono::seconds(1));
		std::cout << "Countdown: " << time << "/" << sleepTime << std::endl;
	}
}

int main()
{
	std::cout << "Hello \n" << std::endl;

	bmp_renderer::Bitmap* bmp;
	sleep(10);

	int snapNr = 0;
	while (true)
	{

		String snap = String("snap_") + std::to_string(snapNr) + String(".bmp");
		std::cout << snap.c_str() << std::endl;

		bmp = getScreen(GetDesktopWindow());
		bmp_renderer::SaveBitmap(bmp, snap.c_str());
		bmp_renderer::DeleteBmp(bmp);
		
		snapNr++;
		sleep(10);
	}

	std::cin.get();
	return 0;
}

#endif
