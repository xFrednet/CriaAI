#include <Cria.hpp>

#include <ctime>

#include <windows.h>
#include <thread>

#define BOI_TITLE "Binding of Isaac: Afterbirth+"

using namespace std;
using namespace cria_ai;
using namespace bmp_renderer;

void MatrixTest()
{
	CRMatrixf* a = LoadMatrixf("mat/a.mat");
	CRMatrixf* b = LoadMatrixf("mat/b.mat");
	if (!a) {
		a = CreateMatrixf(2, 2);
		FillMatrixRand(a);
		a->Data[0] = 1.0f;
		a->Data[1] = 2.0f;
		a->Data[2] = 3.0f;
		a->Data[3] = 4.0f;
		SaveMatrixf(a, "mat/a.mat");
		WriteMatrixf(a, "mat/a.txt");
		WriteMatrixfBmp(a, "mat/a.bmp");
	}
	if (!b) {
		b = CreateMatrixf(2, 2);
		FillMatrixRand(b);
		b->Data[0] = 0.0f;
		b->Data[1] = 1.0f;
		b->Data[2] = 0.0f;
		b->Data[3] = 0.0f;
		SaveMatrixf(b, "mat/b.mat");
		WriteMatrixf(b, "mat/b.txt");
		WriteMatrixfBmp(b, "mat/b.bmp");
	}

	CRMatrixf* c = Mul(a, b);
	SaveMatrixf(c, "mat/c.mat");
	WriteMatrixf(c, "mat/c.txt");
	WriteMatrixfBmp(c, "mat/c.bmp");

	FreeMatrixf(a);
	FreeMatrixf(b);
	FreeMatrixf(c);
}
void BmpTest()
{

	CR_FLOAT_BITMAP* bmp = LoadFBmp("bmptest/test.bmp");

	for (int loop = 0; loop < 10; loop++) {
		clock_t timer = clock();

		DeleteFBmp(PoolBitmap(bmp, 2));
		DeleteFBmp(PoolBitmap(bmp, 3));
		DeleteFBmp(PoolBitmap(bmp, 9));

		std::cout << "Total: " << ((clock() - timer)) << std::endl;
	}
	CR_FLOAT_BITMAP* poolBmp2 = PoolBitmap(bmp, 2);
	CR_FLOAT_BITMAP* poolBmp3 = PoolBitmap(bmp, 3);
	CR_FLOAT_BITMAP* poolBmp9 = PoolBitmap(bmp, 9);

	SaveBitmap(bmp, "bmptest/test2.bmp");
	SaveBitmap(poolBmp2, "bmptest/pool2.bmp");
	SaveBitmap(poolBmp3, "bmptest/pool3.bmp");
	SaveBitmap(poolBmp9, "bmptest/pool9.bmp");

	DeleteFBmp(bmp);
	DeleteFBmp(poolBmp2);
	DeleteFBmp(poolBmp3);
	DeleteFBmp(poolBmp9);

}

void sleep(uint time)
{
	for (uint timer = 0; timer < time; timer++) {
		printf("Sleep: %i/%i \n", timer, time);
		std::this_thread::sleep_for(1s);
	}
}

bmp_renderer::Bitmap* getScreenDC(HWND srcHwnd)
{

	/*
	* Getting the client size
	*/
	RECT clientFrame;
	GetClientRect(srcHwnd, &clientFrame);
	int width = clientFrame.right - clientFrame.left;
	int height = clientFrame.bottom - clientFrame.top;

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

	BitBlt(memDC, 0, 0, width, height, srcDC, 0, 0, SRCCOPY);

	/*
	* Getting the data
	*/
	{
		BITMAPINFO bmpOutInfo;
		memset(&bmpOutInfo, 0, sizeof(bmpOutInfo));
		bmpOutInfo.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
		bmpOutInfo.bmiHeader.biWidth = width;
		bmpOutInfo.bmiHeader.biHeight = -height;
		bmpOutInfo.bmiHeader.biPlanes = 1;
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
void screenCap10Sec()
{
	crresult res;

	HWND target = GetDesktopWindow();
	if (!target) {
		std::cout << "The window could not be found :/" << std::endl;
	}

	uint loop = 0;
	while (true) {
		sleep(10);

		std::string name = string("caps/Cap_") + std::to_string(loop) + std::string(".bmp");

		bmp_renderer::Bitmap* bmp = getScreenDC(target);
		bmp_renderer::SaveBitmap(bmp, name.c_str());
		bmp_renderer::DeleteBmp(bmp);

		std::cout << "Screen Cap: " << name.c_str() << std::endl;

		loop++;
	}


	std::cout << cria_ai::GetCRResultName(res) << std::endl;
}

int main(int argc, char* argv)
{
	cout << "Hello world" << endl;

	sleep(5);

	CR_RECT cArea = api::CRScreenCapturer::GetClientArea(BOI_TITLE);
	std::cout << "window area: " << cArea.X << " " << cArea.Y << " " << cArea.Width << " " << cArea.Height << std::endl;
	
	sleep(5);
	
	crresult res = CRRES_SUCCESS;
	api::CRScreenCapturer* capturer = api::CRScreenCapturer::CreateInstance(cArea, 0, &res);
	
	uint capNo = 0;
	while (true)
	{
		capturer->grabFrame();
		CR_FLOAT_BITMAP* frame = capturer->getLastFrame();
		String capName = String("cap/Capturer") + std::to_string(capNo++) + String(".bmp");
		SaveBitmap(frame, capName.c_str());
		std::cout << "Frame: " << capName.c_str() << std::endl;
		
		sleep(10);
	}

	delete capturer;
	cin.get();
	return 0;
}