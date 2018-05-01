#include <Cria.hpp>

#include <ctime>

#include <thread>

#include "tests/MathTests.h"
#include <windows.h>

#define BOI_TITLE "Binding of Isaac: Afterbirth+"
#define CON_TITLE "C:\\Users\\xFrednet\\My Projects\\VS Projects\\CriaAI\\bin\\Debug\\Sandbox.exe"

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

	CR_FLOAT_BITMAP* bmp = CRLoadFBmp("bmptest/test.bmp");

	for (int loop = 0; loop < 10; loop++) {
		clock_t timer = clock();

		CRDeleteFBmp(CRPoolBitmap(bmp, 2));
		CRDeleteFBmp(CRPoolBitmap(bmp, 3));
		CRDeleteFBmp(CRPoolBitmap(bmp, 9));

		std::cout << "Total: " << ((clock() - timer)) << std::endl;
	}
	CR_FLOAT_BITMAP* poolBmp2 = CRPoolBitmap(bmp, 2);
	CR_FLOAT_BITMAP* poolBmp3 = CRPoolBitmap(bmp, 3);
	CR_FLOAT_BITMAP* poolBmp9 = CRPoolBitmap(bmp, 9);

	CRSaveBitmap(bmp, "bmptest/test2.bmp");
	CRSaveBitmap(poolBmp2, "bmptest/pool2.bmp");
	CRSaveBitmap(poolBmp3, "bmptest/pool3.bmp");
	CRSaveBitmap(poolBmp9, "bmptest/pool9.bmp");

	CRDeleteFBmp(bmp);
	CRDeleteFBmp(poolBmp2);
	CRDeleteFBmp(poolBmp3);
	CRDeleteFBmp(poolBmp9);

}

void sleep(uint time)
{
	for (uint timer = 0; timer < time; timer++) {
		printf("Sleep: %i/%i \n", timer, time);
		std::this_thread::sleep_for(1s);
	}
}
void sleepSec(uint time)
{
	for (uint timer = 0; timer < time; timer++) {
		printf("SleepSec: %i/%i \n", timer, time);
		std::this_thread::sleep_for(1s);
	}
}
void sleepMs(uint time)
{
	for (uint timer = 0; timer < time; timer += 10) {
		//printf("SleepMs: %i/%i \n", timer, time);
		std::this_thread::sleep_for(10ms);
	}
}

void screenCap10Sec()
{
	sleep(10);


	crresult res = CRRES_SUCCESS;
	api::CRWindowPtr window = api::CRWindow::CreateInstance(BOI_TITLE, &res);
	if (CR_FAILED(res))
		return;
	api::CRScreenCapturer* capturer = api::CRScreenCapturer::CreateInstance(window, &res);

	CR_RECT cArea = window->getClientArea();
	std::cout << "window area: " << cArea.X << " " << cArea.Y << " " << cArea.Width << " " << cArea.Height << std::endl;
	
	uint capNo = 0;
	while (true) {
		capturer->grabFrame();
		CR_FLOAT_BITMAP* frame = capturer->getLastFrame();
		String capName = String("cap/Capturer") + std::to_string(capNo++) + String(".bmp");
		CRSaveBitmap(frame, capName.c_str());
		std::cout << "Frame: " << capName.c_str() << std::endl;

		sleep(10);
	}

	delete capturer;
}


bool TestInputSim()
{
	crresult result;
	api::CRWindowPtr window = api::CRWindow::CreateInstance(BOI_TITLE, &result);
	if (CR_FAILED(result))
		return false;
	api::CRInputSimulator* inputSim = api::CRInputSimulator::GetInstance(window, &result);

	POINT p;
	CR_VEC2I vec;
	CR_VEC2I move;

	sleep(2);

	std::cout << "vvv" << std::endl;
	inputSim->scrollMouse(-1);

	sleep(2);

	std::cout << "^^^" << std::endl;
	inputSim->scrollMouse(1);

	for (int i = 0; i < 10; i++) {
		GetCursorPos(&p);
		vec = inputSim->getMousePos();
		printf("ClientArea: X: %3i, Y: %3i | [WindowsArea: X: %3i, Y: %3i ] \n", vec.X, vec.Y, p.x, p.y);

		move.X = (rand() % 1000) - 500;
		move.Y = (rand() % 1000) - 500;
		printf("move: X: %i, Y: %i \n", move.X, move.Y);
		inputSim->moveMouse(move);

		GetCursorPos(&p);
		vec = inputSim->getMousePos();
		printf("ClientArea: X: %3i, Y: %3i | [WindowsArea: X: %3i, Y: %3i ] \n", vec.X, vec.Y, p.x, p.y);

		sleep(5);
	}

	delete inputSim;
	return true;
}

void keyInput(CR_KEY_ID keyID, bool down)
{
	std::cout << CRGetKeyIDName(keyID) << ((down) ? " [X]" : " [ ]") << std::endl;
}

int main(int argc, char* argv)
{
	cout << "Hello world" << endl;

	cout << "CR_VEC2 Test result " << TestVec2() << std::endl;
	cout << "########################################################" << std::endl;
	
	crresult r = api::CRInputLogger::InitInstance();
	api::CRInputLogger::AddKeyCallback(keyInput);

	api::CRWindowPtr targetWin = api::CRWindow::CreateInstance(BOI_TITLE, &r);
	printf("api::CRWindow::CreateInstance: %s \n", GetCRResultName(r).c_str());
	api::CRInputLogger::SetTargetWindow(targetWin);
	CR_RECT rect = (targetWin.get() ? targetWin->getClientArea() : CR_RECT(0, 0, 0, 0));
	std::cout << "Target Win: " << targetWin.get() << " " << rect.X << " " << rect.Y << " " << rect.Width << " " << rect.Height << std::endl;


	for (int timer = 0; timer < 100000; timer++)
	{
		api::CRInputLogger::Update();
		sleepMs(10);

		if (timer % 100 == 0)
		{
			printf("[%s] ", (api::CRInputLogger::GetMButtonState(CR_MBUTTON_LEFT)   ? "X" : " "));
			printf("[%s] ", (api::CRInputLogger::GetMButtonState(CR_MBUTTON_MIDDLE) ? "X" : " "));
			printf("[%s] ", (api::CRInputLogger::GetMButtonState(CR_MBUTTON_RIGHT)  ? "X" : " "));
			printf("POS(%4i, %4i) \n", api::CRInputLogger::GetMouseClientPos().X, api::CRInputLogger::GetMouseClientPos().Y);
		}
	}

	cin.get();
	return 0;
}