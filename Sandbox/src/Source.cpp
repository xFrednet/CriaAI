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


void screenCap10Sec()
{
	sleep(10);

	CR_RECT cArea = api::CRScreenCapturer::GetClientArea(BOI_TITLE);
	std::cout << "window area: " << cArea.X << " " << cArea.Y << " " << cArea.Width << " " << cArea.Height << std::endl;

	crresult res = CRRES_SUCCESS;
	api::CRScreenCapturer* capturer = api::CRScreenCapturer::CreateInstance(cArea, 0, &res);

	uint capNo = 0;
	while (true) {
		capturer->grabFrame();
		CR_FLOAT_BITMAP* frame = capturer->getLastFrame();
		String capName = String("cap/Capturer") + std::to_string(capNo++) + String(".bmp");
		SaveBitmap(frame, capName.c_str());
		std::cout << "Frame: " << capName.c_str() << std::endl;

		sleep(10);
	}

	delete capturer;
}

int main(int argc, char* argv)
{
	cout << "Hello world" << endl;

	cout << "CR_VEC2 Test result " << TestVec2() << std::endl;
	cout << "########################################################" << std::endl;

	sleep(3);

	crresult result;
	api::CRInputSimulator* inputSim = api::CRInputSimulator::GetInstance("", &result);

	//const char* abc = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
	//for (uint no = 0; no < 26; no++)
	//{
	//	inputSim->clickKey(abc[no]); // 'A'
	//}

	//const char* keys = "WASD ";
	//
	//for (uint keyNo = 0; keyNo < strlen(keys); keyNo++)
	//{
	//	inputSim->setKeyState(keys[keyNo], true);
	//	sleep(3);
	//	inputSim->setKeyState(keys[keyNo], false);
	//	sleep(2);
	//}

	/*for (int loopNo = 0; loopNo < 10; loopNo++)
	{
		inputSim->setButtonState(CR_MOUSE_LEFT, true);
		sleep(2);
		inputSim->setButtonState(CR_MOUSE_LEFT, false);
		sleep(3);
	}*/
	POINT p;
	CR_VEC2I vec;
	CR_VEC2I move;

	sleep(2);

	std::cout << "vvv" << std::endl;
	inputSim->scrollMouse(-1);

	sleep(10);

	std::cout << "^^^" << std::endl;
	inputSim->scrollMouse(1);

	for (int i = 0 ; i < 10; i++)
	{
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
	cin.get();
	return 0;
}