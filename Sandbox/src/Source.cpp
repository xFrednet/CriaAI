#include <Cria.hpp>

#include <ctime>

#include <thread>

#include "tests/MathTests.h"
#include <windows.h>

#define BOI_TITLE                      "Binding of Isaac: Afterbirth+"
#define BOI_BASE_WIDTH                 512
#define BOI_BASE_HEIGHT                288
#define BOI_SCALE                      2

#define CON_TITLE "C:\\Users\\xFrednet\\My Projects\\VS Projects\\CriaAI\\bin\\Debug\\Sandbox.exe"

using namespace std;
using namespace cria_ai;
using namespace bmp_renderer;


void keyInput(CR_KEY_ID keyID, bool down)
{
	std::cout << CRGetKeyIDName(keyID) << ((down) ? " [X]" : " [ ]") << std::endl;
}

int main(int argc, char* argv)
{
	cout << "Hello world" << endl;

	cout << "CR_VEC2 Test result " << TestVec2() << std::endl;
	cout << "########################################################" << std::endl;
	
	crresult r;

	/*
	 * CROSContext
	 */
	r = os::CROSContext::InitInstance();
	printf("os::CROSContext::InitInstance: %s \n", CRGetCRResultName(r).c_str());
	os::CROSContext::Sleep(10);

	/*
	 * CRWindow
	 */
	os::CRWindowPtr targetWin = os::CRWindow::CreateInstance(BOI_TITLE, &r);
	printf("os::CRWindow::CreateInstance: %s \n", CRGetCRResultName(r).c_str());
	if (!targetWin.get())
	{
		printf("GOODBYE [ENTER]\n>");
		std::cin.get();
		return 1;
	}

	/*
	 * Debug info
	 */
	targetWin->setPos(50, 50);
	CR_RECT rect = (targetWin.get() ? targetWin->getClientArea() : CR_RECT(0, 0, 0, 0));
	std::cout << "Target Win: " << targetWin.get() << " " << rect.X << " " << rect.Y << " " << rect.Width << " " << rect.Height << std::endl;
	
	targetWin->setClientArea(CR_RECT{50, 50, BOI_BASE_WIDTH * BOI_SCALE, BOI_BASE_HEIGHT * BOI_SCALE});

	rect = (targetWin.get() ? targetWin->getClientArea() : CR_RECT(0, 0, 0, 0));
	std::cout << "Target Win: " << targetWin.get() << " " << rect.X << " " << rect.Y << " " << rect.Width << " " << rect.Height << std::endl;

	/*
	 * CRInputLogger
	 */
	r = os::CRInputLogger::InitInstance();
	printf("os::CRInputLogger::InitInstance: %s \n", CRGetCRResultName(r).c_str());
	os::CRInputLogger::SetTargetWindow(targetWin);
	os::CRInputLogger::AddKeyCallback(keyInput);


	for (int timer = 0; timer < 101; timer++)
	{
		os::CRInputLogger::Update();
		os::CROSContext::Sleep(0, 10);

		if (timer % 100 == 0)
		{
			printf("[%s] ", (os::CRInputLogger::GetMButtonState(CR_MBUTTON_LEFT)   ? "X" : " "));
			printf("[%s] ", (os::CRInputLogger::GetMButtonState(CR_MBUTTON_MIDDLE) ? "X" : " "));
			printf("[%s] ", (os::CRInputLogger::GetMButtonState(CR_MBUTTON_RIGHT)  ? "X" : " "));
			printf("POS(%4i, %4i) \n", os::CRInputLogger::GetMouseClientPos().X, os::CRInputLogger::GetMouseClientPos().Y);
		}
	}

	os::CRInputLogger::TerminateInstance();
	os::CROSContext::TerminateInstance();

	cin.get();
	return 0;
}