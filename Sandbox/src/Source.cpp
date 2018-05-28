#include <Cria.hpp>
#include "src/util/FloatBitmap.h"

#include <thread>

#include "tests/MathTests.h"
#include <AsyncInfo.h>

#define BOI_TITLE                      "Binding of Isaac: Afterbirth+"
#define BOI_BASE_WIDTH                 512
#define BOI_BASE_HEIGHT                288
#define BOI_SCALE                      2
#define BOI_SAMPLE_SIZE                2

#define CON_TITLE "C:\\Users\\xFrednet\\My Projects\\VS Projects\\CriaAI\\bin\\Debug\\Sandbox.exe"

//#define LOG_TIME

using namespace std;
using namespace cria_ai;
using namespace bmp_renderer;
using namespace network;

void setConCursorPos(COORD pos = {0, 0})
{
	HANDLE hCon = GetStdHandle(STD_OUTPUT_HANDLE);
	SetConsoleCursorPosition(hCon, pos);
}
COORD getConCursorPos()
{
	HANDLE hCon = GetStdHandle(STD_OUTPUT_HANDLE);
	CONSOLE_SCREEN_BUFFER_INFO csbi;
	if (!GetConsoleScreenBufferInfo(
		GetStdHandle(STD_OUTPUT_HANDLE),
		&csbi
	))
		return {0, 0};

	return csbi.dwCursorPosition;
}

CRMatrixf* bmpToMat(CR_FLOAT_BITMAP* bmp)
{
	CRMatrixf* mat = CRCreateMatrixf(1, bmp->Width * bmp->Height);
	memcpy(mat->Data, bmp->Data, sizeof(float) * mat->Rows);

	return mat;
}
CRMatrixf* processBOIFrame(CR_FLOAT_BITMAP* inFrame)
{
	if (!inFrame || inFrame->Width != BOI_BASE_WIDTH * BOI_SCALE || inFrame->Height != BOI_BASE_HEIGHT * BOI_SCALE)
	{
		std::cout << "    [INFO] processBOIFrame: something is wrong" << std::endl;
		return nullptr;
	}

#ifdef LOG_TIME
	StopWatch timer;
#endif
	CR_FLOAT_BITMAP* fpp1Out = CRConvertToFloatsPerPixel(inFrame, 1);
#ifdef LOG_TIME
	std::cout << "    [INFO] fpp1Out      : " << timer.getTimeMSSinceStart() << std::endl;
	timer.start();
#endif

	CR_FLOAT_BITMAP* scaleOut = CRScaleFBmpDown(fpp1Out, BOI_SCALE);
#ifdef LOG_TIME
	std::cout << "    [INFO] scaleOut     : " << timer.getTimeMSSinceStart() << std::endl;
	timer.start();
#endif

	CR_FLOAT_BITMAP* poolOut = CRPoolBitmap(scaleOut, BOI_SAMPLE_SIZE);
#ifdef LOG_TIME
	std::cout << "    [INFO] poolOut      : " << timer.getTimeMSSinceStart() << std::endl;
	timer.start();
#endif

	CRMatrixf* mat = bmpToMat(poolOut);

	CRDeleteFBmp(fpp1Out);
	CRDeleteFBmp(scaleOut);
	CRDeleteFBmp(poolOut);

	return mat;
}
CRNeuronNetwork* createBOINetwork(CRNeuronLayerPtr& outputLayer)
{
	std::cout << " > createBOINetwork" << std::endl;
	CRNeuronNetwork* network = new CRNeuronNetwork;
	
	/*
	 * Layer 1
	 */
	CRNeuronLayerPtr layer1 = make_shared<CRNeuronLayer>(nullptr, (BOI_BASE_WIDTH / BOI_SAMPLE_SIZE * BOI_BASE_HEIGHT / BOI_SAMPLE_SIZE));// 36864 Neurons :O
	layer1->setActivationFunc(paco::CRSigmoid, paco::CRSigmoidInv);
	network->addLayer(layer1);

	/*
	 * Layer 2
	 */
	CRNeuronLayerPtr layer2 = make_shared<CRNeuronLayer>(layer1.get(), 100);
	layer2->setActivationFunc(paco::CRSigmoid, paco::CRSigmoidInv);
	network->addLayer(layer2);

	/*
	* Layer 3
	*/
	CRNeuronLayerPtr layer3 = make_shared<CRNeuronLayer>(layer2.get(), 100);
	layer3->setActivationFunc(paco::CRSigmoid, paco::CRSigmoidInv);
	network->addLayer(layer3);

	/*
	* Layer 4
	*/
	outputLayer = make_shared<CRNeuronLayer>(layer3.get(), 12);//WASD[^][<][v][>][ ][STRG]EQ
	outputLayer->setActivationFunc(paco::CRSigmoid, paco::CRSigmoidInv);
	network->addLayer(outputLayer);

	/*
	 * Finish and return
	 */
	network->initRandom();

	std::cout << " < createBOINetwork" << std::endl;
	return network;
}
void printBOIOutput(CRNWMat const* mat)
{
	if (mat->Cols != 1 || mat->Rows != 12)
		return;
	String buttons[] = {"W", "A", "S", "D", "UP", "LEFT", "DOWN", "RIGHT", " ", "STRG", "E", "Q"};

	for (uint index = 0; index < 12; index++)
	{
		printf("BOI Button: [%5s] [%3f]\n", buttons[index].c_str(), mat->Data[index]);
	}
}
void testBOINetwork()
{
	std::cout << "> testBOINetwork" << std::endl;

	crresult result;
	/*
	 * Init
	 */
	// window
	os::CRWindowPtr window = os::CRWindow::CreateInstance(BOI_TITLE, &result);
	if (CR_FAILED(result))
	{
		printf(" [ERROR] os::CRWindow::CreateInstance failed!! Exit");
		return;
	}
	window->setClientArea(CR_RECT{50, 50, BOI_BASE_WIDTH * BOI_SCALE, BOI_BASE_HEIGHT * BOI_SCALE});
	
	// network
	CRNeuronLayerPtr outputLayer;
	CRNeuronNetwork* network = createBOINetwork(outputLayer);
	
	// screen capturer
	os::CRScreenCapturer* capturer = os::CRScreenCapturer::CreateInstance(window, &result);
	if (CR_FAILED(result))
	{
		printf(" [ERROR] os::CRScreenCapturer::CreateInstance failed !! Exit");
		return;
	}

	std::cout << " [INFO] = init finish" << std::endl;

	std::cout << std::endl;
	std::cout << " [INFO] Press X to exit" << std::endl;
	std::cout << std::endl;

	/*
	 * Loop
	 */
	COORD conCursorPos = getConCursorPos();
	uint frameNo = 0;
	StopWatch timer;
	bool running = true;
	uint iterations = 0;
	while (running)
	{
		/*
		 * Exit check
		 */
		if (GetAsyncKeyState('X'))
		{
			running = false;
			printf("\n [EXIT] exit because u wanted it \n\n");
			break;
		}

		setConCursorPos(conCursorPos);

		if (GetAsyncKeyState('O'))
		{
			String frameName = String("frame/frame") + std::to_string(frameNo++) + String(".bmp");
			CRSaveBitmap(capturer->getLastFrame(), frameName.c_str());
		}

		/*
		 * network
		 */
		// retrieve frame
		capturer->grabFrame();
		CR_FLOAT_BITMAP* frame = capturer->getLastFrame();

		// frame processing
		CRMatrixf* data = processBOIFrame(frame);
	
		// process data
		network->process(data);
		CRFreeMatrixf(data);

		// print result
		printBOIOutput(outputLayer->getOutput());
		
		/*
		 * Info 
		 */
		iterations++;
		if (timer.getTimeMSSinceStart() >= 1000)
		{
			std::cout << std::endl;
			std::cout << " [INFO] IPS: " << iterations << ", average time[ms] :" << (timer.getTimeMSSinceStart() / iterations) << std::endl;
			std::cout << std::endl;

			timer.start();
			iterations = 0;
		}
		
	}

	delete network;
	delete capturer;

	std::cout << "< testBOINetwork" << std::endl;
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

	for (uint sleep = 10; sleep > 0; sleep--)
	{
		os::CROSContext::Sleep(1);
		std::cout << "[INFO] network start in: " << sleep << std::endl;
	}

	/*
	 * Network test
	 */
	testBOINetwork();

	/*
	 * cleanup
	 */
	os::CROSContext::TerminateInstance();

	return 0;
}