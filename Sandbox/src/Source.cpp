#include <Cria.hpp>

#include <AsyncInfo.h>

#include "tests/MathTests.h"
#include "src/network/Backprop.h"

#define BOI_TITLE                      "Binding of Isaac: Afterbirth+"
#define BOI_BASE_WIDTH                 512
#define BOI_BASE_HEIGHT                288
#define BOI_SCALE                      2
#define BOI_SAMPLE_SIZE                2
#define BOI_BATCH_SIZE                 500

#define CON_TITLE "C:\\Users\\xFrednet\\My Projects\\VS Projects\\CriaAI\\bin\\Debug\\Sandbox.exe"

#define LOG_TIME

using namespace std;
using namespace cria_ai;
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

/*
 * BOI network
 */
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

CR_FBMP* g_Fpp1Out = CRFBmpCreate(BOI_BASE_WIDTH * BOI_SCALE, BOI_BASE_HEIGHT * BOI_SCALE, 1);
CR_FBMP* g_ScaleOut = CRFBmpCreate(BOI_BASE_WIDTH / BOI_SAMPLE_SIZE, BOI_BASE_HEIGHT / BOI_SAMPLE_SIZE, 1);
CR_MATF* g_MatOut = CRMatFCreate(1, (BOI_BASE_WIDTH / BOI_SAMPLE_SIZE) * (BOI_BASE_HEIGHT / BOI_SAMPLE_SIZE));
CR_MATF* processBOIFrame(CR_FBMP* inFrame)
{
	if (!inFrame || inFrame->Width != BOI_BASE_WIDTH * BOI_SCALE || inFrame->Height != BOI_BASE_HEIGHT * BOI_SCALE)
	{
		std::cout << "    [INFO] processBOIFrame: something is wrong" << std::endl;
		return nullptr;
	}

#ifdef LOG_TIME
	StopWatch timer;
#endif
	paco::CRFBmpConvertToFPP(inFrame, g_Fpp1Out);
#ifdef LOG_TIME
	std::cout << "    [INFO] fpp1Out      : " << timer.getTimeMSSinceStart() << "    "<< std::endl;
	timer.start();
#endif

	paco::CRFBmpScale(g_Fpp1Out, g_ScaleOut, 1.0f / (BOI_SCALE * BOI_SAMPLE_SIZE));
#ifdef LOG_TIME
	std::cout << "    [INFO] scaleOut     : " << timer.getTimeMSSinceStart() << "    " << std::endl;
	timer.start();
#endif

	paco::CRFBmpToMatf(g_ScaleOut, g_MatOut);

	if (GetAsyncKeyState('O')) {
		if (GetAsyncKeyState('2'))
			CRFBmpSave(g_Fpp1Out, "frame/fpp1Out.bmp");
		if (GetAsyncKeyState('3'))
			CRFBmpSave(g_ScaleOut, "frame/scaleOut.bmp");
	}
	  
	return g_MatOut;
}

CR_MATF* g_CurrentInput = CRMatFCreate(12, 1);
CR_MATF* getCurrentInput()
{
	short buttons[12] = {'W', 'A', 'S', 'D', VK_UP, VK_DOWN, VK_LEFT, VK_RIGHT, ' ', VK_LCONTROL, 'E', 'Q'};
	for (uint index = 0; index < 12; index++)
	{
		g_CurrentInput->Data[index] = (GetAsyncKeyState(buttons[index]) == 0) ? 0.0f : 1.0f;
	}

	return g_CurrentInput;
}
void printBOIOutput(CR_MATF const* mat)
{
	if (mat->Cols != 1 || mat->Rows != 12)
		return;
	String buttons[] = {"W", "A", "S", "D", "UP", "LEFT", "DOWN", "RIGHT", " ", "STRG", "E", "Q"};

	for (uint index = 0; index < 12; index++)
	{
		printf("BOI Button: [%5s] [%3f]; y: [%3f]\n", buttons[index].c_str(), mat->Data[index], g_CurrentInput->Data[index]);
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
	window->setClientArea(CR_RECT{50, 100, BOI_BASE_WIDTH * BOI_SCALE, BOI_BASE_HEIGHT * BOI_SCALE});
	
	//Screen cap
	os::CRScreenCapturer* capturer = os::CRScreenCapturer::CreateInstance(window, &result);
	if (CR_FAILED(result)) {
		printf(" [ERROR] os::CRScreenCapturer::CreateInstance failed !! Exit");
		return;
	}
	capturer->runCaptureThread();

	// network
	CRNeuronLayerPtr outputLayer;
	CRNeuronNetwork* network = createBOINetwork(outputLayer);
	CR_NN_BP_LAYER_OUTPUTS* outputs = CRCreateBPLayerOut(network);
	CR_NN_BP_INFO* bpInfo = CRCreateBPInfo(network, BOI_BATCH_SIZE);

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
	uint iterations = 0;
	bool running = true;
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

		/*
		 * Frame
		 */
		CR_FBMP* frame = capturer->getFrame();

		if (!frame)
		{
			continue;
		}

		if (GetAsyncKeyState('O') && GetAsyncKeyState('1'))
		{
			String frameName = String("frame/frame") + std::to_string(frameNo++) + String(".bmp");
			CRFBmpSave(frame, frameName.c_str());
		}

		/*
		 * network
		 */

		// frame processing
		CR_MATF* data = processBOIFrame(frame);
		//CR_MATF* data = nullptr;
		CRFBmpDelete(frame);
		if (!data)
			continue;

		// process data
		network->process(data, outputs);
		if (bpInfo->TotalBPsCount == bpInfo->BatchSize)
		{
			printf("[INFO] Press P to apply backpropagation\n");
			if (GetAsyncKeyState('P'))
			{
				CRApplyBackprop(network, bpInfo);
				CRResetBPInfo(bpInfo);
			}
		} else
		{
			printf(" [INFO] backpropagation: %5i / %5i                       \n", bpInfo->TotalBPsCount, BOI_BATCH_SIZE);
			CRBackprop(bpInfo, getCurrentInput(), outputs, network);
		}
		//CRMatFDelete(data);

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

	capturer->stopCaptureThread();

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

	std::cout << std::endl;
	std::cout << "Press Y to skip" << std::endl;
	std::cout << std::endl;

	COORD conCursorPos = getConCursorPos();
	for (uint sleep = 10; sleep > 0; sleep--)
	{
		if (GetAsyncKeyState('X')) {
			break;
		}
		if (GetAsyncKeyState('Y')) 
		{
			setConCursorPos(conCursorPos);
			break;
		}

		os::CROSContext::Sleep(1);
		setConCursorPos(conCursorPos);
		std::cout << "[INFO] network start in: " << sleep << " " << std::endl;
	}

	/*
	 * Network test
	 */
	if (!GetAsyncKeyState('X')) {
		testBOINetwork();
	}

	/*
	 * cleanup
	 */
	os::CROSContext::TerminateInstance();

	return 0;
}