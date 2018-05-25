#include <Cria.hpp>
#include "src/util/FloatBitmap.h"

#include <thread>

#include "tests/MathTests.h"
#include "src/network/neurons/DataInputNeuron.h"
#include "src/network/neurons/NormalNeuron.h"
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
uint g_frameNo = 0;
uint g_iteration = 0;
CR_FLOAT_BITMAP* processBOIFrame(CR_FLOAT_BITMAP* inFrame)
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

	if (g_iteration == 0)
	{
		String inFrameName = String("test/inFrame") + std::to_string(g_frameNo) + String(".bmp");
		CRSaveBitmap(inFrame, inFrameName.c_str());
		String fpp1OutName = String("test/fpp1Out") + std::to_string(g_frameNo) + String(".bmp");
		CRSaveBitmap(fpp1Out, fpp1OutName.c_str());
		String scaleOutName = String("test/scaleOut") + std::to_string(g_frameNo) + String(".bmp");
		CRSaveBitmap(scaleOut, scaleOutName.c_str());
		String poolOutName = String("test/poolOut") + std::to_string(g_frameNo) + String(".bmp");
		CRSaveBitmap(poolOut, poolOutName.c_str());
		g_frameNo++;
	}

	CRDeleteFBmp(fpp1Out);
	CRDeleteFBmp(scaleOut);
	
	return poolOut;
}
CRNeuronNetwork* createBOINetwork(CRNeuronLayerPtr& outputLayer)
{
	std::cout << " > createBOINetwork" << std::endl;
	CRNeuronNetwork* network = new CRNeuronNetwork;
	
	/*
	 * Layer 1
	 */
	CRNeuronLayerPtr layer1 = make_shared<CRNeuronLayer>(nullptr);
	layer1->setActivationFunc(paco::CRSigmoid, paco::CRSigmoidInv);
	layer1->addNeuronGroup(make_shared<CRDataInputNeuron>(BOI_BASE_WIDTH / BOI_SAMPLE_SIZE * BOI_BASE_HEIGHT / BOI_SAMPLE_SIZE)); // 36864 Neurons :O
	network->addLayer(layer1);

	/*
	 * Layer 2
	 */
	CRNeuronLayerPtr layer2 = make_shared<CRNeuronLayer>(layer1.get());
	layer2->setActivationFunc(paco::CRSigmoid, paco::CRSigmoidInv);
	layer2->addNeuronGroup(make_shared<CRNormalNeuron>(100));
	network->addLayer(layer2);

	/*
	* Layer 3
	*/
	CRNeuronLayerPtr layer3 = make_shared<CRNeuronLayer>(layer2.get());
	layer3->setActivationFunc(paco::CRSigmoid, paco::CRSigmoidInv);
	layer3->addNeuronGroup(make_shared<CRNormalNeuron>(100));
	network->addLayer(layer3);

	/*
	* Layer 4
	*/
	outputLayer = make_shared<CRNeuronLayer>(layer3.get());
	outputLayer->setActivationFunc(paco::CRSigmoid, paco::CRSigmoidInv);
	outputLayer->addNeuronGroup(make_shared<CRNormalNeuron>(12));//WASD[^][<][v][>][ ][STRG]EQ
	network->addLayer(outputLayer);

	/*
	 * Finish and return
	 */
	network->initRandom();

	CRWriteMatrixf(layer2->getWeights(), "l2weights.txt");
	CRWriteMatrixf(layer2->getBias(), "l2bias.txt");
	CRWriteMatrixf(layer3->getWeights(), "l3weights.txt");
	CRWriteMatrixf(layer3->getBias(), "l3bias.txt");
	CRWriteMatrixf(outputLayer->getWeights(), "l4weights.txt");
	CRWriteMatrixf(outputLayer->getBias(), "l4bias.txt");

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
CRMatrixf* bmpToMat(CR_FLOAT_BITMAP* bmp)
{
	CRMatrixf* mat = CRCreateMatrixf(1, bmp->Width * bmp->Height);
	memcpy(mat->Data, bmp->Data, sizeof(float) * mat->Rows);

	return mat;
}
void testBOINetwork()
{
	std::cout << "> testBOINetwork" << std::endl;

	/*
	 * Init
	 */
	CRNeuronLayerPtr outputLayer;
	CRNeuronNetwork* network = createBOINetwork(outputLayer);
	crresult result;
	os::CRWindowPtr window = os::CRWindow::CreateInstance(BOI_TITLE, &result);
	if (CR_FAILED(result))
	{
		printf(" [ERROR] os::CRWindow::CreateInstance failed!! Exit");
		return;
	}
	window->setClientArea(CR_RECT{50, 50, BOI_BASE_WIDTH * BOI_SCALE, BOI_BASE_HEIGHT * BOI_SCALE});
	os::CRScreenCapturer* capturer = os::CRScreenCapturer::CreateInstance(window, &result);
	if (CR_FAILED(result))
	{
		printf(" [ERROR] os::CRScreenCapturer::CreateInstance failed !! Exit");
		return;
	}

	std::cout << " [INFO] = init finish" << std::endl;

	/*
	 * Loop
	 */
	StopWatch timer;
	bool running = true;
	uint iterations = 0;
	while (running)
	{
		g_iteration = iterations;
		/*
		 * Exit check
		 */
		if (GetAsyncKeyState('X'))
		{
			running = false;
			printf(" [EXIT] exit because u wanted it \n");
			break;
		}

		/*
		 * Network
		 */
		capturer->grabFrame();
		CR_FLOAT_BITMAP* frame = capturer->getLastFrame();
		CR_FLOAT_BITMAP* processedFrame = processBOIFrame(frame);
		
		if (true)
		{
			// bmp to mat
			CRMatrixf* data = bmpToMat(processedFrame);
			CRDeleteFBmp(processedFrame);
	
			// process data
			network->process(data);
			CRFreeMatrixf(data);

			// print result
			printBOIOutput(outputLayer->getOutput());
		}
		
		/*
		 * Info 
		 */
		iterations++;
		if (timer.getTimeMSSinceStart() >= 1000)
		{
			std::cout << std::endl;
			std::cout << " [INFO] IPS: " << iterations << ", average:" << (timer.getTimeMSSinceStart() / iterations) << std::endl;
			std::cout << std::endl;

			timer.start();
			iterations = 0;
		}
		
	}

	delete network;

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

	cin.get();
	return 0;
}