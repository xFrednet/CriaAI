#include <Cria.hpp>
#include "src/util/FloatBitmap.h"

#include <thread>

#include "tests/MathTests.h"
#include "src/network/neurons/DataInputNeuron.h"
#include "src/network/neurons/NormalNeuron.h"

#define BOI_TITLE                      "Binding of Isaac: Afterbirth+"
#define BOI_BASE_WIDTH                 512
#define BOI_BASE_HEIGHT                288
#define BOI_SCALE                      1
#define BOI_SAMPLE_SIZE                2

#define CON_TITLE "C:\\Users\\xFrednet\\My Projects\\VS Projects\\CriaAI\\bin\\Debug\\Sandbox.exe"

using namespace std;
using namespace cria_ai;
using namespace bmp_renderer;
using namespace network;


CR_FLOAT_BITMAP* processBOIFrame(CR_FLOAT_BITMAP* inFrame)
{
	if (inFrame->Width != BOI_BASE_WIDTH * BOI_SCALE || inFrame->Height != BOI_BASE_HEIGHT * BOI_SCALE)
	{
		std::cout << "processBOIFrame: something is wrong" << std::endl;
		return nullptr;
	}

	CR_FLOAT_BITMAP* fpp1Out = CRConvertToFloatsPerPixel(inFrame, 1);
	CR_FLOAT_BITMAP* scaleOut = CRScaleFBmpDown(fpp1Out, BOI_SCALE);
	CR_FLOAT_BITMAP* poolOut = CRPoolBitmap(scaleOut, 2);

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
	
	CR_FLOAT_BITMAP* frame = CRLoadFBmp("screenshots/The Binding of Isaac Afterbirth+.bmp");
	CRNeuronLayerPtr outputLayer;
	CRNeuronNetwork* network = createBOINetwork(outputLayer);
	
	std::cout << "= init finish" << std::endl;
	std::cout << std::endl;

	StopWatch timer;
	timer.start();
	
	CR_FLOAT_BITMAP* processedFrame = processBOIFrame(frame);
	CRMatrixf* data = bmpToMat(processedFrame);
	CRDeleteFBmp(processedFrame);
	
	network->process(data);
	CRFreeMatrixf(data);

	timer.stop();
	std::cout << " [INFO] process time[ms]: " << timer.getTimeMS() << std::endl;
	std::cout << std::endl;

	printBOIOutput(outputLayer->getOutput());

	CRDeleteFBmp(frame);
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

	/*
	 * time test
	 */

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