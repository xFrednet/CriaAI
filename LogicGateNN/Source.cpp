#include "Cria.hpp"
#include "src/network/Backprop.h"

#include <windows.h>
#include <AsyncInfo.h>

using namespace cria_ai;
using namespace network;

#define BATCH_SIZE                               1000

#define LG_OR_EXPECTED(in1, in2)                 (in1 != 0 || in2 != 0)
#define LG_XOR_EXPECTED(in1, in2)                (in1 != in2)
#define LG_AND_EXPECTED(in1, in2)                (in1 != 0 && in2 != 0)
#define LG_NAND_EXPECTED(in1, in2)               (!LG_AND_EXPECTED(in1, in2))

#define LOGIC_GATE_EXPECTED(in1, in2)            LG_XOR_EXPECTED(in1, in2)

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
CRNeuronNetwork* createNetwork(CRNeuronLayerPtr& outputLayer)
{
	std::cout << " > createBOINetwork" << std::endl;
	CRNeuronNetwork* network = new CRNeuronNetwork;

	/*
	* Layer 0
	*/
	CRNeuronLayerPtr layer0 = std::make_shared<CRNeuronLayer>(nullptr, 2);
	layer0->setActivationFunc(paco::CRSigmoid, paco::CRSigmoidInv);
	network->addLayer(layer0);

	/*
	* Layer 1
	*/
	CRNeuronLayerPtr layer1 = std::make_shared<CRNeuronLayer>(layer0.get(), 5);
	layer1->setActivationFunc(paco::CRSigmoid, paco::CRSigmoidInv);
	network->addLayer(layer1);

	/*
	* Layer 2
	*/
	outputLayer = std::make_shared<CRNeuronLayer>(layer1.get(), 1);
	outputLayer->setActivationFunc(paco::CRSigmoid, paco::CRSigmoidInv);
	network->addLayer(outputLayer);

	return network;
}
int main()
{
	os::CROSContext::InitInstance();

	CRNeuronLayerPtr outputLayer;
	CRNeuronNetwork* network = createNetwork(outputLayer);
	network->initRandom();
	CR_NN_BP_LAYER_OUTPUTS* outputs = CRCreateBPLayerOut(network);
	CR_NN_BP_INFO* bpInfo = CRCreateBPInfo(network, BATCH_SIZE);
	CRMatrixf* inputData = CRCreateMatrixf(1, 2);
	CRMatrixf* expectedOutput = CRCreateMatrixf(1, 1);

	printf("[ INFO] Hello, please press X to exit.\n\n");

	uint itNo = 0;
	uint epochNo = 0;
	float epochCost = 0.0f;
	COORD conCurPos = getConCursorPos();
	while (!GetAsyncKeyState('X'))
	{
		if (itNo % 50 == 0)
		{
			do {
				os::CROSContext::Sleep(0, 100);
			} while (GetAsyncKeyState('P'));

			setConCursorPos(conCurPos);
			for (uint lineNo = 0; lineNo < 55; lineNo++)
			{
				printf("                                                                \n");
			}
			setConCursorPos(conCurPos);
			printf("[ INFO] Epoch %3u {%1.3f}\n", epochNo, epochCost);
		}
		/*
		 * Data
		 */
		float in1 = (rand() % 2 == 0) ? 0.0f : 1.0f;
		float in2 = (rand() % 2 == 0) ? 0.0f : 1.0f;
		inputData->Data[0] = (in1);
		inputData->Data[1] = (in2);
		expectedOutput->Data[0] = (float)LOGIC_GATE_EXPECTED(inputData->Data[0], inputData->Data[1]);
		
		/*
		 * Process
		 */
		network->process(inputData, outputs);
		printf("[ INFO] %1.3f, %1.3f = ", inputData->Data[0], inputData->Data[1]);
		printf("%1.3f [%1f] {%1.3f}", outputLayer->getOutput()->Data[0], expectedOutput->Data[0], CRGetCost(outputLayer->getOutput(), expectedOutput));

		/*
		 * Backprop
		 */
		CRBackprop(bpInfo, expectedOutput, outputs, network);
		printf(", backpropagation: %5i / %5i\n", bpInfo->TotalBPsCount, BATCH_SIZE);
		if (bpInfo->TotalBPsCount == bpInfo->BatchSize) {
			epochCost = bpInfo->TotalCost;
			CRApplyBackprop(network, bpInfo);
			CRResetBPInfo(bpInfo);
			epochNo++;
		}
		//os::CROSContext::Sleep(0, 10);
		itNo++;
	}

	return 0;
}