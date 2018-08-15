#include <Cria.hpp>

#include <math.h>
#include <iostream>
#include <stdlib.h>
#include <vector>
#include <AsyncInfo.h>

#include <thread>

#define LERN_RATE 0.5f
#define LERN_CYCLES 1000
#define BATCH_SIZE 1

using namespace cria_ai;
typedef unsigned int uint;
typedef std::vector<float> fvec;

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

float GetRandFloat()
{
	return (((float)rand() / (float)RAND_MAX) * 2) - 1;
}


float AFCal(float value)
{
	return 1.0f / (1.0f + exp(-value));
}
float AFInv(float value)
{
	return value * (1 - value);
}

class NLayer
{
	uint m_InputCount;
	uint m_NeuronCount;

	CR_MATF* m_Input;
	CR_MATF* m_Output;
	CR_MATF* m_Blame;

	CR_MATF* m_Weigths;
	CR_MATF* m_Bias;

public:

	NLayer(uint inputCount, uint neuronCount)
		: m_InputCount(inputCount),
		m_NeuronCount(neuronCount),
		m_Input(nullptr),
		m_Output(nullptr),
		m_Blame(nullptr),
		m_Weigths(nullptr),
		m_Bias(nullptr)
	{
		m_Input = CRMatFCreate(1, inputCount);
		m_Output = CRMatFCreate(1, m_NeuronCount);

		m_Blame = CRMatFCreate(1, m_NeuronCount);

		m_Weigths = CRMatFCreate(inputCount, m_NeuronCount);
		m_Bias = CRMatFCreate(1, m_NeuronCount);


		if (!m_Input || !m_Output || !m_Blame  || !m_Weigths || !m_Bias)
			return;

		for (uint index = 0; index < CR_MATF_VALUE_COUNT(m_Bias); index++)
		{
			m_Bias->Data[index] = GetRandFloat();
		}
		for (uint index = 0; index < CR_MATF_VALUE_COUNT(m_Weigths); index++)
		{
			m_Weigths->Data[index] = GetRandFloat();
		}
	}
	~NLayer()
	{
		if (m_Input) {
			CRMatFDelete(m_Input);
			m_Input = nullptr;
		}
		if (m_Output) {
			CRMatFDelete(m_Output);
			m_Output = nullptr;
		}

		if (m_Blame)
		{
			CRMatFDelete(m_Blame);
			m_Blame = nullptr;
		}

		if (m_Weigths) 
		{
			CRMatFDelete(m_Weigths);
			m_Weigths = nullptr;
		}
		if (m_Bias)
		{
			CRMatFDelete(m_Bias);
			m_Bias = nullptr;
		}
	}

	fvec JSFeedForward(const fvec& input)
	{
		/*
		 * Init
		 */
		for (uint inIndex = 0; inIndex < m_InputCount; inIndex++)
		{
			m_Input->Data[inIndex] = input[inIndex];
		}

		/*
		 * Feed the child named "Forward" 
		 */
		CR_MATF* weightOut = CRMatFMul(m_Input, m_Weigths);
		CR_MATF* biasOut = CRMatFAdd(weightOut, m_Bias);
		for (uint outIndex = 0; outIndex < m_NeuronCount; outIndex++)
		{
			m_Output->Data[outIndex] = AFCal(biasOut->Data[outIndex]);
		}
		CRMatFDelete(weightOut);
		CRMatFDelete(biasOut);

		/*
		 * Return
		 */
		fvec output(m_NeuronCount);
		for (uint outIndex = 0; outIndex < m_NeuronCount; outIndex++)
		{
			output[outIndex] = m_Output->Data[outIndex];
		}
		return output;
	}

	void updateWeights(uint neuronNo)
	{
		float pdOutWrtNet = AFInv(m_Output->Data[neuronNo]);
		float pdErrWrtNet = m_Blame->Data[neuronNo] * pdOutWrtNet;

		for (uint wNo = 0; wNo < m_InputCount; wNo++)
		{
			float weightErr = pdErrWrtNet * m_Input->Data[wNo];
			m_Weigths->Data[CR_MATF_VALUE_INDEX(wNo, neuronNo, m_Weigths)] += LERN_RATE * weightErr;
		}
	}
	void updateBias(uint neuronNo)
	{
		float pdOutWrtNet = AFInv(m_Output->Data[neuronNo]);
		float pdErrWrtNet = m_Blame->Data[neuronNo] * pdOutWrtNet;

		m_Bias->Data[neuronNo] += LERN_RATE * pdErrWrtNet;
	}

	void printInfo() const
	{
		printf("[INFO] Layer:\n");
		printf("[INFO] - Neuron count: %u\n", m_NeuronCount);
		for (uint neuronNo = 0; neuronNo < m_NeuronCount; neuronNo++)
		{
			printf("[INFO]   - Neuron No:  %2u [B: %+1.3f] {", neuronNo, m_Bias->Data[neuronNo]);
			
			for (uint wNo = 0; wNo < m_InputCount; wNo++)
			{
				printf("%+1.3f", getWeight(neuronNo, wNo));
				if (wNo < m_InputCount - 1)
				{
					printf(", ");
				}
			}
			printf("}\n");
		}
	}

	void setNeuronBlame(uint neuronNo, float blame)
	{
		m_Blame->Data[neuronNo] = blame;
	}
	void addNeuronBlame(uint neuronNo, float blame)
	{
		m_Blame->Data[neuronNo] += blame;
	}
	float getNeuronBlame(uint neuronNo) const
	{
		return m_Blame->Data[neuronNo];
	}

	uint getNeuronCount() const
	{
		return m_NeuronCount;
	}

	float getOutput(uint neuronNo) const
	{
		return m_Output->Data[neuronNo];
	}
	float getWeight(uint neuronNo, uint weightNo) const
	{
		return m_Weigths->Data[CR_MATF_VALUE_INDEX(weightNo, neuronNo, m_Weigths)];
	}
	void setWeight(uint neuronNo, uint weightNo, float weight)
	{
		m_Weigths->Data[CR_MATF_VALUE_INDEX(weightNo, neuronNo, m_Weigths)] = weight;
	}
	void addToWeight(uint neuronNo, uint weightNo, float weight)
	{
		m_Weigths->Data[CR_MATF_VALUE_INDEX(weightNo, neuronNo, m_Weigths)] += weight;
	}
};

class NN
{
private:
	uint m_InputCount;
	NLayer m_HiddenLayer;
	NLayer m_OutputLayer;
	fvec m_LastInput;
public: 
	NN(uint inCount, uint hidCount, uint outCount)
		: m_InputCount(inCount),
		m_HiddenLayer(inCount, hidCount),
		m_OutputLayer(hidCount, outCount)
	{}
	fvec JSFeedForward(const fvec& input)
	{
		fvec hidOut = m_HiddenLayer.JSFeedForward(input);
		return m_OutputLayer.JSFeedForward(hidOut);
	}
	float JSGetErr(const fvec& input, const fvec& idealOut)
	{
		float total = 0.0f;
		fvec realOut = JSFeedForward(input);

		for (uint outNo = 0; outNo < idealOut.size(); outNo++)
		{
			float err = abs(idealOut[outNo] - realOut[outNo]);
			total += err / idealOut.size();
		}

		return total;
	}
	void JSTrain(const fvec& input, const fvec& idealOut)
	{
		JSFeedForward(input);
		for (uint outNo = 0; outNo < m_OutputLayer.getNeuronCount(); outNo++)
		{
			m_OutputLayer.setNeuronBlame(outNo, idealOut[outNo] - m_OutputLayer.getOutput(outNo));
			m_OutputLayer.updateWeights(outNo);
			m_OutputLayer.updateBias(outNo);
		}
		for (uint hidNo = 0; hidNo < m_HiddenLayer.getNeuronCount(); hidNo++)
		{
			m_HiddenLayer.setNeuronBlame(hidNo, 0.0f);
			for (uint outNo = 0; outNo < m_OutputLayer.getNeuronCount(); outNo++)
			{
				float weightBlame = m_OutputLayer.getWeight(outNo, hidNo) * m_OutputLayer.getNeuronBlame(outNo);
				m_HiddenLayer.addNeuronBlame(hidNo, weightBlame);
			}

			m_HiddenLayer.updateWeights(hidNo);
			m_HiddenLayer.updateBias(hidNo);
		}
	}

	void printInfo() const
	{
		printf("NeuronNetwork: \n");
		
		if (!m_LastInput.empty())
		{
			printf("   Inputs {");
			for (uint inputNo = 0; inputNo < m_InputCount; inputNo++) 
			{
				printf("%1.3f", m_LastInput[inputNo]);
				if (inputNo != m_InputCount - 1)
					printf(", ");
			}
			printf("}\n\n");
			
		}
		
		printf("Hidden Layer\n");
		m_HiddenLayer.printInfo();

		printf("\n");
		printf("Output Layer\n");
		m_OutputLayer.printInfo();
	}


};

int main()
{
	printf("##########################################\n");
	printf("Hello and welcome to this NeuronNetwork!!!\n");
	printf("- XOR gates are fun but AI is the future -\n");
	printf("##########################################\n");
	printf("\n");
	
	/*
	 * Sorry this is just test code (famous last words)
	 */
	srand(0);
	NN nn(2, 5, 1);
	nn.printInfo();
	std::cin.get();
	fvec trainDataInput[4]    = {{0.1f, 0.1f}, {0.1f, 0.9f}, {0.9f, 0.1f}, {0.9f, 0.9f}};
	fvec trainDataIdealOut[4] = {{0.1f}, {0.9f}, {0.9f}, {0.1f}};

	COORD curserPos = getConCursorPos();
	for (uint trainNo = 0; trainNo < 100000; trainNo++)
	{
		uint trainIndex = rand() % 4;

		nn.JSTrain(trainDataInput[trainIndex], trainDataIdealOut[trainIndex]);

		if (trainNo % 25 == 0)
		{
			setConCursorPos(curserPos);

			printf("Epoch No: %u \n\n", trainNo);
			float totalErr = 0.0f;

			for (uint printIndex = 0; printIndex < 4; printIndex++)
			{
				fvec nnInput = trainDataInput[printIndex];
				fvec nnIdealOut = trainDataIdealOut[printIndex];
				fvec nnOut = nn.JSFeedForward(nnInput);
				printf("Input: {%1.1f, %1.1f}, ", nnInput[0], nnInput[1]);
				
				printf("Output: {");
				for (uint outNo = 0; outNo < nnOut.size(); outNo++) 
				{
					printf("%1.1f", nnOut[outNo]);
					printf(" [%1.1f]", nnIdealOut[outNo]);

					if (outNo != nnOut.size() - 1)
						printf(", ");
				}
				printf("} ");

				float error = nn.JSGetErr(nnInput, nnIdealOut);
				printf(", Error: %1.6f \n", error);
				totalErr += error;

			}
			printf("\n");

			printf("Average Error: %1.6f\n", totalErr / 4);

			printf("\n");
			nn.printInfo();
			std::this_thread::sleep_for(std::chrono::milliseconds(100));
		}
		
	}
	printf("\n");

	std::cin.get();
	return 0;
}