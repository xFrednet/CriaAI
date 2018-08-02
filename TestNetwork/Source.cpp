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

typedef unsigned int uint;
typedef std::vector<float> fvec;
typedef cria_ai::CRMatrixf CR_MATF;

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

class NN;
class Nlayer;

class Neuron
{
	friend class NN;
	friend class NLayer;

	fvec m_Inputs;
	
	fvec m_Weights;
	float m_Bias;

	float m_Output;
	
	float m_Blame;

public:

	Neuron(float bias, uint inputCount)
		: m_Inputs(inputCount),
		m_Weights(inputCount),
		m_Bias(bias),
		m_Output(0.0f),
		m_Blame(0.0f)
	{
		for (uint weightNo = 0; weightNo < inputCount; weightNo++) 
		{
			m_Weights[weightNo] = GetRandFloat();
		}
	}

	float JSCalWeightedSum()
	{
		float total = 0.0f;
		for (uint inNo = 0; inNo < m_Inputs.size(); inNo++)
		{
			total += m_Inputs[inNo] * m_Weights[inNo];
		}

		return total + m_Bias;
	}
	float JSCalOutput(const fvec& input)
	{
		m_Inputs = input;
		float weightedSum = JSCalWeightedSum();
		m_Output = AFCal(weightedSum);
		return m_Output;
	}
	fvec JSUpdateWeights()
	{
		float pdOutWrtNet = AFInv(m_Output);
		float pdErrWrtNet = m_Blame * pdOutWrtNet;

		for (uint wNo = 0; wNo < m_Inputs.size(); wNo++) 
		{
			float pdNetWrtInput = m_Inputs[wNo];
			float errWrtWeight = pdErrWrtNet * pdNetWrtInput;
			m_Weights[wNo] += LERN_RATE * errWrtWeight;
		}
		return m_Weights;
	}
	float JSUpdateBias()
	{
		float pdOutWrtNet = AFInv(m_Output);
		float pdErrWrtNet = m_Blame * pdOutWrtNet;

		float outSum = JSCalWeightedSum();
		float errWrtBias = outSum * pdErrWrtNet; //??

		m_Bias += pdErrWrtNet * LERN_RATE;

		return m_Bias;
	}


	void printWeights() const
	{
		for (uint weightNo = 0; weightNo < m_Weights.size(); weightNo++) {
			printf("%+1.3f", m_Weights[weightNo]);
			if (weightNo < m_Weights.size() - 1)
				printf(", ");
		}
	}

	/*
	 * getters
	 */
	float getBias() const
	{
		return m_Bias;
	}
	float getLastOutput() const
	{
		return m_Output;
	}
	float getWeight(uint weightNo) const
	{
		return m_Weights[weightNo];
	}
};

class NLayer
{
	friend class NN;

	uint m_NeuronCount;
	std::vector<Neuron> m_Neurons;

	//CR_MATF m_Weigths;
	//CR_MATF m_Bias;

public:

	NLayer(uint neuronCount, uint inputCount, bool JS)
		: m_NeuronCount(neuronCount)
	{
		float bias = GetRandFloat();
		for (uint nNo = 0; nNo < neuronCount; nNo++)
		{
			m_Neurons.push_back(Neuron(bias, inputCount));
		}
	}
	fvec JSFeedForward(const fvec& input)
	{
		fvec output(m_NeuronCount);
		for (uint nNo = 0; nNo < m_NeuronCount; nNo++)
		{
			output[nNo] = m_Neurons[nNo].JSCalOutput(input);
		}
		return output;
	}

	NLayer(uint neuronCount, uint inputCount, fvec bias = {})
		: m_NeuronCount(neuronCount)
	{
		for (uint neuronNo = bias.size(); neuronNo < neuronCount; neuronNo++) {
			bias.push_back(GetRandFloat());
		}

		for (uint neuronNo = 0; neuronNo < neuronCount; neuronNo++) 
		{
			m_Neurons.push_back(Neuron(bias[neuronNo], inputCount));
		}

	}

	void printInfo() const
	{
		printf("[INFO] Layer:\n");
		printf("[INFO] - Neuron count: %u\n", m_NeuronCount);
		for (uint neuronNo = 0; neuronNo < m_NeuronCount; neuronNo++)
		{
			printf("[INFO]   - Neuron No:  %2u [B: %+1.3f] {", neuronNo, m_Neurons[neuronNo].getBias());
			m_Neurons[neuronNo].printWeights();
			printf("}\n");
		}
	}

	fvec getOutputs() const
	{
		fvec outputs(m_NeuronCount);
		for (uint neuronNo = 0; neuronNo < m_NeuronCount; neuronNo++) {
			outputs[neuronNo] = m_Neurons[neuronNo].getLastOutput();
		}

		return outputs;
	}


	uint getNeuronCount() const
	{
		return m_NeuronCount;
	}

	float getWeight(uint neuronNo, uint weightNo) const
	{
		return m_Neurons[neuronNo].getWeight(weightNo);
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
	NN(uint inCount, uint hidCount, uint outCount, bool JS)
		: m_InputCount(inCount),
		m_HiddenLayer(hidCount, inCount),
		m_OutputLayer(outCount, hidCount)
	{}
	fvec JSFeedForward(const fvec& input)
	{
		fvec hidOut = m_HiddenLayer.JSFeedForward(input);
		return m_OutputLayer.JSFeedForward(hidOut);
	}
	float JSGetAvgErr(const fvec& input, const fvec& idealOut)
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
		fvec out = JSFeedForward(input);
		for (uint outNo = 0; outNo < idealOut.size(); outNo++)
		{
			Neuron& neuron = m_OutputLayer.m_Neurons[outNo];
			neuron.m_Blame = idealOut[outNo] - neuron.m_Output;
			neuron.JSUpdateWeights();
			neuron.JSUpdateBias();

		}
		for (uint hidNo = 0; hidNo < m_HiddenLayer.m_NeuronCount; hidNo++)
		{
			Neuron& hidNeuron = m_HiddenLayer.m_Neurons[hidNo];
			hidNeuron.m_Blame = 0;
			for (uint outNo = 0; outNo < m_OutputLayer.m_NeuronCount; outNo++)
			{
				Neuron& outNeuron = m_OutputLayer.m_Neurons[outNo];
				hidNeuron.m_Blame += outNeuron.m_Weights[hidNo] * outNeuron.m_Blame;
			}

			hidNeuron.JSUpdateWeights();
			hidNeuron.JSUpdateBias();
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
	
	srand(0);
	NN nn(2, 5, 1, true);
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

				float error = nn.JSGetAvgErr(nnInput, nnIdealOut);
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

	/*
	 * Sorry this is just test code (famous last words)
	 */
	bool running = false;
	while (running)
	{
		printf("-1: exit\n");
		printf(" 0: Test input {in1}, {in2}\n");
		printf(" 1: Print network info\n");
		printf(" 2: Train network\n");

		int conIn = 0;
		scanf_s("%u", &conIn);
		printf("\n");
		switch (conIn) {
			case -1:
				printf("Goodbye!\n");
				running = false;
				break;
			case 0:
			{
				fvec in(2);
				if (scanf_s(" %f, %f", &in[0], &in[1]) != 2)
					break;

				fvec out = nn.JSFeedForward(in);
				printf("nn output: {");

				for (uint outNo = 0; outNo < out.size(); outNo++) {
					printf("%1.3f", out[outNo]);
					if (outNo != out.size() - 1)
						printf(", ");
				}
				printf("}\n");
				break;
			}
			case 1:
				nn.printInfo();
				break;
			case 2:
				{
					float avErr = 0.0f;
					for (uint lernNo = 0; lernNo < LERN_CYCLES; lernNo++)
					{
						fvec nnInput(2);
						nnInput[0] = ((rand() % 2) == 0) ? 0.0f : 1.0f;
						nnInput[1] = ((rand() % 2) == 0) ? 0.0f : 1.0f;
						fvec nnIdealOut(1);
						nnIdealOut[0] = (nnInput[0] == nnInput[1]) ? 0.1f : 0.9f;
					
						avErr += nn.JSGetAvgErr(nnInput, nnIdealOut) / LERN_CYCLES;
						nn.JSTrain(nnInput, nnIdealOut);
					}

					printf("Training complete: average error: %1.3f \n", avErr);
				}
				
				break;
			default:
				printf("Unknown input :/");
				break;
		}

		printf("\n");
	}

	std::cin.get();
	return 0;
}