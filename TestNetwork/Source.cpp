#include <math.h>
#include <iostream>
#include <stdlib.h>
#include <vector>
#include <AsyncInfo.h>

#include <thread>

#define LERN_RATE 1.0f
#define LERN_CYCLES 1000
#define BATCH_SIZE 1

typedef unsigned int uint;
typedef std::vector<float> fvec;
typedef std::vector<fvec> fvecvec;

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
	float m_PDErrWrtNet;

	//BP
	fvec m_WeightChange;
	float m_BiasChange;

public:

	Neuron(float bias, uint inputCount)
		: m_Inputs(inputCount),
		m_Weights(inputCount),
		m_Bias(bias),
		m_Output(0.0f),
		m_Blame(0.0f),
		m_PDErrWrtNet(0.0f),
		
		m_WeightChange(inputCount),
		m_BiasChange(0.0f)
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
		m_PDErrWrtNet = m_Blame * pdOutWrtNet;

		for (uint wNo = 0; wNo < m_Inputs.size(); wNo++) 
		{
			float pdNetWrtInput = m_Inputs[wNo];
			float errWrtWeight = m_PDErrWrtNet * pdNetWrtInput;
			m_Weights[wNo] += LERN_RATE * errWrtWeight;
		}
		return m_Weights;
	}

	float calNetOutput(const fvec& inputs)
	{
		if (inputs.size() != m_Weights.size())
			return 0.0f;

		memcpy(&m_Inputs[0], &inputs[0], sizeof(float) * inputs.size());

		float total = 0.0f;
		for (uint inNo = 0; inNo < inputs.size(); inNo++) 
		{
			total += inputs[inNo] * m_Weights[inNo];
		}

		return total + m_Bias;
	}
	float activFunc(float netInput) const
	{
		return 1.0f / (1.0f + exp(-netInput));
	}
	float calOutput(const fvec& inputs)
	{
		m_Output = activFunc(calNetOutput(inputs));
		return m_Output;
	}

	/*
	 * BP
	 */
	float calError(float idealOut) const
	{
		float errorSqrt = idealOut - m_Output;
		return 0.5f * (errorSqrt * errorSqrt);
	}

	float calculate_pd_error_wrt_output(float idealOut) const
	{
		return -(idealOut - m_Output);
	}
	float calculate_pd_total_net_input_wrt_input() const
	{
		return m_Output * (1 - m_Output);
	}
	float calculate_pd_error_wrt_total_net_input(float idealOut) const
	{
		return this->calculate_pd_error_wrt_output(idealOut) * calculate_pd_total_net_input_wrt_input();
	}

	float calculate_pd_total_net_input_wrt_weight(uint index)
	{
		return m_Inputs[index];
	}

	void printWeights() const
	{
		for (uint weightNo = 0; weightNo < m_Weights.size(); weightNo++) {
			printf("%+1.3f", m_Weights[weightNo]);
			if (weightNo < m_Weights.size() - 1)
				printf(", ");
		}
	}

	void applyBP()
	{
		for (uint weightNo = 0; weightNo < m_Weights.size(); weightNo++)
		{
			m_Weights[weightNo] -= m_WeightChange[weightNo];
			m_WeightChange[weightNo] = 0.0f;
		}
		m_Bias -= m_BiasChange;
		m_BiasChange = 0.0f;
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
			printf("[INFO]   - Neuron No: %2u [B: %+1.3f] {", neuronNo, m_Neurons[neuronNo].getBias());
			m_Neurons[neuronNo].printWeights();
			printf("}\n");
		}
	}

	fvec feedForward(const fvec& inputs)
	{
		fvec outputs(m_NeuronCount);
		for (uint neuronNo = 0; neuronNo < m_NeuronCount; neuronNo++) 
		{
			outputs[neuronNo] = m_Neurons[neuronNo].calOutput(inputs);
		}

		return outputs;
	}
	fvec getOutputs() const
	{
		fvec outputs(m_NeuronCount);
		for (uint neuronNo = 0; neuronNo < m_NeuronCount; neuronNo++) {
			outputs[neuronNo] = m_Neurons[neuronNo].getLastOutput();
		}

		return outputs;
	}

	float calError(const fvec& idealOut)
	{
		float totalErr = 0.0f;
		for (uint neuronNo = 0; neuronNo < m_NeuronCount; neuronNo++) 
		{
			totalErr += m_Neurons[neuronNo].calError(idealOut[neuronNo]);
		}

		return totalErr;
	}

	fvec calPDErrNetInput(const fvec& idealOut)
	{
		fvec pdErr(m_NeuronCount);
		for (uint outNo = 0; outNo < m_NeuronCount; outNo++)
		{
			pdErr[outNo] = m_Neurons[outNo].calculate_pd_error_wrt_total_net_input(idealOut[outNo]);
		}

		return pdErr;
	}

	void applyBP()
	{
		for (uint neuronNo = 0; neuronNo < m_NeuronCount; neuronNo++)
		{
			m_Neurons[neuronNo].applyBP();
		}
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

typedef struct TRAINING_SET_ 
{
	fvec Input;
	fvec IdealOut;
} TRAINING_SET;
typedef std::vector<TRAINING_SET> tsvec;

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
		}
	}

	NN(uint inCount, uint hiddenCount, uint outCount) 
		: m_InputCount(inCount),
		m_HiddenLayer(hiddenCount, inCount),
		m_OutputLayer(outCount, hiddenCount)
	{
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

	fvec feedForward(const fvec& input)
	{
		fvec hiddenLayerOut = m_HiddenLayer.feedForward(input);
		return m_OutputLayer.feedForward(hiddenLayerOut);
	}

	float calTotalError(const fvec& input, const fvec& idealOut)
	{
		feedForward(input);
		return m_OutputLayer.calError(idealOut);
	}

	void setBiasesAndWeights()
	{
		//m_HiddenLayer.m_Neurons[0].m_Bias = 0.35f;
		//m_HiddenLayer.m_Neurons[0].m_Weights = {0.15f, 0.2f};
		//m_HiddenLayer.m_Neurons[1].m_Bias = 0.35f;
		//m_HiddenLayer.m_Neurons[1].m_Weights = {0.25f, 0.30f};
		//
		//m_OutputLayer.m_Neurons[0].m_Bias = 0.6f;
		//m_OutputLayer.m_Neurons[0].m_Weights = {0.40f, 0.45f};
		//m_OutputLayer.m_Neurons[1].m_Bias = 0.6f;
		//m_OutputLayer.m_Neurons[1].m_Weights = {0.50f, 0.55f};
	}
	void train(const fvec& traiIn, const fvec& idealOut)
	{
		feedForward(traiIn);
		
		uint hidLayNCount = m_HiddenLayer.getNeuronCount();
		uint outLayNCount = m_OutputLayer.getNeuronCount();

		// output neuron deltas
		fvec pdOutErrNetInput = m_OutputLayer.calPDErrNetInput(idealOut);

		// hidden neuron deltas
		fvec pfHidErrNetInput(hidLayNCount);
		for (uint hnNo = 0; hnNo < hidLayNCount; hnNo++)
		{
			float neuronErr = 0.0f;
			for (uint onNo = 0; onNo < outLayNCount; onNo++)
			{
				neuronErr += pdOutErrNetInput[onNo] * m_OutputLayer.getWeight(onNo, hnNo);
			}

			pfHidErrNetInput[hnNo] = neuronErr * m_HiddenLayer.m_Neurons[hnNo].calculate_pd_total_net_input_wrt_input();
		}


		//update output layer weights
		for (uint onNo = 0; onNo < outLayNCount; onNo++)
		{
			for (uint weightNo = 0; weightNo < hidLayNCount; weightNo++)
			{
				float pdErrWeight = pdOutErrNetInput[onNo] * 
					m_OutputLayer.m_Neurons[onNo].calculate_pd_total_net_input_wrt_weight(weightNo);

				m_OutputLayer.m_Neurons[onNo].m_WeightChange[weightNo] += (LERN_RATE * pdErrWeight) / BATCH_SIZE;
			}
		}

		// Update hidden layer weights
		for (uint hnNo = 0; hnNo < hidLayNCount; hnNo++) 
		{
			for (uint weightNo = 0; weightNo < m_InputCount; weightNo++)
			{
				float pdErrWeight = pfHidErrNetInput[hnNo] *
					m_HiddenLayer.m_Neurons[hnNo].calculate_pd_total_net_input_wrt_weight(weightNo);

				m_HiddenLayer.m_Neurons[hnNo].m_WeightChange[weightNo] += (LERN_RATE * pdErrWeight) / BATCH_SIZE;
			}
		}
	}

	void applyBP()
	{
		m_HiddenLayer.applyBP();
		m_OutputLayer.applyBP();
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
	NN nn(2, 20, 1, true);
	//nn.setBiasesAndWeights();

	COORD curserPos = getConCursorPos();
	for (uint trainNo = 0; trainNo < 100000; trainNo++)
	{
		fvec nnInput(2);
		nnInput[0] = ((rand() % 2) == 0) ? 0.1f : 0.9f;
		nnInput[1] = ((rand() % 2) == 0) ? 0.1f : 0.9f;
		fvec nnIdealOut(1);// = {0.01f, 0.99f};
		nnIdealOut[0] = (nnInput[0] == nnInput[1]) ? 0.1f : 0.9f;

		//nnInput = {0.1f, 0.9f};
		//nnIdealOut = {0.1f};

		nn.JSTrain(nnInput, nnIdealOut);

		//nn.applyBP();
		if (trainNo % 50 == 0)
		{
			setConCursorPos(curserPos);

			printf("Epoch No: %u \n\n", trainNo);

			nnInput = {0.1f, 0.1f};
			nnIdealOut = {0.1f};
			fvec nnOut = nn.JSFeedForward(nnInput);
			printf("Input: {%1.1f, %1.1f}, Output: {%1.1f} [%1.1f], Error: %1.6f \n",
				nnInput[0], nnInput[1],
				nnOut[0], nnIdealOut[0],
				nn.JSGetAvgErr(nnInput, nnIdealOut));

			nnInput = {0.1f, 0.9f};
			nnIdealOut = {0.9f};
			nnOut = nn.JSFeedForward(nnInput);
			printf("Input: {%1.1f, %1.1f}, Output: {%1.1f} [%1.1f], Error: %1.6f \n",
				nnInput[0], nnInput[1],
				nnOut[0], nnIdealOut[0],
				nn.JSGetAvgErr(nnInput, nnIdealOut));

			nnInput = {0.9f, 0.1f};
			nnIdealOut = {0.9f};
			nnOut = nn.JSFeedForward(nnInput);
			printf("Input: {%1.1f, %1.1f}, Output: {%1.1f} [%1.1f], Error: %1.6f \n",
				nnInput[0], nnInput[1],
				nnOut[0], nnIdealOut[0],
				nn.JSGetAvgErr(nnInput, nnIdealOut));

			nnInput = {0.9f, 0.9f};
			nnIdealOut = {0.1f};
			nnOut = nn.JSFeedForward(nnInput);
			printf("Input: {%1.1f, %1.1f}, Output: {%1.1f} [%1.1f], Error: %1.6f \n",
				nnInput[0], nnInput[1],
				nnOut[0], nnIdealOut[0],
				nn.JSGetAvgErr(nnInput, nnIdealOut));

			printf("\n");
			nn.printInfo();
			std::this_thread::sleep_for(std::chrono::milliseconds(50));
			//fvec nnOut = nn.JSFeedForward(nnInput);
			//printf("Input: {%1.1f, %1.1f}, Output: {%1.1f} [%1.1f], Error: %1.6f \n",
			//	nnInput[0], nnInput[1],
			//	nnOut[0], nnIdealOut[0],
			//	nn.JSGetAvgErr(nnInput, nnIdealOut));

			///*float avgErr = 0.0f;
			//avgErr += nn.calTotalError({0.1f, 0.1f}, {0.1f});
			//avgErr += nn.calTotalError({0.1f, 0.9f}, {0.9f});
			//avgErr += nn.calTotalError({0.9f, 0.1f}, {0.9f});
			//avgErr += nn.calTotalError({0.9f, 0.9f}, {0.1f});
			//avgErr /= 4;
			//printf("[%4u] %1.6f\n", trainNo, avgErr);*/
			////printf("[%4u] %1.6f\n", trainNo, nn.calTotalError(nnInput, nnIdealOut));
		}
		
	}
	printf("\n");
	if (false)	{
		fvec nnOut;
		fvec nnInput(2);
		fvec nnIdealOut(1);

		nnInput = {0.1f, 0.1f};
		nnIdealOut = {0.1f};
		nnOut = nn.feedForward(nnInput);
		printf("Input: {%1.1f, %1.1f}, Output: {%1.1f} [%1.1f], Error: %1.6f \n", 
			nnInput[0], nnInput[1],
			nnOut[0], nnIdealOut[0],
			nn.calTotalError(nnInput, nnIdealOut));

		nnInput = {0.1f, 0.9f};
		nnIdealOut = {0.9f};
		nnOut = nn.feedForward(nnInput);
		printf("Input: {%1.1f, %1.1f}, Output: {%1.1f} [%1.1f], Error: %1.6f \n",
			nnInput[0], nnInput[1],
			nnOut[0], nnIdealOut[0],
			nn.calTotalError(nnInput, nnIdealOut));

		nnInput = {0.9f, 0.1f};
		nnIdealOut = {0.9f};
		nnOut = nn.feedForward(nnInput);
		printf("Input: {%1.1f, %1.1f}, Output: {%1.1f} [%1.1f], Error: %1.6f \n",
			nnInput[0], nnInput[1],
			nnOut[0], nnIdealOut[0],
			nn.calTotalError(nnInput, nnIdealOut));

		nnInput = {0.9f, 0.9f};
		nnIdealOut = {0.1f};
		nnOut = nn.feedForward(nnInput);
		printf("Input: {%1.1f, %1.1f}, Output: {%1.1f} [%1.1f], Error: %1.6f \n",
			nnInput[0], nnInput[1],
			nnOut[0], nnIdealOut[0],
			nn.calTotalError(nnInput, nnIdealOut));
	}
	printf("\n");
	nn.printInfo();

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

				fvec out = nn.feedForward(in);
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
					
						avErr += nn.calTotalError(nnInput, nnIdealOut) / LERN_CYCLES;
						nn.train(nnInput, nnIdealOut);
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