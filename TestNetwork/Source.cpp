#include <math.h>
#include <iostream>
#include <stdlib.h>
#include <vector>

#define LERN_RATE 0.5f
#define LERN_CYCLES 1000

typedef unsigned int uint;
typedef std::vector<float> fvec;

float GetRandFloat()
{
	return (((float)rand() / (float)RAND_MAX) * 2) - 1;
}

class NN;
class Nlayer;

class Neuron
{
	friend class NN;
	friend class NLayer;

	float m_Bias;
	float m_LastOutput;
	fvec m_Weights;
	fvec m_Inputs;

public:

	Neuron(float bias, uint weightCount)
		: m_Bias(bias),
		m_LastOutput(0.0f),
		m_Weights(weightCount),
		m_Inputs(weightCount)
	{
		for (uint weightNo = 0; weightNo < weightCount; weightNo++) 
		{
			m_Weights[weightNo] = GetRandFloat();
		}
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
		m_LastOutput = activFunc(calNetOutput(inputs));
		return m_LastOutput;
	}

	/*
	 * BP
	 */
	float calError(float idealOut) const
	{
		float errorSqrt = idealOut - m_LastOutput;
		return 0.5f * (errorSqrt * errorSqrt);
	}

	float calculate_pd_error_wrt_output(float idealOut) const
	{
		return -(idealOut - m_LastOutput);
	}
	float calculate_pd_total_net_input_wrt_input() const
	{
		return m_LastOutput * (1 - m_LastOutput);
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

	/*
	 * getters
	 */
	float getBias() const
	{
		return m_Bias;
	}
	float getLastOutput() const
	{
		return m_LastOutput;
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
		m_HiddenLayer.m_Neurons[0].m_Bias = 0.35f;
		m_HiddenLayer.m_Neurons[0].m_Weights = {0.15f, 0.2f};
		m_HiddenLayer.m_Neurons[1].m_Bias = 0.35f;
		m_HiddenLayer.m_Neurons[1].m_Weights = {0.25f, 0.30f};

		m_OutputLayer.m_Neurons[0].m_Bias = 0.6f;
		m_OutputLayer.m_Neurons[0].m_Weights = {0.40f, 0.45f};
		m_OutputLayer.m_Neurons[1].m_Bias = 0.6f;
		m_OutputLayer.m_Neurons[1].m_Weights = {0.50f, 0.55f};
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

				m_OutputLayer.m_Neurons[onNo].m_Weights[weightNo] -= LERN_RATE * pdErrWeight;
			}
		}

		// Update hidden layer weights
		for (uint hnNo = 0; hnNo < hidLayNCount; hnNo++) 
		{
			for (uint weightNo = 0; weightNo < m_InputCount; weightNo++)
			{
				float pdErrWeight = pfHidErrNetInput[hnNo] *
					m_HiddenLayer.m_Neurons[hnNo].calculate_pd_total_net_input_wrt_weight(weightNo);

				m_HiddenLayer.m_Neurons[hnNo].m_Weights[weightNo] -= LERN_RATE * pdErrWeight;
			}
		}
	}
};

int main()
{
	printf("##########################################\n");
	printf("Hello and welcome to this NeuronNetwork!!!\n");
	printf("- XOR gates are fun but AI is the future -\n");
	printf("##########################################\n");
	printf("\n");
	NN nn(2, 2, 2);
	nn.setBiasesAndWeights();

	for (uint trainNo = 0; trainNo < 1000; trainNo++)
	{
		fvec nnInput(2);
		nnInput[0] = ((rand() % 2) == 0) ? 0.1f : 0.9f;
		nnInput[1] = ((rand() % 2) == 0) ? 0.1f : 0.9f;
		fvec nnIdealOut(2);// = {0.01f, 0.99f};
		nnIdealOut[0] = (nnInput[0] == nnInput[1]) ? 0.1f : 0.9f;
		nnIdealOut[1] = (nnInput[0] != nnInput[1]) ? 0.1f : 0.9f;

		nnInput = {0.1f, 0.9f};
		nnIdealOut = {0.1f, 0.9f};

		fvec nnOut = nn.feedForward(nnInput);
		if (trainNo % 50 == 0)
			printf("[%3u] %1.6f [%1.6f, %1.6f] \n", trainNo,  nn.calTotalError(nnInput, nnIdealOut), nnOut[0], nnOut[1]);
		nn.train(nnInput, nnIdealOut);
		
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