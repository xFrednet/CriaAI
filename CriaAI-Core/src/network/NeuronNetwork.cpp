#include "NeuronNetwork.h"

namespace cria_ai { namespace network {
	
	void CRNeuronNetwork::addLayer(const CRNeuronLayerPtr& layer)
	{
		if (layer.get())
			m_LayerList.push_back(layer);
	}
	void CRNeuronNetwork::initRandom()
	{
		for (CRNeuronLayerPtr& ptr : m_LayerList) {
			ptr->intiRandom();
		}
	}

	CR_MATF* CRNeuronNetwork::feedForward(CR_MATF const* data)
	{
		/*
		 * Validation 
		 */
		if (!data)
		{
			return nullptr;
		}

		/*
		 * feed the layers
		 */
		CR_MATF const* nextLayerInput = data;
		for (uint layerNo = 0; layerNo < m_LayerList.size(); layerNo++)
		{
			m_LayerList[layerNo]->feedForward(nextLayerInput);
			nextLayerInput = m_LayerList[layerNo]->getOutput();
		}

		/*
		 * Hand back the output
		 */
		CR_MATF* output = CRMatFCreate(nextLayerInput->Cols, nextLayerInput->Rows);
		if (!output)
			return nullptr;
		CR_MATF_COPY_DATA(output, nextLayerInput);
		return output;
	}
	void CRNeuronNetwork::train(CR_MATF* data, CR_MATF* idealOutput)
	{
		/*
		 * Validation
		 */
		if (!data || !idealOutput)
			return; // What else can I do??? Tell me!!!
		
		/*
		 * Nom nom nom, I feed the network and I'm ashamed of these "jokes"
		 */
		CR_MATF* output = feedForward(data);

		/*
		 * BackProp the output layer
		 */
		uint layerCount = m_LayerList.size();
		CRNeuronLayerPtr& outLayer = m_LayerList[layerCount - 1];
		uint currentNeuronCount = outLayer->getNeuronCount();

		CR_MATF* currentLayerBlame = CRMatFCreate(1, currentNeuronCount);

		for (uint neuronNo = 0; neuronNo < currentNeuronCount; neuronNo++)
		{
			currentLayerBlame->Data[neuronNo] = idealOutput->Data[neuronNo] - output->Data[neuronNo];
		}
		outLayer->train(currentLayerBlame, CR_BP_WEIGHT_LEARN_RATE, CR_BP_BIAS_LERN_RATE);
		CR_MATF* prevLayerBlame = outLayer->blamePreviousLayer(currentLayerBlame);

		/*
		 * BackProp the hidden layer/layers
		 */
		for (int layerNo = (int)layerCount - 2; layerNo >= 0; layerNo--)
		{
			CRMatFDelete(currentLayerBlame);
			currentLayerBlame = prevLayerBlame;

			CRNeuronLayerPtr& layer = m_LayerList[layerNo];
			
			layer->train(currentLayerBlame, CR_BP_WEIGHT_LEARN_RATE, CR_BP_BIAS_LERN_RATE);

			prevLayerBlame = layer->blamePreviousLayer(currentLayerBlame);
		}

		/*
		 * Clean up
		 */
		CRMatFDelete(output);
		CRMatFDelete(currentLayerBlame);
		CRMatFDelete(prevLayerBlame);
	}

	uint CRNeuronNetwork::getLayerCount() const
	{
		return (uint)m_LayerList.size();
	}
	std::vector<CRNeuronLayer*> CRNeuronNetwork::getLayers()
	{
		std::vector<CRNeuronLayer*> layers;

		for (const CRNeuronLayerPtr& ptr : m_LayerList) {
			layers.push_back(ptr.get());
		}

		return layers;
	}
	std::vector<CRNeuronLayer const*> CRNeuronNetwork::getLayers() const
	{
		std::vector<CRNeuronLayer const*> constList;

		for (const CRNeuronLayerPtr& ptr : m_LayerList)
		{
			constList.push_back(ptr.get());
		}

		return constList;
	}

}}
