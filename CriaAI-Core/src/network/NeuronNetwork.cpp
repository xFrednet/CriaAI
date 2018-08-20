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
