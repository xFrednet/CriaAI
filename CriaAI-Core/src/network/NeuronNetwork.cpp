#include "NeuronNetwork.h"

namespace cria_ai { namespace network {
	
	void CRNeuronNetwork::addLayer(const CRNeuronLayerPtr& layer)
	{
		if (layer.get())
			m_LayerList.push_back(layer);
	}
	void CRNeuronNetwork::removeLayer(const CRNeuronLayerPtr& layer)
	{
		if (layer.get())
			m_LayerList.remove(layer);
	}
	void CRNeuronNetwork::initRandom()
	{
		for (CRNeuronLayerPtr& ptr : m_LayerList) {
			ptr->intiRandom();
		}
	}

	void CRNeuronNetwork::process(CRMatrixf const* data, CR_NN_BP_LAYER_OUTPUTS* outputs)
	{
		uint outputNo = 0;
		CRMatrixf const* processData = data;

		for (CRNeuronLayerPtr& ptr : m_LayerList)
		{
			ptr->processData(processData);
			processData = ptr->getOutput();

			if (outputs)
				CR_MATF_COPY_DATA(outputs->LayerOutputs[outputNo], processData);

			outputNo++;
		}
	}

	uint CRNeuronNetwork::getLayerCount() const
	{
		return m_LayerList.size();
	}
	std::list<CRNeuronLayer*> CRNeuronNetwork::getLayers()
	{
		std::list<CRNeuronLayer*> layers;

		for (const CRNeuronLayerPtr& ptr : m_LayerList) {
			layers.push_back(ptr.get());
		}

		return layers;
	}
	std::list<CRNeuronLayer const*> CRNeuronNetwork::getLayers() const
	{
		std::list<CRNeuronLayer const*> constList;

		for (const CRNeuronLayerPtr& ptr : m_LayerList)
		{
			constList.push_back(ptr.get());
		}

		return constList;
	}
}}
