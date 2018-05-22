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

	void CRNeuronNetwork::process(CRMatrixf const* data)
	{
		CRMatrixf const* processData = data;
		for (CRNeuronLayerPtr& ptr : m_LayerList)
		{
			ptr->processData(processData);
			processData = ptr->getOutput();
		}
	}
}}