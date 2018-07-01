#pragma once

#include "../Common.hpp"

#include "NetworkUtil.h"

#include "NeuronLayer.h"
#include "Backprop.h"

namespace cria_ai { namespace network {
	
	class CRNeuronNetwork 
	{
	private:
		std::list<CRNeuronLayerPtr> m_LayerList;

	public:

		void addLayer(const CRNeuronLayerPtr& layer);
		void removeLayer(const CRNeuronLayerPtr& layer);

		void initRandom();

		void process(CRMatrixf const* data, CR_NN_BP_LAYER_OUTPUTS* outputs = nullptr);

		uint getLayerCount() const;
		std::list<CRNeuronLayer*> getLayers();
		std::list<CRNeuronLayer const*> getLayers() const;
	};

}}
