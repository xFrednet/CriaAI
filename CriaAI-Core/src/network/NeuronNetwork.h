#pragma once

#include "../Common.hpp"

#include "NetworkUtil.h"

#include "NeuronLayer.h"
#include "Backprop.h"

namespace cria_ai { namespace network {
	
	class CRNeuronNetwork 
	{
	private:
		std::vector<CRNeuronLayerPtr> m_LayerList;

	public:

		void addLayer(const CRNeuronLayerPtr& layer);

		void initRandom();

		void process(CR_MATF const* data, CR_NN_BP_LAYER_OUTPUTS* outputs = nullptr);

		uint getLayerCount() const;
		std::vector<CRNeuronLayer*> getLayers();
		std::vector<CRNeuronLayer const*> getLayers() const;
	};

}}
