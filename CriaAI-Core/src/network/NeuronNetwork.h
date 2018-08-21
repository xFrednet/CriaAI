#pragma once

#include "../Common.hpp"

#include "NeuronLayer.h"

#ifndef CR_BP_WEIGHT_LEARN_RATE
#	define CR_BP_WEIGHT_LEARN_RATE 0.5f
#endif

#ifndef CR_BP_BIAS_LERN_RATE
#	define CR_BP_BIAS_LERN_RATE 0.5f
#endif

namespace cria_ai { namespace network {
	
	class CRNeuronNetwork 
	{
	private:
		std::vector<CRNeuronLayerPtr> m_LayerList;

	public:

		void addLayer(const CRNeuronLayerPtr& layer);

		void initRandom();

		CR_MATF* feedForward(CR_MATF const* data);
		void train(CR_MATF* data, CR_MATF* idealOutput);

		uint getLayerCount() const;
		std::vector<CRNeuronLayer*> getLayers();
		std::vector<CRNeuronLayer const*> getLayers() const;
	};

}}
