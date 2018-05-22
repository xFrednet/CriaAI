#pragma once

#include "../Common.hpp"

#include "NetworkUtil.h"

#include "NeuronLayer.h"

namespace cria_ai { namespace network {
	
	class CRNeuronNetwork 
	{
	private:
		std::list<CRNeuronLayerPtr> m_LayerList;

	public:

		void addLayer(const CRNeuronLayerPtr& layer);
		void removeLayer(const CRNeuronLayerPtr& layer);

		void initRandom();

		void process(CRMatrixf const* data);
	};

}}