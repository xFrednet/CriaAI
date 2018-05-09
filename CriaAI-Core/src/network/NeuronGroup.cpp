#include "NeuronGroup.h"

namespace cria_ai { namespace network {
	
	os::CRInputSimulator* CRNeuronGroup::s_InputSim = nullptr;

	crresult CRNeuronGroup::InitStaticMembers(os::CRInputSimulator* inputSim)
	{
		if (!inputSim)
			return CRRES_ERR_INVALUD_ARGUMENTS;

		if (s_InputSim)
			return CRRES_ERR_STATIC_VAR_IS_ALREADY_VALID;

		s_InputSim = inputSim;

		return CRRES_OK;
	}
	crresult CRNeuronGroup::TerminateStaticMembers()
	{
		if (!s_InputSim)
			return CRRES_OK_STATIC_INSTANCE_IS_NULL;

		s_InputSim = nullptr;

		return CRRES_OK;
	}

	CRNeuronGroup::CRNeuronGroup(uint neuronCount)
		: m_NeuronCount(neuronCount)
	{
	}
	
	CRNeuronGroup::~CRNeuronGroup()
	{
	}

	bool CRNeuronGroup::isType(const CR_NEURON_TYPE& type)
	{
		return ((getType() & type) != 0);//black magic
	}
}}