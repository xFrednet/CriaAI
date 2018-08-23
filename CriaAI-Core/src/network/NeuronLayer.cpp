/******************************************************************************
* Cria  - The worst artificial intelligence on the market.                    *
*         <https://github.com/xFrednet/CriaAI>                                *
*                                                                             *
* =========================================================================== *
* Copyright (C) 2017, 2018, xFrednet <xFrednet@gmail.com>                     *
*                                                                             *
* This software is provided 'as-is', without any express or implied warranty. *
* In no event will the authors be held liable for any damages arising from    *
* the use of this software.                                                   *
*                                                                             *
* Permission is hereby granted, free of charge, to anyone to use this         *
* software for any purpose, including the rights to use, copy, modify,        *
* merge, publish, distribute, sublicense, and/or sell copies of this          *
* software, subject to the following conditions:                              *
*                                                                             *
*   1.  The origin of this software must not be misrepresented; you           *
*       must not claim that you wrote the original software. If you           *
*       use this software in a product, an acknowledgment in the              *
*       product documentation would be greatly appreciated but is not         *
*       required                                                              *
*                                                                             *
*   2.  Altered source versions should be plainly marked as such, and         *
*       must not be misrepresented as being the original software.            *
*                                                                             *
*   3.  This code should not be used for any military or malicious            *
*       purposes.                                                             *
*                                                                             *
*   4.  This notice may not be removed or altered from any source             *
*       distribution.                                                         *
*                                                                             *
******************************************************************************/
#include "NeuronLayer.h"

namespace cria_ai { namespace network {

	CRNeuronLayer::CRNeuronLayer(uint inputCount, uint neuronCount, crresult* result)
		: m_InputCount(inputCount),
		m_NeuronCount(neuronCount),
		m_Input(nullptr),
		m_Output(nullptr),
		m_Weights(nullptr),
		m_Bias(nullptr),
		m_ActivationFunc(nullptr),
		m_ActivationFuncInv(nullptr)
	{
		/*
		 * Validation
		 */
		if (inputCount == 0 || neuronCount == 0)
		{
			if (result)
			{
				if (neuronCount == 0)
					*result = CRRES_ERR_NN_LAYER_INVALID_NEURON_COUNT;
				else 
					*result = CRRES_ERR_INVALID_ARGUMENTS;
			}
			return;
		}

		/*
		 * init matrices
		 */
		m_Input   = CRMatFCreate(1           , m_InputCount);
		m_Output  = CRMatFCreate(1           , m_NeuronCount);

		m_Weights = CRMatFCreate(m_InputCount, m_NeuronCount);
		m_Bias    = CRMatFCreate(1           , m_NeuronCount);

		if (!m_Input || !m_Output || !m_Weights || !m_Bias)
		{
			if (result)
				*result = CRRES_ERR_NN_LAYER_MATRICIES_INIT_FAILED;
			
			CR_MATF_DELETE_IF_VALID(m_Input);
			CR_MATF_DELETE_IF_VALID(m_Output);

			CR_MATF_DELETE_IF_VALID(m_Weights);
			CR_MATF_DELETE_IF_VALID(m_Bias);

			return;
		}

		/*
		 * Everything worked yay
		 */
		if (result)
			*result = CRRES_OK;
	}
	CRNeuronLayer::~CRNeuronLayer()
	{
		/*
		 * delete stuff
		 */
		CR_MATF_DELETE_IF_VALID(m_Input);
		CR_MATF_DELETE_IF_VALID(m_Output);

		CR_MATF_DELETE_IF_VALID(m_Weights);
		CR_MATF_DELETE_IF_VALID(m_Bias);

	}

	void CRNeuronLayer::intiRandom()
	{
		if (m_Weights)
			CRMatFFillRand(m_Weights);
		if (m_Bias)
			CRMatFFillRand(m_Bias);
	}

	void CRNeuronLayer::setActivationFunc(paco::cr_activation_func activationFunc,
		paco::cr_activation_func_inv activationFuncInv)
	{
		if (!activationFunc || !activationFuncInv)
		{
			m_ActivationFunc    = nullptr;
			m_ActivationFuncInv = nullptr;

			return;
		}

		m_ActivationFunc = activationFunc;
		m_ActivationFuncInv = activationFuncInv;
	}

	void CRNeuronLayer::feedForward(CR_MATF const* data)
	{
		/*
		 * Validation
		 */
		if (!data)
		{
			CR_MATF_FILL_ZERO(m_Input);
			CR_MATF_FILL_ZERO(m_Output);

			return;
		}

		/*
		 * feed the matrices
		 */
		CR_MATF_COPY_DATA(m_Input, data);
		CR_MATF* wOut = CRMatFMul(m_Input, m_Weights);
		CR_MATF* bOut = CRMatFAdd(wOut   , m_Bias);
		m_ActivationFunc(bOut, m_Output);
		
	}

	void CRNeuronLayer::train(CR_MATF* neuronBlame, float weightLernRate, float biasLernRate)
	{
		/*
		 * Validation
		 */
		if (!neuronBlame)
			return;

		/*
		 * get the Inverse output
		 */
		CR_MATF* invOutput = CRMatFCreate(m_Output->Cols, m_Output->Rows);
		if (!invOutput)
			return;
		m_ActivationFuncInv(m_Output, invOutput);

		/*
		 * loop through the neurons
		 */
		for (uint neuronNo = 0; neuronNo < m_NeuronCount; neuronNo++) 
		{
			float bpOutWrtNet = invOutput->Data[neuronNo];
			float bpErrWrtNet = neuronBlame->Data[neuronNo] * bpOutWrtNet;

			/*
			 * Loop through the weights
			 */
			for (uint weightNo = 0; weightNo < m_InputCount; weightNo++) 
			{
				float weightErr = m_Input->Data[weightNo] * bpErrWrtNet;
				float weightChange = weightErr * weightLernRate;
				
				m_Weights->Data[CR_MATF_VALUE_INDEX(weightNo, neuronNo, m_Weights)] += weightChange;
			}

			/*
			 * Update bias
			 */
			m_Bias->Data[neuronNo] += bpErrWrtNet * biasLernRate;
		}

		/*
		 * Call the cleanup crew
		 */
		CRMatFDelete(invOutput);
	}
	CR_MATF* CRNeuronLayer::blamePreviousLayer(CR_MATF* layerBlame) const
	{
		/*
		 * Validation
		 */
		if (!layerBlame)
			return nullptr;

		/*
		 * Create the output matrix
		 */
		CR_MATF* prevBlame = CRMatFCreate(1, m_InputCount);
		if (!prevBlame)
			return nullptr; 
		CR_MATF_FILL_ZERO(prevBlame);

		/*
		 * Blame the previous layer
		 */
		for (uint neuronNo = 0; neuronNo < m_NeuronCount; neuronNo++) 
		{
			for (uint weightNo = 0; weightNo < m_InputCount; weightNo++)
			{
				float weightBlame = m_Weights->Data[CR_MATF_VALUE_INDEX(weightNo, neuronNo, m_Weights)] * layerBlame->Data[neuronNo];
				prevBlame->Data[weightNo] += weightBlame;
			}
		}

		/*
		 * Return
		 */
		return prevBlame;
	}

	/*
	* getters
	*/
	uint CRNeuronLayer::getInputCount() const
	{
		return m_InputCount;
	}
	uint CRNeuronLayer::getNeuronCount() const
	{
		return m_NeuronCount;
	}

	CR_MATF* CRNeuronLayer::getOutput()
	{
		return m_Output;
	}
	CR_MATF const* CRNeuronLayer::getOutput() const
	{
		return m_Output;
	}
	CR_MATF* CRNeuronLayer::getWeights()
	{
		return m_Weights;
	}
	CR_MATF const* CRNeuronLayer::getWeights() const
	{
		return m_Weights;
	}
	CR_MATF* CRNeuronLayer::getBias()
	{
		return m_Bias;
	}
	CR_MATF const* CRNeuronLayer::getBias() const
	{
		return m_Bias;
	}

	paco::cr_activation_func CRNeuronLayer::getActivationFunc() const
	{
		return m_ActivationFunc;
	}
	paco::cr_activation_func_inv CRNeuronLayer::getActivationFuncInv() const
	{
		return m_ActivationFuncInv;
	}


}}
