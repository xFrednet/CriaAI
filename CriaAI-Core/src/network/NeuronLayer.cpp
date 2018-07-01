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
	
	crresult CRNeuronLayer::initMatrices()
	{
		/*
		 * Cleanup
		 */
		m_IsOperational = false;

		if (m_Output || m_Weights || m_Bias)
			return CRRES_ERR_NN_LAYER_MATRICIES_NOT_NULL;

		/*
		 * Validation check
		 */
		if (m_NeuronCount == 0)
			return CRRES_ERR_NN_LAYER_INVALID_NEURON_COUNT;

		m_Output = CRCreateMatrixf(1, m_NeuronCount);
		if (!m_Output)
			return CRRES_ERR_NN_LAYER_MATRICIES_INIT_FAILED;
		if (m_PrevLayer && m_PrevLayer->getNeuronCount() != 0) // => if not first layer
		{
			// m_Weights has one column per input neuron and one row per output neuron
			m_Weights = CRCreateMatrixf(m_PrevLayer->getNeuronCount(), m_NeuronCount);
			m_Bias = CRCreateMatrixf(1, m_NeuronCount);

			if (!m_Weights || !m_Bias)
				return CRRES_ERR_NN_LAYER_MATRICIES_INIT_FAILED;
		}

		return CRRES_OK;
	}
	void CRNeuronLayer::updateIsOperational()
	{
		m_IsOperational = (
			m_Output &&
			m_NeuronCount != 0 &&
			m_ActivationFunc &&
			m_ActivationFuncInv);
	}

	CRNeuronLayer::CRNeuronLayer(CRNeuronLayer const* prevLayer, uint neuronCount, crresult* result)
		: m_PrevLayer(prevLayer), 
		m_Output(nullptr),
		m_Weights(nullptr), 
		m_Bias(nullptr),
		m_NeuronCount(neuronCount),
		m_ActivationFunc(nullptr), 
		m_ActivationFuncInv(nullptr),
		m_IsOperational(false)
	{
		crresult initRes = initMatrices();

		if (result)
			*result = initRes;
	}

	CRNeuronLayer::~CRNeuronLayer()
	{
		m_IsOperational = false;

		/*
		 * delete stuff
		 */
		if (m_Output)
		{
			CRDeleteMatrixf(m_Output);
			m_Output = nullptr;
		}
		if (m_Weights)
		{
			CRDeleteMatrixf(m_Weights);
			m_Weights = nullptr;
		}
		if (m_Bias)
		{
			CRDeleteMatrixf(m_Bias);
			m_Bias = nullptr;
		}

	}

	void CRNeuronLayer::intiRandom()
	{
		if (m_Weights)
			CRFillMatrixRand(m_Weights);
		if (m_Bias)
			CRFillMatrixRand(m_Bias);
	}

	void CRNeuronLayer::setActivationFunc(paco::cr_activation_func activationFunc,
		paco::cr_activation_func_inv activationFuncInv)
	{
		if (!activationFunc || !activationFuncInv)
		{
			m_ActivationFunc    = nullptr;
			m_ActivationFuncInv = nullptr;
			
			m_IsOperational = false;

			return;
		}

		m_ActivationFunc = activationFunc;
		m_ActivationFuncInv = activationFuncInv;

		initMatrices();
		updateIsOperational();
	}

	void CRNeuronLayer::processData(CRMatrixf const* inputData)
	{
		if (!m_IsOperational || !inputData)
		{
			memset(m_Output, 0, sizeof(float) * m_NeuronCount);
			CRIA_INFO_PRINTF("CRNeuronLayer::processData: m_Output was set to 0");
			return;
		}

		/*
		 * Process weights and biases
		 */
		if (m_PrevLayer)
		{
			if (inputData->Rows != m_Weights->Cols) {
				memset(m_Output, 0, sizeof(float) * m_NeuronCount);
				CRIA_INFO_PRINTF("CRNeuronLayer::processData: m_Output was set to 0");
				return;
			}

			StopWatch timer;
			CRNWMat* weightOut = CRMul(inputData, m_Weights);
			CRNWMat* biasOut   = CRSub(weightOut, m_Bias);

			timer.start();
			m_ActivationFunc(biasOut, m_Output);

			CRDeleteMatrixf(weightOut);
			CRDeleteMatrixf(biasOut);
		}
		else
		{
			memcpy(m_Output, inputData, sizeof(float) * inputData->Rows);
		}

	}

	void CRNeuronLayer::applyBackpropagation(CRMatrixf const* weightChange, CRMatrixf const* biasChange)
	{
		if (!weightChange || !biasChange)
			return;

		CRMatrixf* newBias = CRAdd(m_Bias, biasChange);
		CRMatrixf* newWeights = CRAdd(m_Weights, weightChange);
		CRDeleteMatrixf(m_Bias);
		CRDeleteMatrixf(m_Weights);
		m_Bias = newBias;
		m_Weights = newWeights;
	}

	/*
	* getters
	*/
	CRNWMat* CRNeuronLayer::getOutput()
	{
		return m_Output;
	}
	CRNWMat const* CRNeuronLayer::getOutput() const
	{
		return m_Output;
	}
	CRNWMat* CRNeuronLayer::getWeights()
	{
		return m_Weights;
	}
	CRNWMat const* CRNeuronLayer::getWeights() const
	{
		return m_Weights;
	}
	CRNWMat* CRNeuronLayer::getBias()
	{
		return m_Bias;
	}
	CRNWMat const* CRNeuronLayer::getBias() const
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

	uint CRNeuronLayer::getNeuronCount() const
	{
		return m_NeuronCount;
	}

}}
