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
	void CRNeuronLayer::updateMatrixSizes()
	{
		/*
		 * Cleanup
		 */
		m_IsOperational = false;

		if (m_Output) {
			CRFreeMatrixf(m_Output);
			m_Output = nullptr;
		}
		if (m_PreNeuronOutput) {
			CRFreeMatrixf(m_PreNeuronOutput);
			m_PreNeuronOutput = nullptr;
		}
		if (m_Weights) {
			CRFreeMatrixf(m_Weights);
			m_Weights = nullptr;
		}
		if (m_Bias) {
			CRFreeMatrixf(m_Bias);
			m_Bias = nullptr;
		}

		/*
		 * Validation check
		 */
		if (m_NeuronCount == 0)
			return;

		m_Output = CRCreateMatrixf(1, m_NeuronCount);
		m_PreNeuronOutput = CRCreateMatrixf(1, m_NeuronCount);
		if (m_PrevLayer && m_PrevLayer->getNeuronCount() != 0) // => if not first layer
		{
			// m_Weights has one column per input neuron and one row per output neuron
			m_Weights = CRCreateMatrixf(m_PrevLayer->getNeuronCount(), m_NeuronCount);
			m_Bias = CRCreateMatrixf(1, m_NeuronCount);
		}

		/*
		 * Check if this layer is ready to operate
		 */
		updateIsOperational();
	}
	void CRNeuronLayer::updateIsOperational()
	{
		m_IsOperational = (
			m_Output &&
			m_PreNeuronOutput &&
			m_NeuronList &&
			m_NeuronCount != 0 &&
			m_NeuronGroupCount != 0 &&
			m_ActivationFunc &&
			m_ActivationFuncInv);
	}

	CRNeuronLayer::CRNeuronLayer(CRNeuronLayer const* prevLayer, crresult* result)
		: m_PrevLayer(prevLayer), 
		m_Output(nullptr), 
		m_PreNeuronOutput(nullptr),
		m_Weights(nullptr), 
		m_Bias(nullptr), 
		m_NeuronList(nullptr),
		m_NeuronCount(0),
		m_NeuronGroupCount(0), 
		m_ActivationFunc(nullptr), 
		m_ActivationFuncInv(nullptr),
		m_IsOperational(false)
	{
		if (result)
			*result = CRRES_OK;
	}

	CRNeuronLayer::~CRNeuronLayer()
	{
		m_IsOperational = false;

		/*
		 * delete stuff
		 */
		if (m_Output)
		{
			CRFreeMatrixf(m_Output);
			m_Output = nullptr;
		}
		if (m_PreNeuronOutput)
		{
			CRFreeMatrixf(m_PreNeuronOutput);
			m_PreNeuronOutput = nullptr;
		}
		if (m_Weights)
		{
			CRFreeMatrixf(m_Weights);
			m_Weights = nullptr;
		}
		if (m_Bias)
		{
			CRFreeMatrixf(m_Bias);
			m_Bias = nullptr;
		}

		/*
		 * delete node list
		 */
		CR_NEURON_LIST_NODE* currentNode = m_NeuronList;
		m_NeuronList = nullptr;
		CR_NEURON_LIST_NODE* nextNode;
		while (currentNode)
		{
			nextNode = currentNode->Next;
			delete currentNode;
			currentNode = nextNode;
		}
	}

	void CRNeuronLayer::addNeuronGroup(const CRNeuronGroupPtr& neuronGroup)
	{
		/*
		 * Validation
		 */
		if (!neuronGroup.get() || neuronGroup->getNeuronCount() == 0)
			return;

		/*
		 * Create new Node
		 */
		CR_NEURON_LIST_NODE* node = new CR_NEURON_LIST_NODE;
		if (!node)
			return;
		node->Next = nullptr;
		node->Neurons = neuronGroup;

		/*
		 * add neuron group
		 */
		m_IsOperational = false;
		//CRAdd group
		{
			if (!m_NeuronList)
			{
				m_NeuronList = node;
			} 
			else // -> m_NeuronList is valid
			{
				CR_NEURON_LIST_NODE* loopNode = m_NeuronList;
				while (loopNode->Next)
				{
					loopNode = loopNode->Next;
				}
				loopNode->Next = node;
			}
		}
		m_NeuronGroupCount++;
		m_NeuronCount     += neuronGroup->getNeuronCount();

		updateMatrixSizes();
		updateIsOperational();
	}
	void CRNeuronLayer::removeNeuronGroup(const CRNeuronGroupPtr& neuronGroup)
	{
		/*
		* Validation
		*/
		if (!neuronGroup.get() || neuronGroup->getNeuronCount() != 0 || !m_NeuronList)
			return;

		/*
		 * search node
		 */
		m_IsOperational = false;

		CR_NEURON_LIST_NODE* loopNode = m_NeuronList;
		CR_NEURON_LIST_NODE** prevNextPtr = &m_NeuronList;
		while (loopNode && loopNode->Neurons != neuronGroup)
		{
			prevNextPtr = &loopNode->Next;
			loopNode = loopNode->Next;
		}

		/*
		 * delete node if it was fount
		 */
		if (loopNode)
		{
			m_NeuronGroupCount--;
			m_NeuronCount -= loopNode->Neurons->getNeuronCount();
			*prevNextPtr = loopNode->Next; //remove node from list
			
			delete loopNode; // delete Node
		}

		/*
		 * finishing
		 */
		updateIsOperational();
	}

	void CRNeuronLayer::intiRandom()
	{
		if (m_Weights)
			CRFillMatrixRand(m_Weights);
		if (m_Bias)
			CRFillMatrixRand(m_Bias);

		CR_NEURON_LIST_NODE* node = m_NeuronList;
		while (node) {
			node->Neurons->randInit();

			node = node->Next;
		}
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

		updateMatrixSizes();
		updateIsOperational();
	}

	void CRNeuronLayer::processData(CRNWMat const* inputData)
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
		if (!m_PrevLayer)
		{
			memcpy(m_PreNeuronOutput, inputData, sizeof(float) * inputData->Rows);
		}
		else
		{
			if (inputData->Rows != m_Weights->Cols) {
				memset(m_Output, 0, sizeof(float) * m_NeuronCount);
				CRIA_INFO_PRINTF("CRNeuronLayer::processData: m_Output was set to 0");
				return;
			}

			CRNWMat* weightOut = CRMul(inputData, m_Weights);
			CRNWMat* biasOut   = CRSub(weightOut, m_Bias);
			m_ActivationFunc(biasOut, m_PreNeuronOutput);

			CRFreeMatrixf(weightOut);
			CRFreeMatrixf(biasOut);
		}


		/*
		 * Push to neurons
		 */

		/*memcpy(m_Output->Data, m_PreNeuronOutput->Data, sizeof(float) * m_NeuronCount);
		return;*/

		uint neuronNo = 0;
		CR_NEURON_LIST_NODE* node = m_NeuronList;
		while (node) 
		{
			node->Neurons->processData(&(m_PreNeuronOutput->Data[neuronNo]), &(m_Output->Data[neuronNo]));

			neuronNo = node->Neurons->getNeuronCount();
			node = node->Next;
		}

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
	uint CRNeuronLayer::getNeuronCount() const
	{
		return m_NeuronCount;
	}
	uint CRNeuronLayer::getNeuronGroupCount() const
	{
		return m_NeuronGroupCount;
	}
}}
