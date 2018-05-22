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
#pragma once

#include "NetworkUtil.h"

#include "../maths/Matrixf.hpp"

#include "NeuronGroup.h"

#include "../paco/ActivationFunctions.h"

namespace cria_ai { namespace network {

	typedef struct CR_NEURON_LIST_NODE_
	{
		CR_NEURON_LIST_NODE_* Next;
		CRNeuronGroupPtr Neurons;
	} CR_NEURON_LIST_NODE;

	
	/**
	 * \brief 
	 * 
	 * Structure:
	 * 
	 * [CRNeuronLayer]
	 * 
	 * m_ActivationFunc([input] * [m_Weights] - [m_Bias]) -> [Neurons->processData] -> [m_Output]
	 * 
	 */
	class CRNeuronLayer
	{
	protected:
		CRNeuronLayer const* m_PrevLayer;

		CRNWMat* m_Output;
		CRNWMat* m_PreNeuronOutput;

		CRNWMat* m_Weights;
		CRNWMat* m_Bias;

		CR_NEURON_LIST_NODE* m_NeuronList;

		uint m_NeuronCount;
		uint m_NeuronGroupCount;

		paco::cr_activation_func     m_ActivationFunc;
		paco::cr_activation_func_inv m_ActivationFuncInv;

		bool m_IsOperational;

		void updateMatrixSizes();
		void updateIsOperational();
	public:
		/**
		 * \brief 
		 * 
		 * \param prevLayer This is a pointer to the previous layer, this may only
		 * be null if the created Layer is the first layer. In all other 
		 * circumstances this pointer has to be valid.
		 * 
		 * \param result This is a pointer to retrieve the result of the creation operations.
		 */
		CRNeuronLayer(CRNeuronLayer const* prevLayer, crresult* result = nullptr);
		~CRNeuronLayer();

		void addNeuronGroup(const CRNeuronGroupPtr& neuronGroup);
		void removeNeuronGroup(const CRNeuronGroupPtr& neuronGroup);
		void intiRandom();

		void setActivationFunc(paco::cr_activation_func activationFunc, paco::cr_activation_func_inv activationFuncInv);

		void processData(CRNWMat const* inputData);

		/*
		 * getters
		 */
		CRNWMat* getOutput();
		CRNWMat const* getOutput() const;
		CRNWMat* getWeights();
		CRNWMat const* getWeights() const;
		CRNWMat* getBias();
		CRNWMat const* getBias() const;

		uint getNeuronCount() const;
		uint getNeuronGroupCount() const;
	};

	typedef cr_ptr<CRNeuronLayer> CRNeuronLayerPtr;

}}
