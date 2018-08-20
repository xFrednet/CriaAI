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

#include "../Common.hpp"

#include "../maths/Matrixf.hpp"

#include "../paco/ActivationFunctions.h"

namespace cria_ai { namespace network {

	
	/**
	 * \brief 
	 * 
	 * Structure:
	 * 
	 * [CRNeuronLayer]
	 * 
	 * FeedForward:
	 * m_ActivationFunc([m_Input] * [m_Weights] + [m_Bias]) -> [m_Output]
	 * 
	 */
	class CRNeuronLayer
	{
	protected:
		uint m_InputCount;
		uint m_NeuronCount;

		CR_MATF* m_Input;
		CR_MATF* m_Output;

		CR_MATF* m_Weights;
		CR_MATF* m_Bias;

		paco::cr_activation_func     m_ActivationFunc;
		paco::cr_activation_func_inv m_ActivationFuncInv;
	public:
		CRNeuronLayer(uint inputCount, uint neuronCount, crresult* result = nullptr);
		~CRNeuronLayer();

		void intiRandom();

		void setActivationFunc(paco::cr_activation_func activationFunc, paco::cr_activation_func_inv activationFuncInv);

		void feedForward(CR_MATF const* data);

		/*
		 * getters
		 */
		uint getInputCount() const;
		uint getNeuronCount() const;

		CR_MATF* getOutput();
		CR_MATF const* getOutput() const;

		CR_MATF* getWeights();
		CR_MATF const* getWeights() const;
		CR_MATF* getBias();
		CR_MATF const* getBias() const;

		paco::cr_activation_func getActivationFunc() const;
		paco::cr_activation_func_inv getActivationFuncInv() const;
	};

	typedef cr_ptr<CRNeuronLayer> CRNeuronLayerPtr;

}}
