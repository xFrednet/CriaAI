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

namespace cria_ai { namespace network {

	typedef struct CR_NEURON_LIST_NODE_
	{
		CR_NEURON_LIST_NODE_* Next;
		CRNeuronGroup* Neurons;
	} CR_NEURON_LIST_NODE;

	typedef void(*cr_nw_activation_func)(CRNWMat const* input, CRNWMat* output);

	class CRNeuronLayer
	{
	protected:
		CRNWMat* m_Output;
		
		CRNWMat* m_Conections;
		CRNWMat* m_Bias;

		CR_NEURON_LIST_NODE m_NeuronList;

		//TODO function / invFunction

		uint m_NeuronCount;
		uint m_NeuronGroupCount;
	public:


		/*
		 * getters
		 */
		CRMatrixf* getOutput()
		{
			return m_Output;
		}
		CRMatrixf const* getOutput() const
		{
			return m_Output;
		}
	};

}}
