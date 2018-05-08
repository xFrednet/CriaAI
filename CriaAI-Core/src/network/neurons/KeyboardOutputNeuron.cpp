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
#include "KeyboardOutputNeuron.h"

namespace cria_ai { namespace network {

	CRKeyboardOutputNeuron::CRKeyboardOutputNeuron(uint neuronCount, CR_KEY_ID* keyIDs)
		: CRNeuronGroup(neuronCount)
	{
		m_KeyIDs = new CR_KEY_ID[m_NeuronCount];
		m_LastStates = new bool[m_NeuronCount];
		
		memcpy(m_KeyIDs    , keyIDs, sizeof(CR_KEY_ID) * m_NeuronCount);
		memset(m_LastStates, false , sizeof(bool)      * m_NeuronCount);
	}
	CRKeyboardOutputNeuron::CRKeyboardOutputNeuron(CR_KEY_ID keyID)
		: CRKeyboardOutputNeuron(1, &keyID)
	{
	}
	CRKeyboardOutputNeuron::~CRKeyboardOutputNeuron()
	{
		delete[] m_KeyIDs;
		delete[] m_LastStates;
	}

	void CRKeyboardOutputNeuron::processData(crnwdec const* inData, crnwdec* outData)
	{
		if (!s_InputSim)
		{
			memset(outData, 0, sizeof(crnwdec) * m_NeuronCount);
			return;
		}

		for (uint index = 0; index < m_NeuronCount; index++)
		{
			if (inData[index] >= CR_NEURON_OUTPUT_ACTIVATION_LIMIT)
			{
				if (!m_LastStates[index])
				{
					m_LastStates[index] = true;
					s_InputSim->setKeyState(m_KeyIDs[index], true);
				}
				outData[index] = (crnwdec)1;
			} 
			else //inData[index] < CR_NEURON_OUTPUT_ACTIVATION_LIMIT
			{
				if (m_LastStates[index])
				{
					m_LastStates[index] = false;
					s_InputSim->setKeyState(m_KeyIDs[index], false);
				}
				outData[index] = (crnwdec)0;
			}
		}
	}
	void CRKeyboardOutputNeuron::processDataInverse(crnwdec const* inData, crnwdec* outData)
	{
		for (uint index = 0; index < m_NeuronCount; index++)
		{
			outData[index] = (crnwdec)(inData[index] >= CR_NEURON_OUTPUT_ACTIVATION_LIMIT);
		}
	}
	void CRKeyboardOutputNeuron::randInit()
	{
	}

	CR_NEURON_TYPE CRKeyboardOutputNeuron::getType()
	{
		return CR_NEURON_KEYBOARD_OUTPUT;
	}
}}
