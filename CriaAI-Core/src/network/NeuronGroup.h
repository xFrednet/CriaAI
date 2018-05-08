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

#include "../api/InputLogger.h"
#include "../api/InputSimulator.h"

#ifndef CR_NEURON_OUTPUT_ACTIVATION_LIMIT
#	define CR_NEURON_OUTPUT_ACTIVATION_LIMIT     crnwdec(0.5)
#endif

namespace cria_ai { namespace network {
	
	typedef enum CR_NEURON_TYPE_ {
		CR_NEURON_INPUT_LAYER        = 0x0100,
		CR_NEURON_DATA_INPUT         = 0x0101,
		CR_NEURON_RANDNOISE_INPUT    = 0x0102,

		CR_NEURON_HIDDEN_LAYER       = 0x0200,
		CR_NEURON_NORMAL             = 0x0201,
		CR_NEURON_INPUT_DELAY        = 0x0203,

		CR_NEURON_OUTPUT_LAYER       = 0x0400,
		CR_NEURON_KEYBOARD_OUTPUT    = 0x0401,
		CR_NEURON_MMOVE_OUTPUT       = 0x0402,
		CR_NEURON_MBUTTON_OUTPUT     = 0x0403
	} CR_NEURON_TYPE;

	class CRNeuronGroup
	{
	protected:
		static api::CRInputSimulator* s_InputSim;
	public:
		static crresult InitStaticMembers(api::CRInputSimulator* inputSim);
		static crresult TerminateStaticMembers();
	protected:
		uint m_NeuronCount;

		CRNeuronGroup(uint neuronCount);
	public:
		virtual ~CRNeuronGroup();

		/*
		 * Neuron stuff
		 */
		/**
		 * \brief This method processes the input data and returns the result in outData
		 * 
		 * \param inData This has to be an array of data with a minimum size of the 
		 * neuron count. Each element is processed by one neuron.
		 * \param outData This has to be an array of data with a minimum size of the 
		 * neuron count. It holds the processed data afterwards.
		 */
		virtual void processData(crnwdec const* inData, crnwdec* outData) = 0;
		/**
		 * \brief This method is the inverse process from processData. It returns the expected 
		 * input(outData) for a given output of processData(inData)
		 * 
		 * \param inData This has to be an array of data with a minimum size of the 
		 * neuron count. Each element is processed by one neuron.
		 * \param outData This has to be an array of data with a minimum size of the 
		 * neuron count. It holds the expected input data for the output data from
		 * inData.
		 */
		virtual void processDataInverse(crnwdec const* inData, crnwdec* outData) = 0;
		/**
		 * \brief This is used in the first creation of the neurons. It should initialize
		 * the member values with random values.
		 */
		virtual void randInit() = 0;

		/*
		 * Getters
		 */
		// type stuff and things
		virtual CR_NEURON_TYPE getType() = 0;
		bool isType(const CR_NEURON_TYPE& type);
	};

	typedef cr_ptr<CRNeuronGroup> CRNeuronGroupPtr;
}}