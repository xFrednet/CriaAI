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

#ifndef CR_BP_WEIGHT_LEARN_RATE
#	define CR_BP_WEIGHT_LEARN_RATE 0.5f
#endif

#ifndef CR_BP_BIAS_LERN_RATE
#	define CR_BP_BIAS_LERN_RATE 0.5f
#endif

namespace cria_ai { namespace network {
	
	class CRNeuronNetwork;

	/* //////////////////////////////////////////////////////////////////////////////// */
	// // CR_NN_BP_INFO //
	/* //////////////////////////////////////////////////////////////////////////////// */
	typedef struct CR_NN_BP_INFO_ {

		// Struct info
		uint LayerCount;
		uint BatchSize;
		uint TotalBPsCount;

		// bp info
		float       AverageCost;
		CRMatrixf** NeuronBlame;
		CRMatrixf** BiasChanges;
		CRMatrixf** WeightChanges;

	} CR_NN_BP_INFO;

	CR_NN_BP_INFO* CRCreateBPInfo(CRNeuronNetwork const* targetNN, uint batchSize);
	void CRDeleteBPInfo(CR_NN_BP_INFO* bpInfo);

	void CRResetBPInfo(CR_NN_BP_INFO* bpInfo);

	/* //////////////////////////////////////////////////////////////////////////////// */
	// // CR_NN_BP_LAYER_OUTPUTS //
	/* //////////////////////////////////////////////////////////////////////////////// */
	typedef struct CR_NN_BP_LAYER_OUTPUTS_ {
		uint LayerCount; 
		uint BatchSize;
		CRMatrixf** LayerOutputs; /* [input layer] + [hidden layers] + [output layer] */
	} CR_NN_BP_LAYER_OUTPUTS;

	CR_NN_BP_LAYER_OUTPUTS* CRCreateBPLayerOut(CRNeuronNetwork const* targetNN);
	void CRDeleteBPLayerOut(CR_NN_BP_LAYER_OUTPUTS* lOutInfo);

	float CRGetCost(CRMatrixf const* actualOutput, CRMatrixf const* idealOutput);
	/*
	 * Can run in a different thread
	 */
	void CRBackprop(CR_NN_BP_INFO* bpInfo, CRMatrixf const* expectedOutput, 
		CR_NN_BP_LAYER_OUTPUTS const* layerOutputs, CRNeuronNetwork const* network);

	/*
	 * Run this on the main network thread
	 */
	void CRApplyBackprop(CRNeuronNetwork* network, CR_NN_BP_INFO const* bpInfo);
}}
