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
#include "Backprop.h"
#include "NeuronLayer.h"
#include "NeuronNetwork.h"

namespace cria_ai { namespace network {
	
	/* //////////////////////////////////////////////////////////////////////////////// */
	// // CR_NN_BP_INFO //
	/* //////////////////////////////////////////////////////////////////////////////// */
	CR_NN_BP_INFO* CRCreateBPInfo(CRNeuronNetwork const* targetNN, uint batchSize)
	{
		/*
		 * Validation
		 */
		if (!targetNN || batchSize == 0)
			return nullptr;

		std::list<CRNeuronLayer const*> layers = targetNN->getLayers();
		if (layers.empty())
			return nullptr;

		CR_NN_BP_INFO* bpInfo = nullptr;
		do {
			/*
			 * allocate memory
			 */
			bpInfo = new CR_NN_BP_INFO;
			if (!bpInfo) break;
			memset(bpInfo, 0, sizeof(CR_NN_BP_INFO));

			/*
			 * Fill info
			 */
			bpInfo->LayerCount = (uint)layers.size();
			bpInfo->BatchSize = batchSize;
			bpInfo->TotalBPsCount = 0;
			bpInfo->AverageCost = 0;

			// bpInfo->BiasChanges
			bpInfo->BiasChanges = new CRMatrixf*[bpInfo->LayerCount];
			if (!bpInfo->BiasChanges) 
				break;
			memset(bpInfo->BiasChanges, 0, sizeof(CRMatrixf*) * bpInfo->LayerCount);
			
			// bpInfo->WeightChanges
			bpInfo->WeightChanges = new CRMatrixf*[bpInfo->LayerCount];
			if (!bpInfo->WeightChanges)
				break;
			memset(bpInfo->WeightChanges, 0, sizeof(CRMatrixf*) * bpInfo->LayerCount);

			/*
			 * Fill bpInfo->BiasChanges && bpInfo->WeightChanges
			 */
			uint layerNo = 0;
			for (CRNeuronLayer const* layer : layers)
			{
				if (layerNo == 0)
				{
					bpInfo->BiasChanges[layerNo] = nullptr;
					bpInfo->WeightChanges[layerNo] = nullptr;
					layerNo++;
					continue;
				}
				
				// bpInfo->BiasChanges
				CRMatrixf const* layerBias   = layer->getBias();
				bpInfo->BiasChanges[layerNo] = CRCreateMatrixf(layerBias->Cols, layerBias->Rows);
				if (!bpInfo->BiasChanges[layerNo])
					break;

				// bpInfo->WeightChanges
				CRMatrixf const* layerWeight   = layer->getWeights();
				bpInfo->WeightChanges[layerNo] = CRCreateMatrixf(layerWeight->Cols, layerWeight->Rows);
				if (!bpInfo->WeightChanges[layerNo])
					break;

				// increase layer No
				layerNo++;
			}

			if (!bpInfo->BiasChanges[bpInfo->LayerCount - 1] ||
				!bpInfo->WeightChanges[bpInfo->LayerCount - 1])
				break;

			return bpInfo;
		} while (false);

		CRDeleteBPInfo(bpInfo);

		return nullptr;
	}
	void CRDeleteBPInfo(CR_NN_BP_INFO* bpInfo)
	{
		if (!bpInfo)
			return;

		/*
		 * delete bpInfo->BiasChanges
		 */
		if (bpInfo->BiasChanges) {
			for (uint index = 0; index < bpInfo->LayerCount; index++) {

				if (bpInfo->BiasChanges[index])
					CRDeleteMatrixf(bpInfo->BiasChanges[index]);
			}

			delete[] bpInfo->BiasChanges;
		}

		/*
		 * Delete bpInfo->WeightChanges
		 */
		if (bpInfo->WeightChanges) {
			for (uint index = 0; index < bpInfo->LayerCount; index++) {
				if (!bpInfo->WeightChanges[index])
					CRDeleteMatrixf(bpInfo->WeightChanges[index]);
			}

			delete[] bpInfo->WeightChanges;
		}

		/*
		 * Delete bpInfo
		 */
		delete bpInfo;
	}

	void CRResetBPInfo(CR_NN_BP_INFO* bpInfo)
	{
		if (!bpInfo)
			return;

		bpInfo->TotalBPsCount = 0;
		bpInfo->AverageCost = 0.0f;
		for (uint layerNo = 1; layerNo < bpInfo->LayerCount; layerNo++) 
		{
			CR_MATF_FILL_ZERO(bpInfo->BiasChanges[layerNo]);
			CR_MATF_FILL_ZERO(bpInfo->WeightChanges[layerNo]);
		}
	}

	/* //////////////////////////////////////////////////////////////////////////////// */
	// // CR_NN_BP_LAYER_OUTPUTS //
	/* //////////////////////////////////////////////////////////////////////////////// */
	CR_NN_BP_LAYER_OUTPUTS* CRCreateBPLayerOut(CRNeuronNetwork const* targetNN)
	{
		/*
		 * validation
		 */
		if (!targetNN)
			return nullptr;

		std::list<CRNeuronLayer const*> layers = targetNN->getLayers();
		if (layers.empty())
			return nullptr;

		/*
		 * Create
		 */
		CR_NN_BP_LAYER_OUTPUTS* lOutInfo = nullptr;
		do
		{
			/*
			 * create CR_NN_BP_LAYER_OUTPUTS
			 */
			lOutInfo = new CR_NN_BP_LAYER_OUTPUTS;
			if (!lOutInfo)
				break;
			memset(lOutInfo, 0, sizeof(CR_NN_BP_LAYER_OUTPUTS));

			//fill members
			lOutInfo->LayerCount = (uint)layers.size();

			/*
			 * Fill LayerOutputs
			 */
			lOutInfo->LayerOutputs = new CRMatrixf*[lOutInfo->LayerCount];
			if (!lOutInfo->LayerOutputs)
				break;
			memset(lOutInfo->LayerOutputs, 0, sizeof(CRMatrixf*) * lOutInfo->LayerCount);

			uint layerNo = 0;
			for (CRNeuronLayer const* layer : layers)
			{
				CRMatrixf const* layerOutput = layer->getOutput();
				lOutInfo->LayerOutputs[layerNo] = CRCreateMatrixf(layerOutput->Cols, layerOutput->Rows);
				
				if (!lOutInfo->LayerOutputs[layerNo])
					break;

				layerNo++;
			}

			// Test if lOutInfo->LayerOutputs worked out
			if (!lOutInfo->LayerOutputs[lOutInfo->LayerCount - 1])
				break;

			return lOutInfo;
		} while (false);

		CRDeleteBPLayerOut(lOutInfo);

		return nullptr;
	}
	void CRDeleteBPLayerOut(CR_NN_BP_LAYER_OUTPUTS* lOutInfo)
	{
		if (!lOutInfo)
			return;

		if (lOutInfo->LayerOutputs)
		{
			for (uint layerNo = 0; layerNo < lOutInfo->LayerCount; layerNo++)
			{
				if (lOutInfo->LayerOutputs[layerNo])
					CRDeleteMatrixf(lOutInfo->LayerOutputs[layerNo]);
			}

			delete[] lOutInfo->LayerOutputs;
		}

		delete lOutInfo;
	}

	/* //////////////////////////////////////////////////////////////////////////////// */
	// // Backpropagation //
	/* //////////////////////////////////////////////////////////////////////////////// */
	inline float CRGetCost(CRMatrixf const* realOutput, CRMatrixf const* expectedOutput)
	{
		if (!realOutput || !expectedOutput ||
			CR_MATF_VALUE_COUNT(realOutput) != CR_MATF_VALUE_COUNT(expectedOutput))
			return 0.0f;

		float cost = 0.0f;
		for (uint index = 0; index < CR_MATF_VALUE_COUNT(realOutput); index++) 
		{
			float neuronCostSqrt = realOutput->Data[index] - expectedOutput->Data[index];
			cost += neuronCostSqrt * neuronCostSqrt;
		}

		return cost;
	}
	void CRBackpropLayer(CR_NN_BP_INFO* bpInfo, CRMatrixf const* expecOutInput,
		CR_NN_BP_LAYER_OUTPUTS const* layerOutputs, CRNeuronNetwork const* network,
		uint layerNo)
	{
		layerNo = bpInfo->LayerCount - 1;
		
		std::list<CRNeuronLayer const*> layers = network->getLayers();
		if (layers.empty())
			return;

		layers.reverse();
		CRMatrixf* expecOut = CRCreateMatrixf(expecOutInput->Cols, expecOutInput->Rows);
		CR_MATF_COPY_DATA(expecOut, expecOutInput);
		for (CRNeuronLayer const* layer : layers)
		{
			if (layerNo == 0)
				break;
			/*
			 * Names: https://www.youtube.com/watch?v=tIeHLnjs5U8&t=319s
			 * w: the weight of something
			 * b: the bias
			 * z: the layer output before the activation function
			 * a: the final layer output
			 * 
			 */

			CRMatrixf* a     = layerOutputs->LayerOutputs[layerNo];
			CRMatrixf* aPrev = layerOutputs->LayerOutputs[layerNo - 1];
			CRMatrixf* aExpecOut = CRCreateMatrixf(aPrev->Cols, aPrev->Rows);
			CRMatrixf* z     = CRCreateMatrixf(a->Cols, a->Rows);
			CRMatrixf* w     = (CRMatrixf*)layer->getWeights();
			(layer->getActivationFuncInv())(a, z);

			
			for (uint neuronNo = 0; neuronNo < CR_MATF_VALUE_COUNT(expecOut); neuronNo++)
			{
				CRIA_AUTO_ASSERT(a->Data[neuronNo] != 0 || z->Data[neuronNo] != 0 || w->Data[CR_MATF_VALUE_INDEX(0, neuronNo, w)] != 0, "Rare but can happen");
				float neuronCost = powf(a->Data[neuronNo] - expecOut->Data[neuronNo], 2);

				float divA_Z = z->Data[neuronNo];//a->Data[neuronNo] / z->Data[neuronNo];
				float divC_A = 2 * (a->Data[neuronNo] - expecOut->Data[neuronNo]);//neuronCost        / a->Data[neuronNo];

				float boomForBuckB = 1 * divC_A * divA_Z;
				bpInfo->BiasChanges[layerNo]->Data[neuronNo] += boomForBuckB / bpInfo->BatchSize / 10;
				
				for (uint weightNo = 0; weightNo < w->Cols; weightNo++)
				{
					float boomForBuckW = aPrev->Data[weightNo] * divC_A * divA_Z;
					float boomForBuckAPrev = w->Data[weightNo] * divC_A * divA_Z;
					aExpecOut->Data[weightNo] = boomForBuckAPrev;
					bpInfo->WeightChanges[layerNo]->Data[CR_MATF_VALUE_INDEX(weightNo, neuronNo, bpInfo->WeightChanges[layerNo])] += 
						boomForBuckW / bpInfo->BatchSize / 10;
				}
			}

			layerNo--;
			CRDeleteMatrixf(expecOut);
			expecOut = aExpecOut;
		}
		CRDeleteMatrixf(expecOut);
	}
	void CRBackprop(CR_NN_BP_INFO* bpInfo, CRMatrixf const* expectedOutput, 
		CR_NN_BP_LAYER_OUTPUTS const* layerOutputs, CRNeuronNetwork const* network)
	{
		if (!bpInfo       || !expectedOutput || 
			!layerOutputs || !network ||
			bpInfo->LayerCount != layerOutputs->LayerCount)
			return;

		/*
		 * Backprop
		 */

		CRBackpropLayer(bpInfo, expectedOutput, layerOutputs, network, 0);


		/*
		 * Update some values
		 */
		//float cost = CRGetCost(layerOutputs->LayerOutputs[layerOutputs->LayerCount], expectedOutput);
		//bpInfo->AverageCost = cost / bpInfo->LayerCount;
		bpInfo->TotalBPsCount++;
	}

	void CRApplyBackprop(CRNeuronNetwork* network, CR_NN_BP_INFO const* bpInfo)
	{
		if (!network || !bpInfo)
			return;

		std::list<CRNeuronLayer*> layers = network->getLayers();
		if (layers.size() != bpInfo->LayerCount)
			return;

		uint layerNo = 0;
		for (CRNeuronLayer* layer : layers) 
		{
			if (layerNo != 0)
				layer->applyBackpropagation(bpInfo->WeightChanges[layerNo], bpInfo->BiasChanges[layerNo]);
			
			layerNo++;
		}
	}
}}
