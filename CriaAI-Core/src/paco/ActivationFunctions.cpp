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
#include "ActivationFunctions.h"

#ifdef CRIA_PACO_CUDA
#	include "cuda/CuActivationFunctions.cuh"
#endif

#ifdef CRIA_PACO_NULL
#	include "null/NuActivationFunctions.h"

#	define CRIA_IF_NULL(x)             x
#else
#	define CRIA_IF_NULL(x)
#endif

namespace cria_ai { namespace paco {
	
	void CRSigmoidInv(CRNWMat const* input, CRNWMat* output)
	{
		if (input->Cols != output->Cols ||
			input->Rows != output->Rows) 
		{
			memset(output->Data, 0, sizeof(crnwdec) * output->Cols * output->Rows);
			return;
		}

#ifdef CRIA_PACO_CUDA
		cuda::CRCuSigmoidInv<<<CR_CUDA_AF_BLOCK_COUNT, CR_CUDA_AF_THREAD_COUNT>>>(input, output);
		cudaDeviceSynchronize();
#elif CRIA_PACO_NULL
		CRIA_IF_NULL(null::CRNuSigmoid(input, output));
#else
		memset(output, 0, sizeof(crnwdec) * output->Cols * output->Rows);
#endif

	}

	void CRSigmoid(CRNWMat const* input, CRNWMat* output)
	{
		if (input->Cols != output->Cols ||
			input->Rows != output->Rows) 
		{
			memset(output->Data, 0, sizeof(crnwdec) * output->Cols * output->Rows);
			return;
		}

#ifdef CRIA_PACO_CUDA
		cuda::CRCuSigmoid<<<CR_CUDA_AF_BLOCK_COUNT, CR_CUDA_AF_THREAD_COUNT >>>(input, output);
		cudaDeviceSynchronize();
#elif CRIA_PACO_NULL
		CRIA_IF_NULL(null::CRNuSigmoidInv(input, output));
#else
		memset(output, 0, sizeof(crnwdec) * output->Cols * output->Rows);
#endif

	}
}}