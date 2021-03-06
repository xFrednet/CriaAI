﻿/******************************************************************************
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
#include "../ActivationFunctions.h"

#ifdef CRIA_PACO_NULL

namespace cria_ai { namespace paco {
	
	void CRSigmoid(CR_MATF const* input, CR_MATF* output)
	{
		CRIA_SIGMOID_VALIDATION_CHECK(input, output);
		
		for (uint index = 0; index < CR_MATF_VALUE_COUNT(input); index++) {
			output->Data[index] = 1 / (1 + exp(-input->Data[index]));
		}
	}

	void CRSigmoidInv(CR_MATF const* input, CR_MATF* output)
	{
		CRIA_SIGMOID_VALIDATION_CHECK(input, output);

		for (uint index = 0; index < CR_MATF_VALUE_COUNT(input); index++) {
			output->Data[index] = input->Data[index] * (1.0f - input->Data[index]);
		}
	}

}}

#endif // CRIA_PACO_NULL
