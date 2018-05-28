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

/*
 * These functions are defined inside the paco specific files.
 * 
 * The paco files are inside the specific paco directory, the file name
 * start this the paco specific prefix and continues with "AcrivationFunctions"
 */

#include "../network/NetworkUtil.h"

#define CRIA_SIGMOID_VALIDATION_CHECK(inMap, outMat) \
if (input->Cols != output->Cols || \
	input->Rows != output->Rows) { \
	memset(output->Data, 0, sizeof(crnwdec) * output->Cols * output->Rows); \
	return; \
}

namespace cria_ai { namespace paco {
	
	typedef void(*cr_activation_func) (CRNWMat const* input, CRNWMat* output);
	typedef void(*cr_activation_func_inv) (CRNWMat const* input, CRNWMat* output);

	/**
	 * \brief A activation function
	 * 
	 * Equation:     1 / (1 + e^-x) = r
	 * Output Range: (0 < x < 1)
	 * 
	 * \param input  A matrix containing values for processing.
	 * \param output A matrix that holds the output values.
	 */
	void CRSigmoid(CRNWMat const* input, CRNWMat* output);
	/**
	* \brief A inverse activation function
	*
	* Equation:    -ln((1/r) - 1) = x
	* Input Range: (0 < r < 1)
	*
	* \param input  A matrix containing values for processing.
	* \param output A matrix that holds the output values.
	*/
	void CRSigmoidInv(CRNWMat const* input, CRNWMat* output);

}}
