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

#include "../util/Bitmap.h"

#include "../Common.hpp"

#include "PaCoContext.h"

/*
* These functions are defined inside the paco specific files.
*
* The paco files are inside the specific paco directory, the file name
* start this the paco specific prefix and continues with "BitmapUtil"
*/
namespace cria_ai { namespace paco {

	void           CRBmpConvertToBPP(CR_BMP const* inBmp, CR_BMP* outBmp);
	inline CR_BMP* CRBmpConvertToBPP(CR_BMP const* inBmp, uint targetBpp)
	{
		/*
		 * Validation
		 */
		if (!inBmp)
			return nullptr;
		if (!(targetBpp == 1 || targetBpp == 3 || targetBpp == 4))
			return nullptr; // unsupported target bpp

		/*
		 * Output creator
		 */
		CR_BMP* outBmp = CRCreateBmp(inBmp->Width, inBmp->Height, targetBpp);
		CRIA_AUTO_ASSERT(outBmp, "paco::CRBmpConvertToBPP failed to create the output bitmap.");
		if (!outBmp)
			return nullptr;

		/*
		 * Select the converter
		 */
		CRBmpConvertToBPP(inBmp, outBmp);
		return outBmp;
	}

	void           CRBmpScale(CR_BMP const* inBmp, CR_BMP* outBmp, float scale);
	inline CR_BMP* CRBmpScale(CR_BMP const* inBmp, float scale)
	{
		/*
		 * Validation
		 */
		CRIA_AUTO_ASSERT(inBmp, "CRBmpScale: The input bitmap is null");
		if (!inBmp)
			return nullptr; //input is null bitmap
		CRIA_AUTO_ASSERT(scale != 0, "CRBmpScale: The requested scale is, let's say stupid!");
		if (scale == 0)
			return nullptr; //invalid scale

		/*
		 * Create output
		 */
		uint width = (uint)ceilf((float)inBmp->Width * scale);
		uint height = (uint)ceilf((float)inBmp->Height * scale);
		CR_BMP* outBmp = CRCreateBmp(width, height, inBmp->Bpp);
		CRIA_AUTO_ASSERT(outBmp, "CRBmpScale: The Creation of the output bitmap failed");
		if (!outBmp)
			return nullptr;

		/*
		 * Scale and return
		 */
		paco::CRBmpScale(inBmp, outBmp, scale);
		return outBmp;
	}

	void              CRBmpToMatf(CR_BMP const* inBmp, CRMatrixf* outMat);
	inline CRMatrixf* CRBmpTo1DMatf(CR_BMP const* inBmp)
	{
		/*
		 * Validation
		 */
		if (!inBmp)
			return nullptr;

		/*
		 * Create output
		 */
		CRMatrixf* outMat = CRCreateMatrixf(1, inBmp->Width * inBmp->Height * inBmp->Bpp);
		CRIA_AUTO_ASSERT(outMat, "CRBmpTo1DMatf: The Creation of the output matrix failed");
		if (!outMat)
			return nullptr;

		/*
		 * Convert and return
		 */
		CRBmpToMatf(inBmp, outMat);

		return outMat;
	}
	inline CRMatrixf* CRBmpTo2DMatf(CR_BMP const* inBmp)
	{
		/*
		* Validation
		*/
		if (!inBmp)
			return nullptr;

		/*
		* Create output
		*/
		CRMatrixf* outMat = CRCreateMatrixf(inBmp->Width * inBmp->Bpp, inBmp->Height);
		CRIA_AUTO_ASSERT(outMat, "CRBmpTo1DMatf: The Creation of the output matrix failed");
		if (!outMat)
			return nullptr;

		/*
		* Convert and return
		*/
		CRBmpToMatf(inBmp, outMat);

		return outMat;
	}
}}
