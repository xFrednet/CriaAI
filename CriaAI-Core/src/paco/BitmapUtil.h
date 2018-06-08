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

#include "../util/FBmpFile.h"

#include "../Common.hpp"

#include "PaCoContext.h"

/*
* These functions are defined inside the paco specific files.
*
* The paco files are inside the specific paco directory, the file name
* start this the paco specific prefix and continues with "BitmapUtil"
*/
namespace cria_ai { namespace paco {

	void              CRFBmpConvertBMPToFBMPData(byte const* byteData, float* outFloatData, uint valueCount);
	void              CRFBmpConvertFBMPToBMPData(float const* floatData, byte* outByteData, uint valueCount);

	void              CRFBmpConvertToFPP(CR_FBMP const* inBmp, CR_FBMP* outBmp);
	inline CR_FBMP*   CRFBmpConvertToFPP(CR_FBMP const* inBmp, uint targetFpp)
	{
		/*
		 * Validation
		 */
		if (!inBmp)
			return nullptr;
		if (!(targetFpp == 1 || targetFpp == 3 || targetFpp == 4))
			return nullptr; // unsupported target bpp

		/*
		 * Output creator
		 */
		CR_FBMP* outBmp = CRFBmpCreate(inBmp->Width, inBmp->Height, targetFpp);
		CRIA_AUTO_ASSERT(outBmp, "paco::CRFBmpConvertToFPP failed to create the output bitmap.");
		if (!outBmp)
			return nullptr;

		/*
		 * Select the converter
		 */
		CRFBmpConvertToFPP(inBmp, outBmp);
		return outBmp;
	}

	void              CRFBmpScale(CR_FBMP const* inBmp, CR_FBMP* outBmp, float scale);
	inline CR_FBMP*   CRFBmpScale(CR_FBMP const* inBmp, float scale)
	{
		/*
		 * Validation
		 */
		CRIA_AUTO_ASSERT(inBmp, "CRFBmpScale: The input bitmap is null");
		if (!inBmp)
			return nullptr; //input is null bitmap
		CRIA_AUTO_ASSERT(scale != 0, "CRFBmpScale: The requested scale is, let's say stupid!");
		if (scale == 0)
			return nullptr; //invalid scale

		/*
		 * Create output
		 */
		uint width = (uint)ceilf((float)inBmp->Width * scale);
		uint height = (uint)ceilf((float)inBmp->Height * scale);
		CR_FBMP* outBmp = CRFBmpCreate(width, height, inBmp->Fpp);
		CRIA_AUTO_ASSERT(outBmp, "CRFBmpScale: The Creation of the output bitmap failed");
		if (!outBmp)
			return nullptr;

		/*
		 * Scale and return
		 */
		paco::CRFBmpScale(inBmp, outBmp, scale);
		return outBmp;
	}

	void              CRFBmpToMatf(CR_FBMP const* inBmp, CRMatrixf* outMat);
	inline CRMatrixf* CRFBmpTo1DMatf(CR_FBMP const* inBmp)
	{
		/*
		 * Validation
		 */
		if (!inBmp)
			return nullptr;

		/*
		 * Create output
		 */
		CRMatrixf* outMat = CRCreateMatrixf(1, inBmp->Width * inBmp->Height * inBmp->Fpp);
		CRIA_AUTO_ASSERT(outMat, "CRFBmpTo1DMatf: The Creation of the output matrix failed");
		if (!outMat)
			return nullptr;

		/*
		 * Convert and return
		 */
		CRFBmpToMatf(inBmp, outMat);

		return outMat;
	}
	inline CRMatrixf* CRFBmpTo2DMatf(CR_FBMP const* inBmp)
	{
		/*
		* Validation
		*/
		if (!inBmp)
			return nullptr;

		/*
		* Create output
		*/
		CRMatrixf* outMat = CRCreateMatrixf(inBmp->Width * inBmp->Fpp, inBmp->Height);
		CRIA_AUTO_ASSERT(outMat, "CRFBmpTo1DMatf: The Creation of the output matrix failed");
		if (!outMat)
			return nullptr;

		/*
		* Convert and return
		*/
		CRFBmpToMatf(inBmp, outMat);

		return outMat;
	}
}}
