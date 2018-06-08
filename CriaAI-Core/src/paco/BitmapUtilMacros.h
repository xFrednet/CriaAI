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

#include "BitmapUtil.h"

/*
 * CRFBmpConvertBMPToFBMPData
 */
#define CRIA_CRFBmpConvertBMPToFBMPData_VALIDATION_CHECK(byteData, outFloatData, valueCount)\
if (!byteData || !outFloatData || valueCount == 0) \
{\
	memset(outFloatData, 0, sizeof(float) * valueCount);\
	return;\
}
/*
 * CRFBmpConvertFBMPToBMPData
 */
#define CRIA_CRFBmpConvertFBMPToBMPData_VALIDATION_CHECK(floatData, outByteData, valueCount)\
if (!floatData || !outByteData || valueCount == 0) \
{\
	memset(outByteData, 0, sizeof(byte) * valueCount);\
	return;\
}

/*
 * CRFBmpConvertToFPP
 */
#define CRIA_CRFBmpConvertToFPP_VALIDATION_CHECK(inFBmp, outFBmp)\
{\
	if (!inFBmp || !outFBmp ||\
		outFBmp->Width  != inFBmp->Width ||\
		outFBmp->Height != inFBmp->Height)\
	{\
		CR_FBMP_FILL_ZERO(outFBmp);\
		return;\
	}\
}
#define CRIA_CRFBmpConvertToFPP_IF_SAME_BPP(inFBmp, outFBmp) \
if (inFBmp->Fpp == outFBmp->Fpp)\
{\
	CR_FBMP_COPY_DATA(inFBmp, outFBmp);\
	return; /* done */\
}

/*
 * CRFBmpScale
 */
#define CRIA_CRFBmpScale_VALIDATION_CHECK(inFBmp, outFBmp, scale)\
{\
	uint outWidth = (uint)ceilf((float)inFBmp->Width * scale);\
	uint outHeight = (uint)ceilf((float)inFBmp->Width * scale); \
	if (!inFBmp || !outFBmp || \
		outFBmp->Width != outWidth || \
		outFBmp->Height != outHeight || \
		scale == 0) \
	{\
		CR_FBMP_FILL_ZERO(outFBmp);\
		return;\
	}\
}
#define CRIA_CRFBmpScale_IF_SCALE_1(inFBmp, outFBmp, scale)\
if (scale == 1.0f) {\
	CR_FBMP_COPY_DATA(inFBmp, outFBmp);\
	return;\
}

/*
 * CRFBmpToMatf
 */
#define CRIA_CRFBmpToMatf_VALIDATION_CHECK(inFBmp, outMat) \
if (!inFBmp || !outMat ||\
	CR_MATF_VALUE_COUNT(outMat) != CR_FBMP_VALUE_COUNT(inFBmp))\
{\
	CR_MATF_FILL_ZERO(outMat);\
	return;\
}

