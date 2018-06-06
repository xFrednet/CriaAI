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
 * CRBmpConvertToBPP
 */
#define CRIA_CRBMPCONVERTFROM_TO_BPP_VALIDATION_CHECK(inBmp, outBmp, inBpp, outBpp)\
{\
	if (!inBmp || !outBmp ||\
		inBmp->Width  != outBmp->Width  ||\
		inBmp->Height != outBmp->Height ||\
		inBmp->Bpp    != inBpp ||\
		outBmp->Bpp   != outBpp)\
	{\
		CR_BMP_FILL_ZERO(outBmp);\
		return;\
	}\
}
#define CRIA_CRBMPCONVERTTOBPP_IF_SAME_BPP(inBmp, outBmp) \
if (inBmp->Bpp == outBmp->Bpp)\
{\
	CR_BMP_COPY_DATA(inBmp, outBmp);\
	return; /* done */\
}
#define CRIA_CRBMPCONVERTTOBPP_VALIDATION_CHECK(inBmp, outBmp)\
{\
	if (!inBmp || !outBmp ||\
		outBmp->Width != inBmp->Width ||\
		outBmp->Height != inBmp->Height)\
	{\
		CR_BMP_FILL_ZERO(outBmp);\
		return;\
	}\
}

/*
 * CRBmpScale
 */
#define CRIA_CRBMPSCALE_VALIDATION_CHECK(inBmp, outBmp, scale)\
{\
	uint outWidth = (uint)ceilf((float)inBmp->Width * scale);\
	uint outHeight = (uint)ceilf((float)inBmp->Width * scale); \
	if (!inBmp || !outBmp || \
		outBmp->Width != outWidth || \
		outBmp->Height != outHeight || \
		scale == 0) \
	{\
		CR_BMP_FILL_ZERO(outBmp);\
		return;\
	}\
}
#define CRIA_CRBMPSCALE_IF_SCALE_1(inBmp, outBmp, scale)\
if (scale == 1.0f) {\
	CR_BMP_COPY_DATA(inBmp, outBmp);\
	return;\
}

/*
 * CRBmpToMatf
 */
#define CRIA_CRBMPTOMATF_VALIDATION_CHECK(inBmp, outMat) \
if (!inBmp || !outMat ||\
	CR_MATF_VALUE_COUNT(outMat) != CR_BMP_VALUE_COUNT(inBmp))\
{\
	CR_MATF_FILL_ZERO(outMat);\
	return;\
}

