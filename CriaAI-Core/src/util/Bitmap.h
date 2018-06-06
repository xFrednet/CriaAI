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

#include "../Types.hpp"
#include "CRResult.h"

#define CR_BMP_MAX_BPP                 4
#define CR_BMP_DEFAULT_BPP             4

#define CR_BMP_PX_INDEX(x, y, bmp)               (((x) + (y) * (bmp)->Width) * (bmp)->Bpp)
#define CR_BMP_DATA_SIZE(bmp)                    ((bmp)->Width * (bmp)->Height * (bmp)->Bpp)
#define CR_BMP_FILL_ZERO(bmp)                    (memset((bmp)->Data, 0, CR_BMP_DATA_SIZE(bmp)))
#define CR_BMP_COPY_DATA(inBmp, outBmp)          (memcpy((outBmp)->Data, (inBmp)->Data, CR_BMP_DATA_SIZE(inBmp)))
#define CR_BMP_VALUE_COUNT(bmp)                  ((bmp)->Width * (bmp)->Height * (bmp)->Bpp)


namespace cria_ai
{
	typedef struct CR_BMP_ {
		uint  Width;
		uint  Height;
		uint  Bpp;
		byte* Data;          /* r, g, b, a*/
	} CR_BMP;

	/*
	 * CRCreateBmp
	 */
	CR_BMP* CRCreateBmpPACO(uint width, uint height, uint bpp = CR_BMP_DEFAULT_BPP);
	CR_BMP* CRCreateBmpNormal(uint width, uint height, uint bpp = CR_BMP_DEFAULT_BPP);
	inline CR_BMP* CRCreateBmp(uint width, uint height, uint bpp = CR_BMP_DEFAULT_BPP)
	{
		return CRCreateBmpPACO(width, height, bpp);
	}

	/*
	 * CRDeleteBmp
	 */
	void CRDeleteBmpPACO(CR_BMP* bmp);
	void CRDeleteBmpNormal(CR_BMP* bmp);
	inline void CRDeleteBmp(CR_BMP* bmp)
	{
		CRDeleteBmpPACO(bmp);
	}

	CR_BMP* CRLoadBmp(const String& file, crresult* result = nullptr);
	CR_BMP* CRCreateCopyBmp(CR_BMP const* srcBmp);
	crresult CRSaveBmp(CR_BMP const* bmp, const String& file);
}
