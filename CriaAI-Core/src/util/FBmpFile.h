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

#define CR_FBMP_MAX_FPP                 4
#define CR_FBMP_DEFAULT_FPP             4

#define CR_FBMP_VALUE_COUNT(fbmp)                ((fbmp)->Width * (fbmp)->Height * (fbmp)->Fpp)
#define CR_FBMP_DATA_SIZE(fbmp)                  (sizeof(float) * CR_FBMP_VALUE_COUNT(fbmp))
#define CR_FBMP_PX_INDEX(x, y, fbmp)             (((x) + (y) * (fbmp)->Width) * (fbmp)->Fpp)
#define CR_FBMP_FILL_ZERO(fbmp)                  (memset((fbmp)->Data, 0, CR_FBMP_DATA_SIZE(fbmp)))
#define CR_FBMP_COPY_DATA(inFbmp, outFbmp)       (memcpy((outFbmp)->Data, (inFbmp)->Data, CR_FBMP_DATA_SIZE(inFbmp)))


namespace cria_ai
{
	typedef struct CR_FBMP_ {
		uint   Width;
		uint   Height;
		uint   Fpp;
		float* Data;          /* r, g, b, a*/
	} CR_FBMP;

	/*
	 * CRFBmpCreate
	 */
	CR_FBMP*        CRFBmpCreatePACO(uint width, uint height, uint fpp = CR_FBMP_DEFAULT_FPP);
	CR_FBMP*        CRFBmpCreateNormal(uint width, uint height, uint fpp = CR_FBMP_DEFAULT_FPP);
	inline CR_FBMP* CRFBmpCreate(uint width, uint height, uint fpp = CR_FBMP_DEFAULT_FPP)
	{
		return CRFBmpCreatePACO(width, height, fpp);
	}

	/*
	 * CRFBmpDelete
	 */
	void            CRFBmpDeletePACO(CR_FBMP* fbmp);
	void            CRFBmpDeleteNormal(CR_FBMP* fbmp);
	inline void     CRFBmpDelete(CR_FBMP* fbmp)
	{
		CRFBmpDeletePACO(fbmp);
	}

	/*
	 * CRFBmp utility
	 */
	CR_FBMP*        CRFBmpLoad(const String& file, crresult* result = nullptr);
	CR_FBMP*        CRFBmpCreateCopy(CR_FBMP const* srcBmp);
	crresult        CRFBmpSave(CR_FBMP const* srcBmp, const String& file);
}
