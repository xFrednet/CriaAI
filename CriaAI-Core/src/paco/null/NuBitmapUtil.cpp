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
#include "../BitmapUtil.h"
#include "../BitmapUtilMacros.h"

#ifdef CRIA_PACO_NULL

namespace cria_ai { namespace paco {
	
	void CRBmpConvertFrom3To1BPP(CR_BMP const* inBmp, CR_BMP* outBmp)
	{
		CRIA_CRBMPCONVERTFROM_TO_BPP_VALIDATION_CHECK(inBmp, outBmp, 3, 1);
		uint srcIndex = 0;
		uint dstIndex = 0;
		uint sum = 0;
		for (uint pixel = 0; pixel < inBmp->Width * inBmp->Height; pixel++) 
		{
			sum  = inBmp->Data[srcIndex++];
			sum += inBmp->Data[srcIndex++];
			sum += inBmp->Data[srcIndex++];

			outBmp->Data[dstIndex++] = (byte)roundf(sum / 3.0f);
		}
	}
	void CRBmpConvertFrom4To1BPP(CR_BMP const* inBmp, CR_BMP* outBmp)
	{
		CRIA_CRBMPCONVERTFROM_TO_BPP_VALIDATION_CHECK(inBmp, outBmp, 4, 1);

		float sum;
		uint srcIndex = 0;
		uint dstIndex = 0;
		for (uint index = 0; index < inBmp->Width * inBmp->Height; index++) 
		{
			sum  = inBmp->Data[srcIndex++]; /* Red   */
			sum += inBmp->Data[srcIndex++]; /* Green */
			sum += inBmp->Data[srcIndex++]; /* Blue  */
			sum *= inBmp->Data[srcIndex++]; /* Alpha */

			outBmp->Data[dstIndex++] = (byte)roundf(sum / 3.0f);
		}
	}
	void CRBmpConvertFrom1To3BPP(CR_BMP const* inBmp, CR_BMP* outBmp)
	{
		CRIA_CRBMPCONVERTFROM_TO_BPP_VALIDATION_CHECK(inBmp, outBmp, 1, 3);
		
		uint srcIndex = 0;
		uint dstIndex = 0;
		for (uint pixel = 0; pixel < inBmp->Width * inBmp->Height; pixel++)
		{
			outBmp->Data[dstIndex++] = inBmp->Data[srcIndex];
			outBmp->Data[dstIndex++] = inBmp->Data[srcIndex];
			outBmp->Data[dstIndex++] = inBmp->Data[srcIndex];
			srcIndex++;
		}
	}
	void CRBmpConvertFrom4To3BPP(CR_BMP const* inBmp, CR_BMP* outBmp)
	{
		CRIA_CRBMPCONVERTFROM_TO_BPP_VALIDATION_CHECK(inBmp, outBmp, 4, 3);

		uint srcIndex = 0;
		uint dstIndex = 0;
		for (uint pixel = 0; pixel < inBmp->Width * inBmp->Height; pixel++) 
		{
			float alpha = inBmp->Data[srcIndex + 4];

			outBmp->Data[dstIndex++] = (byte)floorf(inBmp->Data[srcIndex++] * alpha);
			outBmp->Data[dstIndex++] = (byte)floorf(inBmp->Data[srcIndex++] * alpha);
			outBmp->Data[dstIndex++] = (byte)floorf(inBmp->Data[srcIndex++] * alpha);
			srcIndex++;
		}
	}
	void CRBmpConvertFrom1To4BPP(CR_BMP const* inBmp, CR_BMP* outBmp)
	{
		CRIA_CRBMPCONVERTFROM_TO_BPP_VALIDATION_CHECK(inBmp, outBmp, 1, 4);

		uint srcIndex = 0;
		uint dstIndex = 0;
		for (uint pixel = 0; pixel < inBmp->Width * inBmp->Height; pixel++) 
		{
			outBmp->Data[dstIndex++] = inBmp->Data[srcIndex];
			outBmp->Data[dstIndex++] = inBmp->Data[srcIndex];
			outBmp->Data[dstIndex++] = inBmp->Data[srcIndex];
			outBmp->Data[dstIndex++] = 0xff; //100% alpha
			srcIndex++;
		}
	}
	void CRBmpConvertFrom3To4BPP(CR_BMP const* inBmp, CR_BMP* outBmp)
	{
		CRIA_CRBMPCONVERTFROM_TO_BPP_VALIDATION_CHECK(inBmp, outBmp, 3, 4);

		uint srcIndex = 0;
		uint dstIndex = 0;
		for (uint pixel = 0; pixel < inBmp->Width * inBmp->Height; pixel++) {
			outBmp->Data[dstIndex++] = inBmp->Data[srcIndex++];
			outBmp->Data[dstIndex++] = inBmp->Data[srcIndex++];
			outBmp->Data[dstIndex++] = inBmp->Data[srcIndex++];
			outBmp->Data[dstIndex++] = 0xff; //100% alpha
			srcIndex++;
		}
	}
	void    CRBmpConvertToBPP(CR_BMP const* inBmp, CR_BMP* outBmp)
	{
		/*
		* Validation
		*/
		CRIA_CRBMPCONVERTTOBPP_VALIDATION_CHECK(inBmp, outBmp);

		/*
		* Select converter
		*/
		CRIA_CRBMPCONVERTTOBPP_IF_SAME_BPP(inBmp, outBmp);

		/*
		* The first 4 bits indicates the input bitmap bpp and
		* the last 4 bits indicates the output bitmap bpp.
		*
		* 1 => 0x01
		* 3 => 0x03
		* 4 => 0x04
		*
		* This is trashy yet genius                       ~xFrednet 06.06.2018
		* Reworked, even more genius and less trashy!     ~xFrednet 06.06.2018
		*/
		byte coversion = (inBmp->Bpp << 4) | (outBmp->Bpp);
		switch (coversion) {
			case 0x13: CRBmpConvertFrom1To3BPP(inBmp, outBmp); break;
			case 0x14: CRBmpConvertFrom1To4BPP(inBmp, outBmp); break;

			case 0x31: CRBmpConvertFrom3To1BPP(inBmp, outBmp); break;
			case 0x34: CRBmpConvertFrom3To4BPP(inBmp, outBmp); break;

			case 0x41: CRBmpConvertFrom4To1BPP(inBmp, outBmp); break;
			case 0x43: CRBmpConvertFrom4To3BPP(inBmp, outBmp); break;

			default:
				CRIA_AUTO_ASSERT(false, "The conversion failed: inBmp->Bpp: %u, outBmp->Bpp: %u ", inBmp->Bpp, outBmp->Bpp);
				CR_BMP_FILL_ZERO(outBmp);;
				break;
		}
	}

	void CRBmpScale(CR_BMP const* inBmp, CR_BMP* outBmp, float scale)
	{
		/*
		 * Validation
		 */
		CRIA_CRBMPSCALE_VALIDATION_CHECK(inBmp, outBmp, scale);

		CRIA_CRBMPSCALE_IF_SCALE_1(inBmp, outBmp, scale);

		float srcScale = 1 / scale;

		float srcX = 0.0f;
		float srcY = 0.0f;
		for (uint y = 0; y < outBmp->Height; y++)
		{
			srcX = 0.0f;
			for (uint x = 0; x < outBmp->Width; x++)
			{
				void* src = &inBmp->Data[CR_BMP_PX_INDEX((uint)floor(srcX), (uint)floor(srcY), inBmp)];
				void* dst = &outBmp->Data[CR_BMP_PX_INDEX(x, y, outBmp)];

				memcpy(dst, src, inBmp->Bpp);

				srcX += srcScale;
			}
			srcY += srcScale;
		}
	}

	void CRBmpToMatf(CR_BMP const* inBmp, CRMatrixf* outMat)
	{
		/*
		* Validation
		*/
		CRIA_CRBMPTOMATF_VALIDATION_CHECK(inBmp, outMat);

		/*
		* Convert
		*/
		for (uint index = 0; index < outMat->Cols * outMat->Rows; index++) {
			outMat->Data[index] = (float)inBmp->Data[index] / 255.0f;
		}
	}

}}

#endif CRIA_PACO_NULL
