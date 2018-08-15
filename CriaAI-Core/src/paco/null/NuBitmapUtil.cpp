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
	
	void            CRFBmpConvertBMPToFBMPData(byte const* byteData, float* outFloatData, uint valueCount)
	{
		/*
		* Validation
		*/
		CRIA_CRFBmpConvertBMPToFBMPData_VALIDATION_CHECK(byteData, outFloatData, valueCount);

		/*
		* convert
		*/
		for (uint index = 0; index < valueCount; index++) {
			outFloatData[index] = (float)byteData[index] / 255.0f;
		}
	}
	void            CRFBmpConvertFBMPToBMPData(float const* floatData, byte* outByteData, uint valueCount)
	{
		/*
		* Validation
		*/
		CRIA_CRFBmpConvertFBMPToBMPData_VALIDATION_CHECK(floatData, outByteData, valueCount);

		/*
		* convert
		*/
		for (uint index = 0; index < valueCount; index++) {
			outByteData[index] = (byte)rintf(floatData[index] * 255.0f);
		}
	}

	void CRBmpConvertFrom3To1BPP(CR_FBMP const* inBmp, CR_FBMP* outBmp)
	{
		uint srcIndex = 0;
		uint dstIndex = 0;
		float sum = 0;
		for (uint pixel = 0; pixel < inBmp->Width * inBmp->Height; pixel++) 
		{
			sum  = inBmp->Data[srcIndex++];
			sum += inBmp->Data[srcIndex++];
			sum += inBmp->Data[srcIndex++];

			outBmp->Data[dstIndex++] = sum / 3.0f;
		}
	}
	void CRBmpConvertFrom4To1BPP(CR_FBMP const* inBmp, CR_FBMP* outBmp)
	{
		float sum;
		uint srcIndex = 0;
		uint dstIndex = 0;
		for (uint index = 0; index < inBmp->Width * inBmp->Height; index++) 
		{
			sum  = inBmp->Data[srcIndex++]; /* Red   */
			sum += inBmp->Data[srcIndex++]; /* Green */
			sum += inBmp->Data[srcIndex++]; /* Blue  */
			sum *= inBmp->Data[srcIndex++]; /* Alpha */

			outBmp->Data[dstIndex++] = sum / 3.0f;
		}
	}
	void CRBmpConvertFrom1To3BPP(CR_FBMP const* inBmp, CR_FBMP* outBmp)
	{
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
	void CRBmpConvertFrom4To3BPP(CR_FBMP const* inBmp, CR_FBMP* outBmp)
	{
		uint srcIndex = 0;
		uint dstIndex = 0;
		for (uint pixel = 0; pixel < inBmp->Width * inBmp->Height; pixel++) 
		{
			float alpha = inBmp->Data[srcIndex + 4];

			outBmp->Data[dstIndex++] = inBmp->Data[srcIndex++] * alpha;
			outBmp->Data[dstIndex++] = inBmp->Data[srcIndex++] * alpha;
			outBmp->Data[dstIndex++] = inBmp->Data[srcIndex++] * alpha;
			srcIndex++;
		}
	}
	void CRBmpConvertFrom1To4BPP(CR_FBMP const* inBmp, CR_FBMP* outBmp)
	{
		uint srcIndex = 0;
		uint dstIndex = 0;
		for (uint pixel = 0; pixel < inBmp->Width * inBmp->Height; pixel++) 
		{
			outBmp->Data[dstIndex++] = inBmp->Data[srcIndex];
			outBmp->Data[dstIndex++] = inBmp->Data[srcIndex];
			outBmp->Data[dstIndex++] = inBmp->Data[srcIndex];
			outBmp->Data[dstIndex++] = 1.0f; //100% alpha
			srcIndex++;
		}
	}
	void CRBmpConvertFrom3To4BPP(CR_FBMP const* inBmp, CR_FBMP* outBmp)
	{
		uint srcIndex = 0;
		uint dstIndex = 0;
		for (uint pixel = 0; pixel < inBmp->Width * inBmp->Height; pixel++) {
			outBmp->Data[dstIndex++] = inBmp->Data[srcIndex++];
			outBmp->Data[dstIndex++] = inBmp->Data[srcIndex++];
			outBmp->Data[dstIndex++] = inBmp->Data[srcIndex++];
			outBmp->Data[dstIndex++] = 1.0f; //100% alpha
			srcIndex++;
		}
	}
	void    CRFBmpConvertToFPP(CR_FBMP const* inBmp, CR_FBMP* outBmp)
	{
		/*
		* Validation
		*/
		CRIA_CRFBmpConvertToFPP_VALIDATION_CHECK(inBmp, outBmp);

		/*
		* Select converter
		*/
		CRIA_CRFBmpConvertToFPP_IF_SAME_BPP(inBmp, outBmp);

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
		byte coversion = (inBmp->Fpp << 4) | (outBmp->Fpp);
		switch (coversion) {
			case 0x13: CRBmpConvertFrom1To3BPP(inBmp, outBmp); break;
			case 0x14: CRBmpConvertFrom1To4BPP(inBmp, outBmp); break;

			case 0x31: CRBmpConvertFrom3To1BPP(inBmp, outBmp); break;
			case 0x34: CRBmpConvertFrom3To4BPP(inBmp, outBmp); break;

			case 0x41: CRBmpConvertFrom4To1BPP(inBmp, outBmp); break;
			case 0x43: CRBmpConvertFrom4To3BPP(inBmp, outBmp); break;

			default:
				CRIA_AUTO_ASSERT(false, "The conversion failed: inBmp->Fpp: %u, outBmp->Fpp: %u ", inBmp->Fpp, outBmp->Fpp);
				CR_FBMP_FILL_ZERO(outBmp);;
				break;
		}
	}

	void CRFBmpScale(CR_FBMP const* inBmp, CR_FBMP* outBmp, float scale)
	{
		/*
		 * Validation
		 */
		CRIA_CRFBmpScale_VALIDATION_CHECK(inBmp, outBmp, scale);

		/*
		 * Scaling
		 */
		CRIA_CRFBmpScale_IF_SCALE_1(inBmp, outBmp, scale);

		float srcScale = 1.0f / scale;

		float srcX = 0.0f;
		float srcY = 0.0f;
		for (uint y = 0; y < outBmp->Height; y++)
		{
			srcX = 0.0f;
			for (uint x = 0; x < outBmp->Width; x++)
			{
				void* src = &inBmp->Data[CR_FBMP_PX_INDEX((uint)floor(srcX), (uint)floor(srcY), inBmp)];
				void* dst = &outBmp->Data[CR_FBMP_PX_INDEX(x, y, outBmp)];

				memcpy(dst, src, sizeof(float) * inBmp->Fpp);

				srcX += srcScale;
			}
			srcY += srcScale;
		}
	}

	void CRFBmpToMatf(CR_FBMP const* inBmp, CR_MATF* outMat)
	{
		/*
		* Validation
		*/
		CRIA_CRFBmpToMatf_VALIDATION_CHECK(inBmp, outMat);

		/*
		* Convert
		*/
		for (uint index = 0; index < outMat->Cols * outMat->Rows; index++) {
			outMat->Data[index] = (float)inBmp->Data[index] / 255.0f;
		}
	}

	void CRFBmpPool(CR_FBMP const* inBmp, CR_FBMP* outBmp, uint poolSize)
	{
		/*
		* Validation
		*/
		CRIA_CRFBmpPool_VALIDATION_CHECK(inBmp, outBmp, poolSize);

		/*
		* Pooling
		*/
		CRIA_CRFBmpPool_IS_POOLSIZE_1(inBmp, outBmp, poolSize);

		uint poolCount = outBmp->Width * outBmp->Height;
		for (uint poolNo = 0; poolNo < poolCount; poolNo++) {
			uint poolX = poolNo % outBmp->Width;
			uint poolY = poolNo / outBmp->Height;

			/*
			* Save first pool Value
			*/
			uint xa = poolX * poolSize;
			uint ya = poolY * poolSize;
			float maxValues[4];
			for (uint channel = 0; channel < inBmp->Fpp; channel++)
				maxValues[channel] = inBmp->Data[CR_FBMP_PX_INDEX(xa, ya, inBmp) + channel];

			/*
			* scanning the pool
			*/
			for (uint y = 0; y < poolSize && ya < inBmp->Height; y++, ya++) {
				xa = poolX * poolSize;

				for (uint x = 0; x < poolSize && xa < inBmp->Width; x++, xa++) {

					for (uint channel = 0; channel < inBmp->Fpp; channel++) {
						uint index = CR_FBMP_PX_INDEX(xa, ya, inBmp);

						if (maxValues[channel] < inBmp->Data[index + channel])
							maxValues[channel] = inBmp->Data[index + channel];
					}

				}
			}

			/*
			* saving the result
			*/
			for (uint channel = 0; channel < inBmp->Fpp; channel++)
				outBmp->Data[CR_FBMP_PX_INDEX(poolX, poolY, outBmp) + channel] = maxValues[channel];
		}
	}

	void CRFBmpNormalize(CR_FBMP const* inBmp, CR_FBMP const* outBmp)
	{
		/*
		* Validation
		*/
		CRIA_CRFBmpNormalize_IS_VALID(inBmp, outBmp);

		/*
		* Normalizing
		*/
		CRIA_CRFBmpNormalize_IF_1_FPP(inBmp, outBmp);

		/*
		* Note only the RGB channels are normalized.
		* The alpha value will just be copied.
		*
		* Only 3 and 4 Fpp Bitmaps are passed into this function(If not hot fix it!)
		*/
		for (uint index = 0; index < inBmp->Width * inBmp->Height; index++) {
			uint pxIndex = index = inBmp->Fpp;

			float total = 0.0f;
			for (uint fNo = 0; fNo < 3; fNo++) {
				total += inBmp->Data[pxIndex + fNo];
			}

			/*
			* Copy the alpha channel because it not normalized.
			* Ff the bitmap has only 3 floats per pixel it will be overridden
			* in the next step anyways
			*/
			outBmp->Data[pxIndex + inBmp->Fpp - 1] = inBmp->Data[pxIndex + inBmp->Fpp - 1];

			/*
			* Set data to 0 if total is 0
			* Normalize if total is not 0
			*/
			if (total == 0.0f) {
				for (uint fNo = 0; fNo < inBmp->Fpp; fNo++) {
					outBmp->Data[pxIndex + fNo] = 0;
				}
			}
			else {
				for (uint fNo = 0; fNo < 3; fNo++) {
					outBmp->Data[pxIndex + fNo] = inBmp->Data[pxIndex + fNo] / total;
				}
			}

		}
	}
}}

#endif CRIA_PACO_NULL
