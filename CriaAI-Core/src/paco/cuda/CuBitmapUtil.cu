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

#ifdef CRIA_PACO_CUDA 

#include "CuContext.cuh"

#define CR_CUDA_AF_BLOCK_COUNT         1
#define CR_CUDA_AF_THREAD_COUNT        256

/*
 * This is just a small macro setup to reduce the shown errors and to make it
 * easier to make it easier and cleaner to call cuda functions
 * 
 */
#ifdef __NVCC__
#	define CRIA_CUDA_CALL                 <<<CR_CUDA_AF_BLOCK_COUNT, CR_CUDA_AF_THREAD_COUNT>>>
#else
#	define CRIA_CUDA_CALL
#	error CRIA_CUDA_CALL should be defined differenly during compilation with the cuda compiler
#endif

namespace cria_ai { namespace paco {
	
	/*
	* CRBmpConvertToBPP
	*/
	__global__ void CRCuBmpConvertFrom3To1BPP(CR_BMP const* inBmp, CR_BMP* outBmp)
	{
		CRIA_CRBMPCONVERTFROM_TO_BPP_VALIDATION_CHECK(inBmp, outBmp, 3, 1);
		
		int startIndex = blockIdx.x * blockDim.x + threadIdx.x;
		int stride = blockDim.x * gridDim.x;

		for (uint pixel = startIndex; pixel < inBmp->Width * inBmp->Height; pixel += stride) 
		{
			uint srcIndex = 0;
			uint dstIndex = 0;
			
			float sum = 0;
			sum  = inBmp->Data[srcIndex++];
			sum += inBmp->Data[srcIndex++];
			sum += inBmp->Data[srcIndex++];

			outBmp->Data[dstIndex++] = (byte)rintf(sum / 3.0f);
		}
	}
	__global__ void CRCuBmpConvertFrom4To1BPP(CR_BMP const* inBmp, CR_BMP* outBmp)
	{
		CRIA_CRBMPCONVERTFROM_TO_BPP_VALIDATION_CHECK(inBmp, outBmp, 4, 1);

		int startIndex = blockIdx.x * blockDim.x + threadIdx.x;
		int stride = blockDim.x * gridDim.x;

		for (uint pixel = startIndex; pixel < inBmp->Width * inBmp->Height; pixel += stride)
		{
			uint srcIndex = pixel * 4;

			float sum;
			sum  = inBmp->Data[srcIndex++]; /* Red   */
			sum += inBmp->Data[srcIndex++]; /* Green */
			sum += inBmp->Data[srcIndex++]; /* Blue  */
			sum *= inBmp->Data[srcIndex  ]; /* Alpha */
			sum /= 3;

			outBmp->Data[pixel] = (byte)rintf(sum / 3.0f);
		}
	}
	__global__ void CRCuBmpConvertFrom1To3BPP(CR_BMP const* inBmp, CR_BMP* outBmp)
	{
		CRIA_CRBMPCONVERTFROM_TO_BPP_VALIDATION_CHECK(inBmp, outBmp, 1, 3);
		
		int startIndex = blockIdx.x * blockDim.x + threadIdx.x;
		int stride = blockDim.x * gridDim.x;

		for (uint pixel = startIndex; pixel < inBmp->Width * inBmp->Height; pixel += stride) 
		{
			uint dstIndex = pixel * 3;

			outBmp->Data[dstIndex++] = inBmp->Data[pixel];
			outBmp->Data[dstIndex++] = inBmp->Data[pixel];
			outBmp->Data[dstIndex  ] = inBmp->Data[pixel];
		}
	}
	__global__ void CRCuBmpConvertFrom4To3BPP(CR_BMP const* inBmp, CR_BMP* outBmp)
	{
		CRIA_CRBMPCONVERTFROM_TO_BPP_VALIDATION_CHECK(inBmp, outBmp, 4, 3);

		int startIndex = blockIdx.x * blockDim.x + threadIdx.x;
		int stride     = blockDim.x * gridDim.x;

		for (uint pixel = startIndex; pixel < inBmp->Width * inBmp->Height; pixel += stride)
		{
			uint srcIndex = pixel * 4;
			uint dstIndex = pixel * 3;

			float alpha = inBmp->Data[srcIndex + 4];

			outBmp->Data[dstIndex++] = (byte)floorf(inBmp->Data[srcIndex++] * alpha);
			outBmp->Data[dstIndex++] = (byte)floorf(inBmp->Data[srcIndex++] * alpha);
			outBmp->Data[dstIndex  ] = (byte)floorf(inBmp->Data[srcIndex  ] * alpha);
		}
	}
	__global__ void CRCuBmpConvertFrom1To4BPP(CR_BMP const* inBmp, CR_BMP* outBmp)
	{
		CRIA_CRBMPCONVERTFROM_TO_BPP_VALIDATION_CHECK(inBmp, outBmp, 1, 4);

		int startIndex = blockIdx.x * blockDim.x + threadIdx.x;
		int stride = blockDim.x * gridDim.x;

		for (uint pixel = startIndex; pixel < inBmp->Width * inBmp->Height; pixel += stride) 
		{
			uint dstIndex = pixel * 4;

			outBmp->Data[dstIndex++] = inBmp->Data[pixel];
			outBmp->Data[dstIndex++] = inBmp->Data[pixel];
			outBmp->Data[dstIndex++] = inBmp->Data[pixel];
			outBmp->Data[dstIndex  ] = 0xff; //100% alpha
		}
	}
	__global__ void CRCuBmpConvertFrom3To4BPP(CR_BMP const* inBmp, CR_BMP* outBmp)
	{
		CRIA_CRBMPCONVERTFROM_TO_BPP_VALIDATION_CHECK(inBmp, outBmp, 3, 4);

		int startIndex = blockIdx.x * blockDim.x + threadIdx.x;
		int stride = blockDim.x * gridDim.x;

		for (uint pixel = startIndex; pixel < inBmp->Width * inBmp->Height; pixel += stride)
		{
			uint srcIndex = pixel * 3;
			uint dstIndex = pixel * 4;

			outBmp->Data[dstIndex++] = inBmp->Data[srcIndex++];
			outBmp->Data[dstIndex++] = inBmp->Data[srcIndex++];
			outBmp->Data[dstIndex++] = inBmp->Data[srcIndex++];
			outBmp->Data[dstIndex  ] = 0xff; //100% alpha
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
			case 0x13: 
				CRCuBmpConvertFrom1To3BPP CRIA_CUDA_CALL (inBmp, outBmp);
				break;
			case 0x14: 
				CRCuBmpConvertFrom1To4BPP CRIA_CUDA_CALL (inBmp, outBmp);
				break;
			
			case 0x31:
				CRCuBmpConvertFrom3To1BPP CRIA_CUDA_CALL (inBmp, outBmp);
				break;
			case 0x34: 
				CRCuBmpConvertFrom3To4BPP CRIA_CUDA_CALL (inBmp, outBmp);
				break;
			
			case 0x41: 
				CRCuBmpConvertFrom4To1BPP CRIA_CUDA_CALL (inBmp, outBmp);
				break;
			case 0x43: 
				CRCuBmpConvertFrom4To3BPP CRIA_CUDA_CALL (inBmp, outBmp);
				break;

			default:
				CRIA_AUTO_ASSERT(false, "The conversion failed: inBmp->Bpp: %u, outBmp->Bpp: %u ", inBmp->Bpp, outBmp->Bpp);
				CR_BMP_FILL_ZERO(outBmp);;
				return;
		}
		cudaDeviceSynchronize();
	}

	/*
	* CRBmpScale
	*/
	__global__ void CRCuBmpScale(CR_BMP const* inBmp, CR_BMP* outBmp, float srcScale)
	{
		int startIndex = blockIdx.x * blockDim.x + threadIdx.x;
		int stride = blockDim.x * gridDim.x;

		for (uint pixel = startIndex; pixel < outBmp->Width * outBmp->Height; pixel += stride)
		{
			uint dstX = pixel % outBmp->Width;
			uint dstY = pixel / outBmp->Width;
			uint dstIndex = CR_BMP_PX_INDEX(dstX, dstY, outBmp);
			uint srcIndex = CR_BMP_PX_INDEX((uint)floorf(dstX * srcScale), (uint)floorf(dstY * srcScale), outBmp);

			for (uint byteNo = 0; byteNo < outBmp->Bpp; byteNo++) 
			{
				outBmp->Data[dstIndex + byteNo] = inBmp->Data[srcIndex + byteNo];
			}
		}

	}
	void CRBmpScale(CR_BMP const* inBmp, CR_BMP* outBmp, float scale)
	{
		/*
		 * Validation
		 */
		CRIA_CRBMPSCALE_VALIDATION_CHECK(inBmp, outBmp, scale);

		/*
		 * Scaling
		 */
		CRIA_CRBMPSCALE_IF_SCALE_1(inBmp, outBmp, scale);

		CRCuBmpScale CRIA_CUDA_CALL (inBmp, outBmp, 1.0f / scale);
		cudaDeviceSynchronize();
	}

	/*
	 * CRBmpToMatf
	 */
	__global__ void CRCuBmpToMatf(CR_BMP const* inBmp, CRMatrixf* outMat)
	{
		int startIndex = blockIdx.x * blockDim.x + threadIdx.x;
		int stride = blockDim.x * gridDim.x;

		for (uint index = startIndex; index < outMat->Cols * outMat->Rows; index += stride)
		{
			outMat->Data[index] = (float)inBmp->Data[index] / 255.0f;
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
		CRCuBmpToMatf CRIA_CUDA_CALL (inBmp, outMat);
		cudaDeviceSynchronize();
	}
}}

#endif //CRIA_PACO_CUDA
