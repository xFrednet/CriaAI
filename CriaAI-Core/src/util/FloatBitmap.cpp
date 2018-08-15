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
#include "FloatBitmap.h"

#include "../paco/PaCoContext.h"
#include "../paco/BitmapUtil.h"

#include "../../Dependencies/BmpRenderer/Dependencies/libbmpread/bmpread.h"
#include "../os/FileSystem.h"
#include "../../Dependencies/BmpRenderer/src/RendererBitmap.hpp"

namespace cria_ai {
	/*
	* CRFBmpCreate
	*/
#define CRIA_FBMP_IS_CREATE_INPUT_VALID(width, height, fpp) \
	(width != 0 && height != 0 && fpp != 0 && fpp <= CR_FBMP_MAX_FPP)
#define CRIA_FBMP_FILL_MEMBERS(fbmp, width, height, fpp) \
fbmp->Width = width;\
fbmp->Height = height;\
fbmp->Fpp = fpp;\
fbmp->Data = (float*)((uintptr_t)bmp + sizeof(CR_FBMP));\
memset(bmp->Data, 0, CR_FBMP_DATA_SIZE(bmp));

	CR_FBMP* CRFBmpCreatePACO(uint width, uint height, uint fpp)
	{
		static std::mutex hello;
		std::lock_guard<std::mutex> helloFromTheOtherSide(hello);

		/*
		 * Validation
		 */
		CRIA_AUTO_ASSERT(CRIA_FBMP_IS_CREATE_INPUT_VALID(width, height, fpp), "CRFBmpCreatePACO invalid input: width %u, height %u, fpp %i", width, height, fpp);
		if (!CRIA_FBMP_IS_CREATE_INPUT_VALID(width, height, fpp))
			return nullptr;

		/*
		 * Creation
		 */
		size_t dataSize = sizeof(float) * width * height * fpp;
		CR_FBMP* bmp = (CR_FBMP*)paco::CRPaCoMalloc(sizeof(CR_FBMP) + dataSize);
		CRIA_AUTO_ASSERT(bmp, "CRFBmpCreatePACO paco::CRPaCoMalloc failed: width %u, height %u, fpp %i", width, height, fpp);
		if (!bmp)
			return nullptr;

		/*
		 * Fill struct members
		 */
		CRIA_FBMP_FILL_MEMBERS(bmp, width, height, fpp);
		
		return bmp;
	}
	CR_FBMP* CRFBmpCreateNormal(uint width, uint height, uint fpp)
	{
		/*
		* Validation
		*/
		CRIA_AUTO_ASSERT(CRIA_FBMP_IS_CREATE_INPUT_VALID(width, height, fpp), "CRFBmpCreateNormal invalid input: width %u, height %u, fpp %i", width, height, fpp);
		if (!CRIA_FBMP_IS_CREATE_INPUT_VALID(width, height, fpp))
			return nullptr;

		/*
		* Creation
		*/
		size_t dataSize = sizeof(float) * width * height * fpp;
		CR_FBMP* bmp = (CR_FBMP*)malloc(sizeof(CR_FBMP) + dataSize);
		CRIA_AUTO_ASSERT(bmp, "CRFBmpCreateNormal paco::CRPaCoMalloc failed: width %u, height %u, fpp %i", width, height, fpp);
		if (!bmp)
			return nullptr;

		/*
		* Fill struct members
		*/
		CRIA_FBMP_FILL_MEMBERS(bmp, width, height, fpp);

		return bmp;
	}

	/*
	* CRFBmpDelete
	*/
	void     CRFBmpDeletePACO(CR_FBMP* fbmp)
	{
		if (fbmp)
			paco::CRPaCoFree(fbmp);
	}
	void     CRFBmpDeleteNormal(CR_FBMP* fbmp)
	{
		if (fbmp)
			free(fbmp);
	}

	CR_FBMP* CRFBmpLoad(const String& file, crresult* result)
	{
		/*
		* Validation
		*/
		CRIA_AUTO_ASSERT(!file.empty(), "LoadFloatBmp the source file is undefined");
		if (file.empty())
			return nullptr;

		/*
		 * Read file
		 */
		bmpread_t bmpReadIn;
		if (!bmpread(file.c_str(), BMPREAD_TOP_DOWN | BMPREAD_ANY_SIZE | BMPREAD_LOAD_ALPHA, &bmpReadIn)) {
			return nullptr;
		}

		/*
		* new Bitmap
		*/
		CR_FBMP* bmp = CRFBmpCreate(bmpReadIn.width, bmpReadIn.height, 4);
		byte* inData = (byte*)paco::CRPaCoMalloc(CR_FBMP_VALUE_COUNT(bmp));
		memcpy(inData, bmpReadIn.rgb_data, CR_FBMP_VALUE_COUNT(bmp));
		paco::CRFBmpConvertBMPToFBMPData(inData, bmp->Data, CR_FBMP_VALUE_COUNT(bmp));
		paco::CRPaCoFree(inData);

		//return
		return bmp;
	}
	CR_FBMP* CRFBmpCreateCopy(CR_FBMP const* srcBmp)
	{
		/*
		 * Validation
		 */
		CRIA_AUTO_ASSERT(srcBmp, "CRCreateCopyBmp: The input bitmap is a null pointer.");
		if (!srcBmp)
			return nullptr;

		/*
		 * new Bitmap
		 */
		CR_FBMP* dstBmp = CRFBmpCreate(srcBmp->Width, srcBmp->Height, srcBmp->Fpp);
		memcpy(dstBmp->Data, srcBmp->Data, CR_FBMP_DATA_SIZE(srcBmp));

		//return
		return dstBmp;
	}
	crresult CRFBmpSave(CR_FBMP const* srcBmp, const String& file)
	{
		/*
		* validation
		*/
		CRIA_AUTO_ASSERT(!file.empty(), "CRSaveBmp: The file name is invalid. The bitmap address is: %p", (srcBmp) ? srcBmp : nullptr);
		CRIA_AUTO_ASSERT(srcBmp, "CRSaveBmp: The bitmap is invalid. The file Name is: %s", file.c_str());
		if (!srcBmp || file.empty())
			return CRRES_ERR_INVALUD_ARGUMENTS; /* you broke it */

		 /*
		 * making sure the directory exists.
		 */
		if (!CRCreateContainingDir(file)) 
		{
			CRIA_AUTO_ASSERT(false, "CRSaveBmp: The creation of the containing directory failed, file: \"%s\"", file.c_str());
			return CRRES_ERR_OS_FAILED_TO_CREATE_DIR;
		}

		/*
		* The conversion
		*
		* Kids cover your eyes this should not be done.
		* 
		* I copied this from the FloatBitmap.cu but I just love this command :D
		*/
		crresult result = CRRES_OK;
		bmp_renderer::Bitmap* bmpRendererBmp = nullptr;
		byte* intData = nullptr;
		do {
			/*
			 * Paco int data
			 */
			size_t dataSize = srcBmp->Width * srcBmp->Height * 4;
			intData = (byte*)paco::CRPaCoMalloc(dataSize);
			if (!intData)
			{
				CRIA_AUTO_ASSERT(intData, "CRSaveBmp: paco::CRPaCoMalloc failed.");
				result = CRRES_ERROR;
				break; 
			}

			/*
			 * Conversion to 4FPP
			 */
			if (srcBmp->Fpp == 4) {
				paco::CRFBmpConvertFBMPToBMPData(srcBmp->Data, intData, CR_FBMP_VALUE_COUNT(srcBmp));
			}
			else 
			{
				CR_FBMP* srcBmp4fpp = paco::CRFBmpConvertToFPP(srcBmp, 4);
				if (!srcBmp4fpp) 
				{
					CRIA_AUTO_ASSERT(srcBmp4fpp, "CRSaveBmp: paco::CRFBmpConvertToFPP failed.");
					result = CRRES_ERROR;
					break;
				}
				paco::CRFBmpConvertFBMPToBMPData(srcBmp4fpp->Data, intData, CR_FBMP_VALUE_COUNT(srcBmp4fpp));
				CRFBmpDelete(srcBmp4fpp);
			}

			/*
			 * Create bmpRenderer bitmap
			 */
			bmpRendererBmp = bmp_renderer::CreateBmp(srcBmp->Width, srcBmp->Height);
			if (!bmpRendererBmp)
			{
				CRIA_AUTO_ASSERT(bmpRendererBmp, "CRSaveBmp: bmp_renderer::CreateBmp failed.");
				result = CRRES_ERROR;
				break;
			}
			uint srcIndex = 0;
			for (uint dstIndex = 0; dstIndex < srcBmp->Width * srcBmp->Height; dstIndex++)
			{
				bmpRendererBmp->Data[dstIndex].R = intData[srcIndex++];
				bmpRendererBmp->Data[dstIndex].G = intData[srcIndex++];
				bmpRendererBmp->Data[dstIndex].B = intData[srcIndex++];
				bmpRendererBmp->Data[dstIndex].A = intData[srcIndex++];
			}
			
			/*
			* Saving
			*/
			if (!bmp_renderer::SaveBitmap(bmpRendererBmp, file.c_str())) 
			{
				CRIA_AUTO_ASSERT(false, "CRSaveBitmap: bmp_renderer::SaveBitmap failed.");
				result = CRRES_ERROR;
				break;
			}

		} while (false);

		/*
		 * Delete
		 */
		if (bmpRendererBmp)
			bmp_renderer::DeleteBmp(bmpRendererBmp);
		if (intData)
			paco::CRPaCoFree(intData);

		/*
		 * return
		 */
		return result;
	}
}
