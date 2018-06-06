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
#include "Bitmap.h"

#include "../paco/PaCoContext.h"
#include "../paco/BitmapUtil.h"

#include "../../Dependencies/BmpRenderer/Dependencies/libbmpread/bmpread.h"
#include "../os/FileSystem.h"

namespace cria_ai
{
	/*
	* CRCreateBmp
	*/
#define CRIA_BMP_IS_CREATE_INPUT_VALID(width, height, bpp) \
	(width != 0 && height != 0 && bpp != 0 && bpp <= CR_BMP_MAX_BPP)
#define CRIA_BMP_FILL_MEMBERS(bmp, width, height, bpp) \
bmp->Width = width;\
bmp->Height = height;\
bmp->Bpp = bpp;\
bmp->Data = (byte*)((uintptr_t)bmp + sizeof(CR_BMP));\
memset(bmp->Data, 0, CR_BMP_DATA_SIZE(bmp));

	CR_BMP* CRCreateBmpPACO(uint width, uint height, uint bpp)
	{
		/*
		 * Validation
		 */
		CRIA_AUTO_ASSERT(CRIA_BMP_IS_CREATE_INPUT_VALID(width, height, bpp), "CRCreateBmpPACO invalid input: width %u, height %u, bpp %i", width, height, bpp);
		if (!CRIA_BMP_IS_CREATE_INPUT_VALID(width, height, bpp))
			return nullptr;

		/*
		 * Creation
		 */
		size_t dataSize = sizeof(byte) * width * height * bpp;
		CR_BMP* bmp = (CR_BMP*)paco::CRPaCoMalloc(sizeof(CR_BMP) + dataSize);
		CRIA_AUTO_ASSERT(bmp, "CRCreateBmpPACO paco::CRPaCoMalloc failed: width %u, height %u, bpp %i", width, height, bpp);
		if (!bmp)
			return nullptr;

		/*
		 * Fill struct members
		 */
		CRIA_BMP_FILL_MEMBERS(bmp, width, height, bpp);
		
		return bmp;
	}
	CR_BMP* CRCreateBmpNormal(uint width, uint height, uint bpp)
	{
		/*
		* Validation
		*/
		CRIA_AUTO_ASSERT(CRIA_BMP_IS_CREATE_INPUT_VALID(width, height, bpp), "CRCreateBmpNormal invalid input: width %u, height %u, bpp %i", width, height, bpp);
		if (!CRIA_BMP_IS_CREATE_INPUT_VALID(width, height, bpp))
			return nullptr;

		/*
		* Creation
		*/
		size_t dataSize = sizeof(byte) * width * height * bpp;
		CR_BMP* bmp = (CR_BMP*)malloc(sizeof(CR_BMP) + dataSize);
		CRIA_AUTO_ASSERT(bmp, "CRCreateBmpPACO malloc failed: width %u, height %u, bpp %i", width, height, bpp);
		if (!bmp)
			return nullptr;

		/*
		* Fill struct members
		*/
		CRIA_BMP_FILL_MEMBERS(bmp, width, height, bpp);

		return bmp;
	}

	/*
	* CRDeleteBmp
	*/
	void CRDeleteBmpPACO(CR_BMP* bmp)
	{
		if (bmp)
			paco::CRPaCoFree(bmp);
	}
	void CRDeleteBmpNormal(CR_BMP* bmp)
	{
		if (bmp)
			free(bmp);
	}

	CR_BMP* CRLoadBmp(const String& file, crresult* result)
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
		CR_BMP* bmp = CRCreateBmp(bmpReadIn.width, bmpReadIn.height, 4);
		memcpy(bmp->Data, bmpReadIn.rgb_data, CR_BMP_DATA_SIZE(bmp));

		for (uint i = 0; i < 4; i++)
		{
			std::cout << (uint)bmp->Data[i] << std::endl;
		}

		//return
		return bmp;
	}
	CR_BMP* CRCreateCopyBmp(CR_BMP const* srcBmp)
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
		CR_BMP* dstBmp = CRCreateBmp(srcBmp->Width, srcBmp->Height, srcBmp->Bpp);
		memcpy(dstBmp->Data, srcBmp->Data, CR_BMP_DATA_SIZE(srcBmp));

		//return
		return dstBmp;
	}
	crresult CRSaveBmp(CR_BMP const* bmp, const String& file)
	{
		/*
		* validation
		*/
		CRIA_AUTO_ASSERT(!file.empty(), "CRSaveBmp: The file name is invalid. The bitmap address is: %p", (bmp) ? bmp : nullptr);
		CRIA_AUTO_ASSERT(bmp, "CRSaveBmp: The bitmap is invalid. The file Name is: %s", file.c_str());
		if (!bmp || file.empty())
			return CRRES_ERR_INVALUD_ARGUMENTS; /* you broke it */

		 /*
		 * making sure the directory exists.
		 */
		if (!CreateContainingDir(file)) {
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
		bmp_renderer::Bitmap* bmpRendererBmp = bmp_renderer::CreateBmp(bmp->Width, bmp->Height);
		CRIA_AUTO_ASSERT(bmpRendererBmp, "CRSaveBmp: bmp_renderer::CreateBmp failed.");
		if (!bmpRendererBmp)
			return CRRES_ERROR;

		if (bmp->Bpp == 4)
		{
			memcpy(bmpRendererBmp->Data, bmp->Data, CR_BMP_DATA_SIZE(bmp));
		}
		else
		{
			CR_BMP* bmp4bpp = paco::CRBmpConvertToBPP(bmp, 4);
			if (!bmp4bpp)
			{
				bmp_renderer::DeleteBmp(bmpRendererBmp);
				return CRRES_ERROR;
			}
			memcpy(bmpRendererBmp->Data, bmp4bpp->Data, CR_BMP_DATA_SIZE(bmp4bpp));
			CRDeleteBmp(bmp4bpp);
		}
		/*
		* Saving
		*/
		if (!bmp_renderer::SaveBitmap(bmpRendererBmp, file.c_str())) {
			CRIA_AUTO_ASSERT(false, "CRSaveBitmap: bmp_renderer::SaveBitmap failed.");
			bmp_renderer::DeleteBmp(bmpRendererBmp);
			return CRRES_ERROR;
		}

		bmp_renderer::DeleteBmp(bmpRendererBmp);
		return CRRES_OK;
	}
}
