#include "FloatBitmap.h"

#include "../Common.hpp"
#include "../../Dependencies/BmpRenderer/Dependencies/libbmpread/bmpread.h"
#include "../api/FileSystem.h"

#include "../api/cuda/CudaContext.cuh"

#define FBMP_PX_INDEX(x, y, bmp)            (((x) + (y) * bmp->Width) * bmp->FloatsPerPixel)

namespace cria_ai
{
	CR_FLOAT_BITMAP* CreateFBmp(uint32_t width, uint32_t height, uint8_t floatsPerPixel)
	{
		CRIA_AUTO_ASSERT(width != 0 && height != 0 && floatsPerPixel != 0,
			"CreateFloatBmp failed to create a bmp: width %u, height %u, floats per pixel %i",
			width, height, floatsPerPixel);
		if (width == 0 || height == 0 || floatsPerPixel == 0)
			return nullptr;

		CR_FLOAT_BITMAP* bmp;// = (CR_FLOAT_BITMAP*)malloc(sizeof(CR_FLOAT_BITMAP) + sizeof(float) * width * height * floatsPerPixel);
		cudaMallocManaged(&bmp, sizeof(CR_FLOAT_BITMAP) + sizeof(float) * width * height * floatsPerPixel);
		CRIA_AUTO_ASSERT(bmp,
			"CreateFloatBmp failed to create a bmp: width %u, height %u, floats per pixel %i",
			width, height, floatsPerPixel);
		if (!bmp)
			return nullptr;

		bmp->Width = width;
		bmp->Height = height;
		bmp->FloatsPerPixel = floatsPerPixel;
		bmp->Data = (float*)((uintptr_t)bmp + sizeof(CR_FLOAT_BITMAP));
		memset(bmp->Data, 0, sizeof(float) * width * height * floatsPerPixel);

		return bmp;
	}
	CR_FLOAT_BITMAP* LoadFBmp(const char* file)
	{
		CRIA_AUTO_ASSERT(file, "LoadFloatBmp the source file is undefined")
			if (!file)
				return nullptr;

		bmpread_t bmpIn;
		if (!bmpread(file, BMPREAD_TOP_DOWN | BMPREAD_ANY_SIZE | BMPREAD_LOAD_ALPHA, &bmpIn)) {
			return nullptr;
		}

		CR_FLOAT_BITMAP* bmp = CreateFBmp(bmpIn.width, bmpIn.height, 4);

		for (uint32_t index = 0; index < bmp->Width * bmp->Height * 4; index++) {
			bmp->Data[index] = (float)bmpIn.rgb_data[index] / 255.0f;
		}

		return bmp;
	}
	CR_FLOAT_BITMAP* CreateFBmpCopy(CR_FLOAT_BITMAP* bmp)
	{
		CRIA_AUTO_ASSERT(bmp, "CreateFBmpCopy: the source bmp should be existence");
		if (!bmp)
			return nullptr;

		CR_FLOAT_BITMAP* copy = CreateFBmp(bmp->Width, bmp->Height, bmp->FloatsPerPixel);
		CRIA_AUTO_ASSERT(copy, "CreateFBmpCopy: The copy bmp could not be created");
		if (!copy)
			return nullptr;
		
		//TODO does this work
		cudaMemcpy(&copy->Data[0], &bmp->Data[0], sizeof(float) * bmp->Width * bmp->Height * bmp->FloatsPerPixel, cudaMemcpyDeviceToDevice);
		//memcpy(&copy->Data[0], &bmp->Data[0], sizeof(float) * bmp->Width * bmp->Height * bmp->FloatsPerPixel);
		
		return copy;
	}
	void DeleteFBmp(CR_FLOAT_BITMAP* bmp)
	{
		if (bmp)
			cudaFree(bmp); /* this should free both the struct and the data since they are created by one malloc */
	}

	void ConvertTo1fpp(CR_FLOAT_BITMAP const* bmpIn, CR_FLOAT_BITMAP* bmpOut)
	{
		float total;
		uint32_t inIndex = 0;
		
		if (bmpIn->FloatsPerPixel == 4)
		{
			for (uint32_t index = 0; index < bmpIn->Width * bmpIn->Height; index++)
			{
				total  = bmpIn->Data[inIndex++]; /* Red   */
				total += bmpIn->Data[inIndex++]; /* Green */
				total += bmpIn->Data[inIndex++]; /* Blue  */
				total *= bmpIn->Data[inIndex++]; /* Alpha */
				total /= 3;

				bmpOut->Data[index] = total;
			}
		} else if (bmpIn->FloatsPerPixel == 3)
		{
			for (uint32_t index = 0; index < bmpIn->Width * bmpIn->Height; index++) {
				total  = bmpIn->Data[inIndex++]; /* Blue  */
				total += bmpIn->Data[inIndex++]; /* Green */
				total += bmpIn->Data[inIndex++]; /* Red   */
				total /= 3;

				bmpOut->Data[index] = total;
			}
		}
	}
	void ConvertTo3fpp(CR_FLOAT_BITMAP const* bmpIn, CR_FLOAT_BITMAP* bmpOut)
	{
		if (bmpIn->FloatsPerPixel == 4) 
		{
			uint32_t inIndex  = 0;
			uint32_t outInput = 0;
			for (uint32_t pixel = 0; pixel < bmpIn->Width * bmpIn->Height; pixel++) 
			{
				float alpha = bmpOut->Data[outInput + 4];

				bmpOut->Data[outInput++] = bmpIn->Data[inIndex++] * alpha;
				bmpOut->Data[outInput++] = bmpIn->Data[inIndex++] * alpha;
				bmpOut->Data[outInput++] = bmpIn->Data[inIndex++] * alpha;
			}
		} else if (bmpIn->FloatsPerPixel == 1)
		{
			uint32_t outInput = 0;
			for (uint32_t pixel = 0; pixel < bmpIn->Width * bmpIn->Height; pixel++) 
			{
				bmpOut->Data[outInput++] = bmpIn->Data[pixel];
				bmpOut->Data[outInput++] = bmpIn->Data[pixel];
				bmpOut->Data[outInput++] = bmpIn->Data[pixel];
			}
		}
	}
	void ConvertTo4fpp(CR_FLOAT_BITMAP const* bmpIn, CR_FLOAT_BITMAP* bmpOut)
	{
		if (bmpIn->FloatsPerPixel == 3) 
		{
			uint32_t inIndex = 0;
			uint32_t outInput = 0;
			for (uint32_t pixel = 0; pixel < bmpIn->Width * bmpIn->Height; pixel++)
			{
				bmpOut->Data[outInput++] = bmpIn->Data[inIndex++];
				bmpOut->Data[outInput++] = bmpIn->Data[inIndex++];
				bmpOut->Data[outInput++] = bmpIn->Data[inIndex++];
				bmpOut->Data[outInput++] = 1.0f;
			}
		} else if (bmpIn->FloatsPerPixel == 1)
		{
			uint32_t outInput = 0;
			for (uint32_t pixel = 0; pixel < bmpIn->Width * bmpIn->Height; pixel++) 
			{
				bmpOut->Data[outInput++] = bmpIn->Data[pixel];
				bmpOut->Data[outInput++] = bmpIn->Data[pixel];
				bmpOut->Data[outInput++] = bmpIn->Data[pixel];
				bmpOut->Data[outInput++] = 1.0f;
			}
		}
	}
	CR_FLOAT_BITMAP* ConvertToFloatsPerPixel(CR_FLOAT_BITMAP const* bmpIn, uint8_t floatsPerPixel)
	{
		/* validation */
		CRIA_AUTO_ASSERT(bmpIn, "ConvertToFloatsPerPixel: the submitted bmp is null.");
		CRIA_AUTO_ASSERT(floatsPerPixel == 1 || floatsPerPixel == 3 || floatsPerPixel == 4,
			"ConvertToFloatsPerPixel: The selected value for floats per pixel (%u) is unsupported.", floatsPerPixel);
		if (!bmpIn && !(floatsPerPixel == 1 || floatsPerPixel == 3 || floatsPerPixel != 4))
			return nullptr;

		/* creating the output bmp */
		CR_FLOAT_BITMAP* bmpOut = CreateFBmp(bmpIn->Width, bmpIn->Height, floatsPerPixel);
		CRIA_AUTO_ASSERT(bmpOut, "ConvertToFloatsPerPixel failed to create a bmp.");
		if (!bmpOut)
			return nullptr;

		/* selecting the converter */
		if (bmpIn->FloatsPerPixel == floatsPerPixel)
		{
			memcpy(&bmpOut->Data[0], &bmpIn->Data[0], sizeof(float) * bmpIn->Width * bmpIn->Height * floatsPerPixel);
			return bmpOut;
		}

		/* selecting the converter */
		switch (floatsPerPixel)
		{
			case 1:
				ConvertTo1fpp(bmpIn, bmpOut);
				break;
			case 3:
				ConvertTo3fpp(bmpIn, bmpOut);
				break;
			case 4:
				ConvertTo4fpp(bmpIn, bmpOut);
				break;
			default:
				break;
		}
		
		return bmpOut;
	}

#define CR_DEFAULT_ALPHA_VALUE         255
	bmp_renderer::Bitmap* ConvertToIntBmp(CR_FLOAT_BITMAP const* bmp)
	{
		/*
		 * Validation
		 */
		CRIA_AUTO_ASSERT(bmp, "ConvertToIntBmp: The provided float bitmap is false.");
		if (!bmp)
			return nullptr;

		/*
		 * creating the int bitmap
		 */
		bmp_renderer::Bitmap* intBmp = bmp_renderer::CreateBmp(bmp->Width, bmp->Height);
		CRIA_AUTO_ASSERT(intBmp, "ConvertToIntBmp: The creating of the int bitmap failed. FBitmap: %p", bmp);
		if (!intBmp)
			return nullptr;

		/*
		 * converting the data
		 */
		int dstIndex;
		int srcIndex = 0;
		switch (bmp->FloatsPerPixel)
		{
		case 4:
			for (dstIndex = 0; dstIndex < intBmp->WIDTH * intBmp->HEIGHT; dstIndex++)
			{
				intBmp->Data[dstIndex].R = (byte)(255.0f * bmp->Data[srcIndex++]);
				intBmp->Data[dstIndex].G = (byte)(255.0f * bmp->Data[srcIndex++]);
				intBmp->Data[dstIndex].B = (byte)(255.0f * bmp->Data[srcIndex++]);
				intBmp->Data[dstIndex].A = (byte)(255.0f * bmp->Data[srcIndex++]);
			}

			return intBmp;
		case 3:
			for (dstIndex = 0; dstIndex < intBmp->WIDTH * intBmp->HEIGHT; dstIndex++) {
				intBmp->Data[dstIndex].R = (byte)(255.0f * bmp->Data[srcIndex++]);
				intBmp->Data[dstIndex].G = (byte)(255.0f * bmp->Data[srcIndex++]);
				intBmp->Data[dstIndex].B = (byte)(255.0f * bmp->Data[srcIndex++]);
				intBmp->Data[dstIndex].A = CR_DEFAULT_ALPHA_VALUE;
			}

			return intBmp;
		case 1:
			for (dstIndex = 0; dstIndex < intBmp->WIDTH * intBmp->HEIGHT; dstIndex++) {
				intBmp->Data[dstIndex].R = (byte)(255.0f * bmp->Data[srcIndex]);
				intBmp->Data[dstIndex].G = (byte)(255.0f * bmp->Data[srcIndex]);
				intBmp->Data[dstIndex].B = (byte)(255.0f * bmp->Data[srcIndex]);
				intBmp->Data[dstIndex].A = CR_DEFAULT_ALPHA_VALUE;

				srcIndex++;
			}

			return intBmp;
		default:
			CRIA_AUTO_ASSERT(false, "ConvertToIntBmp: the float bitmap(%p) has a unsupported float per pixel count: %u",
				bmp, bmp->FloatsPerPixel);

			bmp_renderer::DeleteBmp(intBmp);
			return nullptr;
		}
	}

	bool SaveBitmap(CR_FLOAT_BITMAP const* bmp, char const* fileName)
	{
		/*
		 * validation
		 */
		CRIA_AUTO_ASSERT(fileName, "SaveBitmap: The file name is invalid. The bitmap address is: %p", (bmp) ? bmp : 0);
		CRIA_AUTO_ASSERT(bmp, "SaveBitmap: The bitmap is invalid. The file Name is: %s", fileName);
		if (!bmp || !fileName)
			return false; /* you broke it */

		/*
		 * making sure the directory exists. 
		 */
		if (!CreateContainingDir(fileName)) {
			CRIA_AUTO_ASSERT("The creation of the containing directory failed, file: \"%s\"", fileName);
			return false;
		}

		/*
		 * The conversion
		 * 
		 * Kids cover your eyes this should not be done.
		 */
		bmp_renderer::Bitmap* intBmp = ConvertToIntBmp(bmp);
		CRIA_AUTO_ASSERT(intBmp, "SaveBitmap: the conversion to the int bitmap failed.");
		if (!intBmp)
			return false;

		/*
		 * Saving
		 */
		if (!bmp_renderer::SaveBitmap(intBmp, fileName))
		{
			CRIA_AUTO_ASSERT(false, "SaveBitmap: The saving of the int bitmap failed");
			bmp_renderer::DeleteBmp(intBmp);
			return false;
		}

		bmp_renderer::DeleteBmp(intBmp);
		return true;

	}

	CR_FLOAT_BITMAP* CalculateFeatureMap(CR_FLOAT_BITMAP const* bitmap, CR_FLOAT_BITMAP const* feature)
	{
		//TODO CalculateFeatureMap
		return nullptr;
	}
	
	__global__ void PoolBitmapKernel(CR_FLOAT_BITMAP const* inBmp, CR_FLOAT_BITMAP* outBmp, uint poolCount, uint poolSize)
	{

		int index = blockIdx.x * blockDim.x + threadIdx.x;
		int stride = blockDim.x * gridDim.x;

		//int index = 0;
		//int stride = 1;

		uint outWidth = outBmp->Width;
		uint inWidth = inBmp->Width;
		uint inHeight = inBmp->Height;
		uint fpp = inBmp->FloatsPerPixel;

		for (uint poolNo = index; poolNo < poolCount; poolNo += stride)
		{
			uint poolX = poolNo % outWidth;
			uint poolY = poolNo / outWidth;
			
			/* pool */
			uint ya = poolY * poolSize;
			float maxValues[4];
			for (uint channel = 0; channel < inBmp->FloatsPerPixel; channel++) 
				maxValues[channel] = inBmp->Data[FBMP_PX_INDEX(poolX * poolSize, ya, inBmp) + channel];

			/* scanning the pool */
			uint xa;
			uint x;
			for (uint y = 0; y < poolSize && ya < inHeight; y++, ya++)
			{
				xa = poolX * poolSize;

				for (x = 0; x < poolSize && xa < inWidth; x++, xa++)
				{
					for (uint channel = 0; channel < inBmp->FloatsPerPixel; channel++)
					{
						if (maxValues[channel] < inBmp->Data[((xa + ya * inWidth) * fpp) + channel])
							maxValues[channel] = inBmp->Data[((xa + ya * inWidth) * fpp) + channel];
					}
				}
			}

			/* saving the result */
			for (uint channel = 0; channel < inBmp->FloatsPerPixel; channel++)
				outBmp->Data[FBMP_PX_INDEX(poolX, poolY, outBmp) + channel] = maxValues[channel];
		}
	}
	void PoolBitmapCPU(CR_FLOAT_BITMAP const* inBmp, CR_FLOAT_BITMAP* outBmp, uint poolCount, uint poolSize)
	{
		int index = 0;
		int stride = 1;

		for (uint poolNo = index; poolNo < poolCount; poolNo += stride) {
			uint poolX = poolNo % outBmp->Width;
			uint poolY = poolNo / outBmp->Width;

			/* pool */
			uint ya = poolY * poolSize;
			float maxValues[4];
			for (uint channel = 0; channel < inBmp->FloatsPerPixel; channel++)
				maxValues[channel] = inBmp->Data[FBMP_PX_INDEX(poolX * poolSize, ya, inBmp) + channel];

			/* scanning the pool */
			uint xa;
			uint x;
			for (uint y = 0; y < poolSize && ya < inBmp->Height; y++, ya++) {
				xa = poolX * poolSize;

				for (x = 0; x < poolSize && xa < inBmp->Width; x++, xa++) {
					for (uint channel = 0; channel < inBmp->FloatsPerPixel; channel++) {
						if (maxValues[channel] < inBmp->Data[FBMP_PX_INDEX(xa, ya, inBmp) + channel])
							maxValues[channel] = inBmp->Data[FBMP_PX_INDEX(xa, ya, inBmp) + channel];
					}
				}
			}

			/* saving the result */
			for (uint channel = 0; channel < inBmp->FloatsPerPixel; channel++)
				outBmp->Data[FBMP_PX_INDEX(poolX, poolY, outBmp) + channel] = maxValues[channel];
		}
	}
	CR_FLOAT_BITMAP* PoolBitmap(CR_FLOAT_BITMAP const* bmp, uint32_t poolSize)
	{
		/* input validation */
		CRIA_AUTO_ASSERT(bmp, "PoolBitmap the bitmap that should be pooled should also be valid.");
		CRIA_AUTO_ASSERT(poolSize != 0, "PoolBitmap: A pool size of 0 is invalid");
		if (!bmp || poolSize == 0)
			return nullptr;

		/* creating the output bitmap */
		CR_FLOAT_BITMAP* outBmp = CreateFBmp(
			((bmp->Width  / poolSize) + ((bmp->Width  % poolSize == 0) ? 0 : 1)),
			((bmp->Height / poolSize) + ((bmp->Height % poolSize == 0) ? 0 : 1)),
			bmp->FloatsPerPixel
		);
		CRIA_AUTO_ASSERT(outBmp, "PoolBitmap: Failed to create the output bitmap");
		if (!outBmp)
			return nullptr;

		/* pooling 
		 * I'm so sorry for the following messy code... 
		 */
		PoolBitmapKernel<<<8, 256>>>(bmp, outBmp, outBmp->Width * outBmp->Height, poolSize);
		cudaDeviceSynchronize();
		
		//PoolBitmapCPU(bmp, outBmp, outBmp->Width * outBmp->Height, poolSize);
		
		return outBmp;
	}
	CR_FLOAT_BITMAP* NormalizeBitmap(CR_FLOAT_BITMAP const* bmp)
	{
		/* input validation */
		CRIA_AUTO_ASSERT(bmp, "NormalizeBitmap: the given bitmap is invalid.");
		if (!bmp)
			return nullptr;

		/* creating the result bitmap */
		CR_FLOAT_BITMAP* result = CreateFBmp(bmp->Width, bmp->Height, bmp->FloatsPerPixel);
		CRIA_AUTO_ASSERT(result, "NormalizeBitmap: failed to create the result bitmap :/.");
		if (!result)
			return nullptr;

		/* normalizing the bitmap*/
		for (uint32_t index = 0; index < bmp->Width * bmp->Height * bmp->FloatsPerPixel; index++)
		{
			if (bmp->Data[index] >= 0)
				result->Data[index] = bmp->Data[index];
			else
				result->Data[index] = 0;
		}

		/* returning the result */
		return result;
	}
}