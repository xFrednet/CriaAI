#pragma once

#include "../Types.hpp"

namespace cria_ai
{
	struct CRMatrixf;

	/*
	 *  (^v^)
	 * ((   ))
	 *   v v
	 * 
	 * 
	 * This is the code reviewing penguin. 
	 */

	typedef struct CR_FLOAT_BITMAP_ {
		uint32_t Width;
		uint32_t Height;

		uint8_t FloatsPerPixel;

		float* Data; /* r, g, b, a*/
	} CR_FLOAT_BITMAP;

	CR_FLOAT_BITMAP* CreateFBmp(uint32_t width, uint32_t height, uint8_t floatsPerPixel);
	CR_FLOAT_BITMAP* LoadFBmp(const char* file);
	CR_FLOAT_BITMAP* CreateFBmpCopy(CR_FLOAT_BITMAP* bmp);
	void             DeleteFBmp(CR_FLOAT_BITMAP* bmp);

	CR_FLOAT_BITMAP* ConvertToFloatsPerPixel(CR_FLOAT_BITMAP* bmp, uint8_t floatsPerPixel);
	
	/*
	 * The following functions are used in Convolutional Neural Network.
	 * Brandon Rohrer summed up this type of network ind this video:
	 * https://www.youtube.com/watch?v=FmpDIaiMIeA 
	 */

	/**
	 * \brief This tries to apply the given feature to every position in the bitmap.
	 * The result is a 1 float per pixel bitmap with floats ranging from 0(totally different)
	 * to 1(exactly the same).
	 * 
	 * In case of mismatching floats per pixel between the bitmap and the feature bitmap,
	 * the bitmap will be translated to use the float per pixel count of the feature.
	 * 
	 * \param bitmap  The bitmap where the feature is searched in.
	 * \param feature The feature that is searched for in the bitmap
	 * 
	 * \return This returns a new bitmap with the result of the calculation.
	 */
	CR_FLOAT_BITMAP* CalculateFeatureMap(CR_FLOAT_BITMAP* bitmap, CR_FLOAT_BITMAP* feature);
	/**
	 * \brief This scales down the bitmap by creating a pool with the size given by poolSize
	 * the highest number from this pool is than copied into the resulting bitmap.
	 * 
	 * The resulting bitmaps dimensions equal the source bitmaps dimensions divided by the poolSize.
	 * (rounded up). The pool is created and the value selected for every channel.
	 * 
	 * \param bitmap   The bitmap that the pools are created from.
	 * \param poolSize The width and height of each pool. (0 is invalid and 1 is unnecessary)
	 * 
	 * \return This returns a new bitmap with the result of the pooling.
	 */
	CR_FLOAT_BITMAP* PoolBitmap(CR_FLOAT_BITMAP* bitmap, uint32_t poolSize);
	/**
	 * \brief  This sets every negative value to zero.
	 * \param  bitmap The bitmap that should be normalized.
	 * \return This returns a new normalized bitmap.
	 */
	CR_FLOAT_BITMAP* NormalizeBitmap(CR_FLOAT_BITMAP* bitmap);
}