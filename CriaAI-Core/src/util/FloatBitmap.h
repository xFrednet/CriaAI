#pragma once

#include "../Types.hpp"

#include "../../Dependencies/BmpRenderer/BmpRenderer.hpp"

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

	CR_FLOAT_BITMAP* CRCreateFBmp(uint32_t width, uint32_t height, uint8_t floatsPerPixel);
	CR_FLOAT_BITMAP* CRLoadFBmp(const char* file);
	CR_FLOAT_BITMAP* CRCreateFBmpCopy(CR_FLOAT_BITMAP* bmp);
	void             CRDeleteFBmp(CR_FLOAT_BITMAP* bmp);

	CR_FLOAT_BITMAP* CRConvertToFloatsPerPixel(CR_FLOAT_BITMAP const* bmp, uint8_t floatsPerPixel);

	/**
	 * \brief This function converts the content of this bitmap to a bitmap that uses
	 * one unsigned byte per pixel color. This is the format bitmaps are usually saved in.
	 *  
	 * \param bmp The bitmap that should be converted. (note that this bitmap )
	 * 
	 * \return 
	 */
	bmp_renderer::Bitmap* CRConvertToIntBmp(CR_FLOAT_BITMAP const* bmp);

	/**
	 * \brief This function tries to save the given bitmap to the provided file.
	 * 
	 * Note: This function is not trustworthy is just converts this bitmap to a bitmap
	 * from the BmpRenderer library and calls the save function for that bitmap.
	 * 
	 * \param bmp      The bitmap that should be saved
	 * \param fileName The name and path of the output file (if everything works out)
	 * 
	 * \return This returns true if everything works out (prepare for a false).
	 */
	bool             CRSaveBitmap(CR_FLOAT_BITMAP const* bmp, char const* fileName);

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
	CR_FLOAT_BITMAP* CRCalculateFeatureMap(CR_FLOAT_BITMAP const* bitmap, CR_FLOAT_BITMAP const* feature);
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
	CR_FLOAT_BITMAP* CRPoolBitmap(CR_FLOAT_BITMAP const* bitmap, uint32_t poolSize);
	/**
	 * \brief  This sets every negative value to zero.
	 * 
	 * \param  bitmap The bitmap that should be normalized.
	 * 
	 * \return This returns a new normalized bitmap.
	 */
	CR_FLOAT_BITMAP* CRNormalizeBitmap(CR_FLOAT_BITMAP const* bitmap);
}