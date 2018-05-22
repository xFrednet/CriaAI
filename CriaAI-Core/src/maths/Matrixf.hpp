#pragma once
#include "../Types.hpp"

namespace cria_ai
{
	struct CRMatrixf
	{
		uint Cols; /* the width */
		uint Rows; /* the height */

		float* Data;
	};

	CRMatrixf* CRCreateMatrixf(uint cols, uint rows);
	void       CRFreeMatrixf(CRMatrixf* matrix);

	bool       CRSaveMatrixf(CRMatrixf* mat, char const* fileName);
	CRMatrixf* CRLoadMatrixf(char const* file);
	/**
	 * \brief This function writes matrix in form of multiple floats
	 * rounded up to the number of decimals specified.
	 * 
	 * (The loading of this format is currently not supported)
	 * 
	 * \param mat      The CRMatrixf that should be saved
	 * \param fileName The name of the destination file. (It will be created or overridden)
	 * \param decimals The amount of decimals that should be written
	 * 
	 * \return This returns 1(true) on success.
	 */
	bool       CRWriteMatrixf(CRMatrixf* mat, char const* fileName, uint decimals = 3);
	bool       CRWriteMatrixfBmp(CRMatrixf* mat, char const* fileName);

	bool       CRIsMatValid(CRMatrixf* mat);

	void       CRFillMatrixRand(CRMatrixf* mat);

	float      CRGetMaxValue(CRMatrixf const* mat);
	float      CRGetMinValue(CRMatrixf const* mat);
	CRMatrixf* CRClamp(CRMatrixf const* mat, float min, float max);

	CRMatrixf* CRAdd(CRMatrixf const* a, CRMatrixf const* b);
	CRMatrixf* CRSub(CRMatrixf const* a, CRMatrixf const* b);
	CRMatrixf* CRMul(CRMatrixf const* a, CRMatrixf const* b);

	CRMatrixf* CRMul(CRMatrixf const* a, float b);
	CRMatrixf* CRDiv(CRMatrixf const* a, float b);
}
