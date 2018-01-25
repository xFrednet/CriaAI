#pragma once
#include "../Common.hpp"

namespace cria_ai
{
	struct CRMatrixf
	{
		uint Cols; /* the width */
		uint Rows; /* the height */

		float* Data;
	};

	CRMatrixf* CreateMatrixf(uint cols, uint rows);
	void       FreeMatrixf(CRMatrixf* matrix);

	bool       SaveMatrixf(CRMatrixf* mat, char const* fileName);
	CRMatrixf* LoadMatrixf(char const* file);
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
	bool       WriteMatrixf(CRMatrixf* mat, char const* fileName, uint decimals = 3);

	void       FillMatrixRand(CRMatrixf* mat);

	float      GetMaxValue(CRMatrixf const* mat);
	float      GetMinValue(CRMatrixf const* mat);
	CRMatrixf* Clamp(CRMatrixf const* mat, float min, float max);

	CRMatrixf* Add(CRMatrixf const* a, CRMatrixf const* b);
	CRMatrixf* Sub(CRMatrixf const* a, CRMatrixf const* b);
	CRMatrixf* Mul(CRMatrixf const* a, CRMatrixf const* b);

	CRMatrixf* Mul(CRMatrixf const* a, float b);
	CRMatrixf* Div(CRMatrixf const* a, float b);
}
