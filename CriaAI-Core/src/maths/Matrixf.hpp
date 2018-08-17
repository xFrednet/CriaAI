#pragma once
#include "../Types.hpp"
#include "../util/CRResult.h"
#include "../os/FileSystem.h"

#define CR_MATF_DATA_SIZE(mat)                   (sizeof(float) * (mat)->Cols * (mat)->Rows)
#define CR_MATF_FILL_ZERO(mat)                   (memset((mat)->Data, 0, CR_MATF_DATA_SIZE(mat)))
#define CR_MATF_FILL_ZERO_IF_VALID(mat)          if (mat) {CR_MATF_FILL_ZERO(mat);}
#define CR_MATF_COPY_DATA(dst, src)              memcpy((dst)->Data, (src)->Data, CR_MATF_DATA_SIZE(src))
#define CR_MATF_VALUE_INDEX(col, row, mat)       ((col) + (row) * (mat)->Cols)
#define CR_MATF_VALUE_COUNT(mat)                 ((mat)->Cols * (mat)->Rows)

namespace cria_ai
{
	struct CR_MATF
	{
		uint Cols; /* the width */
		uint Rows; /* the height */

		float* Data;
	};

	/*
	 * CRMatFCreate
	 */
	CR_MATF* CRMatFCreatePaco(uint cols, uint rows);
	CR_MATF* CRMatFCreateNormal(uint cols, uint rows);
	inline CR_MATF* CRMatFCreate(uint cols, uint rows)
	{
		return CRMatFCreatePaco(cols, rows);
	}

	/*
	 * CRMatFDelete
	 */
	void CRMatFDeletePaco(CR_MATF* matrix);
	void CRMatFDeleteNormal(CR_MATF* matrix);
	inline void CRMatFDelete(CR_MATF* matrix)
	{
		CRMatFDeletePaco(matrix);
	}

	/*
	 * CRMatF utility
	 */
	size_t   CRMatFGetSaveBufferSize(CR_MATF const* mat);

	crresult CRMatFSave(CR_MATF const* mat, CR_BYTE_BUFFER* buffer);
	crresult CRMatFSave(CR_MATF const* mat, const String& fileName);

	CR_MATF* CRMatFLoad(CR_BYTE_BUFFER const* buffer, crresult* result = nullptr);
	CR_MATF* CRMatFLoad(const String& file, crresult* result = nullptr);

	/**
	 * \brief This function writes matrix in form of multiple floats
	 * rounded up to the number of decimals specified.
	 * 
	 * (The loading of this format is currently not supported)
	 * 
	 * \param mat      The CR_MATF that should be saved
	 * \param file     The destination file that should be written do.
	 * \param decimals The amount of decimals that should be written
	 * 
	 * \return This returns 1(true) on success.
	 */
	crresult CRMatFSaveAsText(CR_MATF const* mat, std::ofstream& file    , uint decimals = 3);
	crresult CRMatFSaveAsText(CR_MATF const* mat, const String&  fileName, uint decimals = 3);

	bool     CRMatFValid(CR_MATF const* mat);

	void     CRMatFFillRand(CR_MATF* mat);

	float    CRMatFGetMaxValue(CR_MATF const* mat);
	float    CRMatFGetMinValue(CR_MATF const* mat);
	CR_MATF* CRMatFClamp(CR_MATF const* mat, float min, float max);
	float    CRMatFSum(CR_MATF const* mat);

	/* //////////////////////////////////////////////////////////////////////////////// */
	// // CRMatF operators //
	/* //////////////////////////////////////////////////////////////////////////////// */
	CR_MATF* CRMatFAdd(CR_MATF const* a, CR_MATF const* b);
	CR_MATF* CRMatFSub(CR_MATF const* a, CR_MATF const* b);
	CR_MATF* CRMatFMul(CR_MATF const* a, CR_MATF const* b);

	CR_MATF* CRMatFMul(CR_MATF const* a, float b);
	CR_MATF* CRMatFDiv(CR_MATF const* a, float b);
}
