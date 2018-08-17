#include "Matrixf.hpp"

#include "../Common.hpp"

#include "../../Dependencies/BmpRenderer/BmpRenderer.hpp"
#include "../os/FileSystem.h"

#include "../paco/PaCoContext.h"

#define VALID_MAT(mat)                 (mat && mat->Cols != 0 && mat->Rows != 0)

namespace cria_ai
{
	/* //////////////////////////////////////////////////////////////////////////////// */
	// // CRMatFCreate //
	/* //////////////////////////////////////////////////////////////////////////////// */
	inline CR_MATF* CRMatFCreate(uint cols, uint rows, bool usePacoMalloc)
	{
		/*
		 * Validation
		 */
		CRIA_AUTO_ASSERT(cols != 0 && rows != 0, "A matrix with 0 columns or rows is dumdum");
		CRIA_AUTO_ASSERT(CR_CAN_UINT32_MUL(cols, rows), "The index would exceed UINT32_MAX in this case");
		if (cols == 0 || rows == 0 || !CR_CAN_UINT32_MUL(cols, rows))
			return nullptr;

		/*
		 * Memory allocation
		 */
		size_t matMemSize = sizeof(CR_MATF) + sizeof(float) * cols * rows;
		CR_MATF* matrix;
		if (usePacoMalloc)
		{
			matrix = (CR_MATF*)paco::CRPaCoMalloc(matMemSize);
		} 
		else
		{
			matrix = (CR_MATF*)malloc(matMemSize);
		}
		CRIA_AUTO_ASSERT(matrix, "");
		if (!matrix)
			return nullptr;

		/*
		 * Filling the memory
		 */
		matrix->Cols = cols;
		matrix->Rows = rows;
		matrix->Data = (float*)((uintptr_t)matrix + (uintptr_t)sizeof(CR_MATF));

		memset(matrix->Data, 0, sizeof(float) * cols * rows);

		/*
		 * Return the matrix
		 */
		return matrix;
	}
	CR_MATF* CRMatFCreatePaco(uint cols, uint rows)
	{
		return CRMatFCreate(cols, rows, true);
	}
	CR_MATF* CRMatFCreateNormal(uint cols, uint rows)
	{
		return CRMatFCreate(cols, rows, false);
	}
	
	/* //////////////////////////////////////////////////////////////////////////////// */
	// // CRMatFDelete //
	/* //////////////////////////////////////////////////////////////////////////////// */
	void       CRMatFDeletePaco(CR_MATF* matrix)
	{
		if (matrix)
			paco::CRPaCoFree(matrix);
	}
	void       CRMatFDeleteNormal(CR_MATF* matrix)
	{
		if (matrix)
			free(matrix);
	}


	/* //////////////////////////////////////////////////////////////////////////////// */
	// // CRMatF Save //
	/* //////////////////////////////////////////////////////////////////////////////// */
	typedef struct CR_MATF_FILE_HEADER_
	{
		byte   Magic[8];  /* these bytes contain the chars "CRAIMATF"*/
		uint32 Cols;      /* This is the number of columns of the matrix */
		uint32 Rows;      /* This is the number of rows of the matrix */
		uint32 DataStart; /* This is the number of bytes from the file/buffer beginning where the matrix data starts */
	} CR_MATF_FILE_HEADER;
	/*
	* Why do I need to write extra methods? well the structs may use padding that
	* isn't supported by the loader
	*/
	inline void WriteData(CR_BYTE_BUFFER* buffer, void* data, size_t size)
	{
		memcpy(&(buffer->Data[buffer->Position]), data, size);
		buffer->Position += size;
	}
	inline void ReadData(CR_BYTE_BUFFER const* data, void* outBuffer, size_t size)
	{
		memcpy(outBuffer, &(data->Data[data->Position]), size);
		data->Position += size;
	}
	inline void CRWriteMatFFileHeader(CR_BYTE_BUFFER* buffer, CR_MATF_FILE_HEADER* header)
	{
		WriteData(buffer, &header->Magic[0] , 8);
		WriteData(buffer, &header->Cols     , 4);
		WriteData(buffer, &header->Rows     , 4);
		WriteData(buffer, &header->DataStart, 4);
	}
	inline void CRReadMatFFileHeader(CR_BYTE_BUFFER const* data, CR_MATF_FILE_HEADER* header)
	{
		ReadData(data, &header->Magic[0] , 8);
		ReadData(data, &header->Cols     , 4);
		ReadData(data, &header->Rows     , 4);
		ReadData(data, &header->DataStart, 4);
	}

	size_t CRMatFGetSaveBufferSize(CR_MATF const* mat)
	{
		/*
		 * Validation
		 */
		if (!VALID_MAT(mat))
			return 0;

		/*
		 * Calculation
		 */
		size_t size = 0;
		size += 8 + 4 + 4 + 4; // CR_MATF_FILE_HEADER
		size += sizeof(float) * CR_MATF_VALUE_COUNT(mat); // The matrix data

		/*
		 * return
		 */
		return size;
	}

	crresult CRMatFSave(CR_MATF const* mat, CR_BYTE_BUFFER* buffer)
	{
		/*
		 * Validation check
		 */
		if (!VALID_MAT(mat) || !CRByteBufferValid(buffer))
			return CRRES_ERR_INVALID_ARGUMENTS;
		if (buffer->Size < CRMatFGetSaveBufferSize(mat))
			return CRRES_ERR_INVALID_BYTE_BUFFER_SIZE;

		size_t bufferStartPos = buffer->Position;

		/*
		 * File header
		 */
		CR_MATF_FILE_HEADER header;
		memcpy(&header.Magic[0], "CRAIMATF", 8);
		header.Cols = mat->Cols;
		header.Rows = mat->Rows;
		header.DataStart = 8 + 4 + 4 + 4;

		/*
		 * Write the data
		 */
		CRWriteMatFFileHeader(buffer, &header);
		buffer->Position = bufferStartPos + header.DataStart;
		WriteData(buffer, &(mat->Data[0]), sizeof(float) * mat->Cols * mat->Rows);

		return CRRES_OK;
	}
	crresult CRMatFSave(CR_MATF const* mat, const String& fileName)
	{
		/*
		 * Validation
		 */
		if (!VALID_MAT(mat) || fileName.empty())
			return CRRES_ERR_INVALID_ARGUMENTS;

		/*
		 * Create byte buffer
		 */
		CR_BYTE_BUFFER* buffer = CRByteBufferCreate(CRMatFGetSaveBufferSize(mat));
		if (!CRByteBufferValid(buffer))
			return CRRES_ERR_FAILED_TO_CREATE_BYTE_BUFFER;

		/*
		 * Call the CRMatFSave function with the byte buffer
		 */
		crresult result = CRMatFSave(mat, buffer);
		if (CR_FAILED(result))
		{
			CRByteBufferDelete(buffer);
			return result;
		}

		/*
		 * Save the byte buffer content
		 */
		result = CRFileWrite(fileName, buffer); 
		// Note I don't care if the result is negative I have to delete the buffer and return the result anyways

		CRByteBufferDelete(buffer);
		return result; //bye bye
	}
	CR_MATF* CRMatFLoad(CR_BYTE_BUFFER const* buffer, crresult* result)
	{
		/*
		 * Validation
		 */
		if (!CRByteBufferValid(buffer))
		{
			if (result)
				*result = CRRES_ERR_INVALID_ARGUMENTS;
			return nullptr;
		}
		size_t bufferStartPos = buffer->Position;

		/*
		 * fill the file header and check for validation
		 */
		CR_MATF_FILE_HEADER header;
		CRReadMatFFileHeader(buffer, &header);
		
		// validation
		if (memcmp(&header.Magic[0], "CRAIMATF", 8) != 0 ||
			header.Cols == 0 || header.Rows == 0)
		{
			if (result)
				*result = CRRES_ERR_OS_FILE_FORMAT_UNKNOWN;
			return nullptr;
		}
		
		size_t dataSize = sizeof(float) * header.Cols * header.Rows;
		if (bufferStartPos + header.DataStart + dataSize > buffer->Size)
		{
			if (result)
				*result = CRRES_ERR_INVALID_BYTE_BUFFER_SIZE;
			return nullptr;
		}

		/*
		 *  Create the matrix and read the data
		 */
		CR_MATF* mat = CRMatFCreate(header.Cols, header.Rows);
		if (!CRMatFValid(mat))
		{
			if (result)
				*result = CRRES_ERR_MALLOC_FAILED;
			return nullptr;
		}
		buffer->Position = bufferStartPos + header.DataStart;
		ReadData(buffer, mat->Data, dataSize);

		/*
		 * throw the matrix back I don't want it anymore
		 */
		if (result)
			*result = CRRES_OK;
		return mat;
	}
	CR_MATF* CRMatFLoad(const String& fileName, crresult* result)
	{
		/*
		* Validation
		*/
		if (fileName.empty())
		{
			if (result)
				*result = CRRES_ERR_INVALID_ARGUMENTS;
			return nullptr;
		}

		/*
		* Create byte buffer
		*/
		CR_BYTE_BUFFER* buffer = CRFileRead(fileName, result);
		if (!CRByteBufferValid(buffer))
		{
			if (buffer)
				CRByteBufferDelete(buffer);

			// The error should be saved to the result by the CRFileRead function
			return nullptr;
		}

		/*
		* Call the CRMatFSave function with the byte buffer
		*/
		CR_MATF* mat = CRMatFLoad(buffer, result);
		if (!CRMatFValid(mat))
		{
			if (buffer)
				CRByteBufferDelete(buffer);
			if (mat)
				CRMatFDelete(mat);

			// The error should be saved to the result by the CRMatFLoad function
			return nullptr; 
		}

		/*
		* Finish the loading
		*/
		if (result)
			*result = CRRES_OK;
		CRByteBufferDelete(buffer);
		return mat; 
	}

	crresult CRMatFSaveAsText(CR_MATF const* mat, std::ofstream& file, uint decimals)
	{
		using namespace std;

		/*
		 * Validation
		 */
		if (!CRMatFValid(mat) || !file.good())
			return CRRES_ERR_INVALID_ARGUMENTS;

		/*
		 * init the format values
		 */
		uint predecimals = (uint)log10f(
			MAX(abs(CRMatFGetMaxValue(mat)), abs(CRMatFGetMinValue(mat))) /* Getting the longest value */
			) + 2 /* plus one for the sign and one because log10 returns one too short */;

		/*
		 * Create buffer
		 */
		size_t bufferSize = 1 + predecimals + 1 + decimals + 1 + 1;
		char* buffer = new char[bufferSize];
		if (!buffer)
			return CRRES_ERR_NEW_FAILED;
		memset(buffer, '\0', bufferSize);
		
		/*
		 * Write 
		 */
		for (uint index = 0; index < CR_MATF_VALUE_COUNT(mat); index++)
		{
			// format and write
			sprintf(buffer, "%+*.*f ", predecimals, decimals, mat->Data[index]);
			file << buffer;

			// add a line break after row
			if ((index + 1) % mat->Cols == 0)
				file << CR_FILE_ENDL;

			// Error check
			if (!file.good())
				return CRRES_ERR_OS_WRITE_TO_FILE_FAILED; // since I don't open the file I'll keep it open
		}

		/*
		 * Finishing
		 */
		delete[] buffer;

		return CRRES_OK; // since I don't open the file I'll keep it open
	}
	crresult CRMatFSaveAsText(CR_MATF const* mat, const String& fileName, uint decimals)
	{
		/*
		 * Validation
		 */
		if (!CRMatFValid(mat) || fileName.empty())
			return CRRES_ERR_INVALID_ARGUMENTS;

		/*
		 * Open file
		 */
		std::ofstream file = CROpenFileOut(fileName);
		if (!file.good())
		{
			file.close();
			return CRRES_ERR_OS_FILE_COULD_NOT_BE_OPENED;
		}
		
		/*
		 * Save the matrix
		 */
		crresult result = CRMatFSaveAsText(mat, file, decimals);

		/*
		 * Calling the cleaning crew
		 */
		file.close();

		return result;
	}

	bool     CRMatFValid(CR_MATF const* mat)
	{
		return VALID_MAT(mat);
	}

	void     CRMatFFillRand(CR_MATF* mat)
	{
		uint index;

		CRIA_AUTO_ASSERT(VALID_MAT(mat), "The given matrix is not valid");
		if (!VALID_MAT(mat))
			return;

		for (index = 0; index < mat->Rows * mat->Cols; index++)
		{
			mat->Data[index] = (CRRandFloat() * 2.0f) - 1.0f;
		}
	}

	float    CRMatFGetMaxValue(CR_MATF const* mat)
	{
		uint index;
		float min;

		/* validation */
		CRIA_AUTO_ASSERT(VALID_MAT(mat), "I can't find the max value for a invalid matrix");
		if (!VALID_MAT(mat))
			return -FLT_MAX;

		/* searching */
		min = mat->Data[0];
		for (index = 1; index < mat->Cols * mat->Rows; index++)
		{
			if (mat->Data[index] < min)
				min = mat->Data[index];
		}

		/* return */
		return min;
	}
	float    CRMatFGetMinValue(CR_MATF const* mat)
	{
		uint index;
		float max;

		/* validation */
		CRIA_AUTO_ASSERT(VALID_MAT(mat), "I can't find the min value for a invalid matrix");
		if (!VALID_MAT(mat))
			return FLT_MAX;

		/* searching */
		max = mat->Data[0];
		for (index = 1; index < mat->Cols * mat->Rows; index++)
		{
			if (mat->Data[index] > max)
				max = mat->Data[index];
		}

		/* return */
		return max;
	}
	CR_MATF* CRMatFClamp(CR_MATF const* srcMat, float min, float max)
	{
		uint index;
		CR_MATF* dstMat;

		/* validate */
		CRIA_AUTO_ASSERT(VALID_MAT(srcMat), "The matrix has to be valid.");
		CRIA_AUTO_ASSERT(min < max, "min should be less than max");
		if (!VALID_MAT(srcMat) || min > max)
			return 0;

		/* creating the matrix */
		dstMat = CRMatFCreate(srcMat->Cols, srcMat->Rows);
		CRIA_AUTO_ASSERT(dstMat, "Failed to create the output matrix");
		if (!dstMat)
			return 0;

		/* clamping */
		for (index = 0; index < srcMat->Cols * srcMat->Rows; index++)
		{
			if (srcMat->Data[index] > max)
				dstMat->Data[index] = max;
			else if (srcMat->Data[index] < min)
				dstMat->Data[index] = min;
			else
				dstMat->Data[index] = srcMat->Data[index];
		}

		return dstMat;
	}
	float    CRMatFSum(CR_MATF const* mat)
	{
		/*
		 * Validation
		 */
		if (!mat)
			return 0;

		/*
		 * Add all values together aka sum
		 */
		float sum = 0.0f;
		for (uint index = 0; index < CR_MATF_VALUE_COUNT(mat); index++)
		{
			sum += mat->Data[index];
		}

		return sum;
	}

	CR_MATF* CRMatFAdd(CR_MATF const* a, CR_MATF const* b)
	{
		CR_MATF* mat;
		uint index;

		/* matrix validation */
		CRIA_AUTO_ASSERT(VALID_MAT(a) || VALID_MAT(b), "Sorry is wasn't me... I hope");
		CRIA_AUTO_ASSERT(a->Cols == b->Cols || a->Rows == b->Rows, "Different dimensions yay... nooooooooooo");
		if (!VALID_MAT(a) || !VALID_MAT(b) ||
			a->Cols != b->Cols || a->Rows != b->Rows)
			return 0;

		/* matrix creation */
		mat = CRMatFCreate(a->Cols, a->Rows);
		CRIA_AUTO_ASSERT(mat, "Matrix creation failed!");
		if (!mat)
			return 0;

		/* adding the numbers */
		for (index = 0; index < a->Cols * a->Rows; index++)
		{
			mat->Data[index] = a->Data[index] + b->Data[index];
		}

		return mat;
	}
	CR_MATF* CRMatFSub(CR_MATF const* a, CR_MATF const* b)
	{
		CR_MATF* mat;
		uint index;

		/* matrix validation */
		CRIA_AUTO_ASSERT(VALID_MAT(a) || VALID_MAT(b), "Sorry is wasn't me... I hope");
		CRIA_AUTO_ASSERT(a->Cols == b->Cols || a->Rows == b->Rows, "Different dimensions yay... nooooooooooo");
		if (!VALID_MAT(a) || !VALID_MAT(b) ||
			a->Cols != b->Cols || a->Rows != b->Rows)
			return 0;

		/* matrix creation */
		mat = CRMatFCreate(a->Cols, a->Rows);
		CRIA_AUTO_ASSERT(mat, "Matrix creation failed!");
		if (!mat)
			return 0;

		/* adding the numbers */
		for (index = 0; index < a->Cols * a->Rows; index++) {
			mat->Data[index] = a->Data[index] - b->Data[index];
		}

		return mat;
	}
	CR_MATF* CRMatFMul(CR_MATF const* a, CR_MATF const* b)
	{
		CR_MATF* mat;

		/* validation */
		CRIA_AUTO_ASSERT(VALID_MAT(a) && VALID_MAT(b), "The matrices have to be valid sorry.");
		CRIA_AUTO_ASSERT(a->Rows == b->Cols, "The rows form matrix a and the columns from matrix b have to be the same.");
		if (!VALID_MAT(a) || !VALID_MAT(b) ||
			a->Rows != b->Cols)
			return 0;

		/* create the output matrix */
		mat = CRMatFCreate(a->Cols, b->Rows);
		CRIA_AUTO_ASSERT(mat, "The creation of the output matrix failed!")
		if (!mat)
			return 0;

		/* multiplication (not optimized)
		 * a : move along cols (starts at result row)
		 * b : move along rows (starts at result column)
		 */
		uint calCount = a->Rows;
		for (uint index = 0; index < mat->Cols * mat->Rows; index++)
		{
			uint matAIndex = index / mat->Rows;
			uint matBIndex = index % mat->Rows;

			for (uint calNo = 0; calNo < calCount; calNo++)
			{
				mat->Data[index] += a->Data[matAIndex] * b->Data[matBIndex];

				matAIndex += 1;
				matBIndex += b->Rows;
			}
		}

		return mat;
	}

	CR_MATF* CRMatFMul(CR_MATF const* a, float b)
	{
		CR_MATF* mat;
		uint index;

		/* matrix validation */
		CRIA_AUTO_ASSERT(VALID_MAT(a), "Sorry is wasn't me... I hope");
		if (!VALID_MAT(a))
			return 0;

		/* matrix creation */
		mat = CRMatFCreate(a->Cols, a->Rows);
		CRIA_AUTO_ASSERT(mat, "Matrix creation failed!");
		if (!mat)
			return 0;

		/* adding the numbers */
		for (index = 0; index < a->Cols * a->Rows; index++) {
			mat->Data[index] = a->Data[index] * b;
		}

		return mat;
	}
	CR_MATF* CRMatFDiv(CR_MATF const* a, float b)
	{
		CRIA_AUTO_ASSERT(b == 0, "Devision by 0 is undefined behavior");
		if (b == 0)
			return nullptr;

		return CRMatFMul(a, 1.0f / b);
	}
}
 