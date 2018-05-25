#include "Matrixf.hpp"

#include "../Common.hpp"

#include "../../Dependencies/BmpRenderer/BmpRenderer.hpp"
#include "../os/FileSystem.h"

#include "../paco/PaCoContext.h"

#define VALID_MAT(mat)                 (mat && mat->Cols != 0 && mat->Rows != 0)

namespace cria_ai
{
	CRMatrixf* CRCreateMatrixf(uint cols, uint rows)
	{
		CRIA_AUTO_ASSERT(cols != 0 && rows != 0, "A matrix with 0 columns or rows is dumdum");
		CRIA_AUTO_ASSERT(CR_CAN_UINT32_MUL(cols, rows), "The index would exceed UINT32_MAX in this case");
		if (cols == 0 || rows == 0 || !CR_CAN_UINT32_MUL(cols, rows))
			return nullptr;

		CRMatrixf* matrix = (CRMatrixf*)paco::CRPaCoMalloc(sizeof(CRMatrixf) + sizeof(float) * cols * rows);
		CRIA_AUTO_ASSERT(matrix, "");

		matrix->Cols = cols;
		matrix->Rows = rows;
		matrix->Data = (float*)((uintptr_t)matrix + (uintptr_t)sizeof(CRMatrixf));

		memset(matrix->Data, 0, sizeof(float) * cols * rows);

		return matrix;
	}
	void       CRFreeMatrixf(CRMatrixf* matrix)
	{
		if (matrix)
			paco::CRPaCoFree(matrix);
	}

	struct CR_FILE_HEADER
	{
		byte   Magic[8];  /* these bytes contain the chars "CRAIMATF"*/
		uint32 Cols;      /* This is the number of columns of the matrix */
		uint32 Rows;      /* This is the number of rows of the matrix */
		uint32 DataStart; /* This is the information where the matrix data starts */
	};
	/*
	* Why do I need to write extra methods? well the structs may use padding that
	* isn't supported by loaders
	*/
	inline int WriteData(FILE* file, void* data, size_t size)
	{
		return fwrite(data, 1, size, file) == size;
	}
	inline int ReadData(FILE* file, void* buffer, size_t size)
	{
		return fread(buffer, 1, size, file) == size;
	}
	inline FILE* fileopen(const char* fileName, char const* mode)
	{
		/*
		 * create directory
		 */
		if (!CreateContainingDir(fileName))
		{
			CRIA_AUTO_ASSERT("The creation of the containing directory failed, file: \"%s\"", fileName);
			return nullptr;
		}

		/*
		 * open file
		 */
		FILE* file = nullptr;
		errno_t err = fopen_s(&file, fileName, mode);
		CRIA_AUTO_ASSERT(file, "fopen_s failed to open the file. fileName:\"%s\", mode:\"%s\", errno_t\"%i\"", fileName, mode, err);
		if (!file)
			return nullptr;

		return file;
	}
	inline int WriteFileHeader(FILE* file, CR_FILE_HEADER* header)
	{
		if (!WriteData(file, &header->Magic[0] , 8)) return 0;
		if (!WriteData(file, &header->Cols     , 4)) return 0;
		if (!WriteData(file, &header->Rows     , 4)) return 0;
		if (!WriteData(file, &header->DataStart, 4)) return 0;
		return 1;
	}
	inline int ReadFileHeader(FILE* file, CR_FILE_HEADER* header)
	{
		if (!ReadData(file, &header->Magic[0] , 8)) return 0;
		if (!ReadData(file, &header->Cols     , 4)) return 0;
		if (!ReadData(file, &header->Rows     , 4)) return 0;
		if (!ReadData(file, &header->DataStart, 4)) return 0;
		return 1;
	}
	bool       CRSaveMatrixf(CRMatrixf* mat, char const* fileName)
	{
		CR_FILE_HEADER header;
		FILE* file = nullptr;

		/* validation */
		CRIA_AUTO_ASSERT(VALID_MAT(mat) && fileName, " ");
		if (!VALID_MAT(mat) || !fileName)
			return false;

		/* file header*/
		memcpy(&header.Magic[0], "CRAIMATF", 8);
		header.Cols = mat->Cols;
		header.Rows = mat->Rows;
		header.DataStart = 8 + 4 + 4 + 4;

		/* open file */
		file = fileopen(fileName, "wb");
		CRIA_AUTO_ASSERT(file, "");
		if (!file)
			return false;

		/* writing file */
		if (!WriteFileHeader(file, &header)) {
			CRIA_AUTO_ASSERT(false, "WriteFileHeader failed");
			fclose(file);
			return false;
		}
		if (!WriteData(file, &mat->Data[0], sizeof(float) * mat->Cols * mat->Rows)) {
			CRIA_AUTO_ASSERT(false, "WriteData failed");
			fclose(file);
			return false;
		}

		/* finishing */
		fclose(file);
		return true;
	}
	CRMatrixf* CRLoadMatrixf(char const* fileName)
	{
		CR_FILE_HEADER header;
		FILE* file;
		CRMatrixf* mat = 0;

		/* validation */
		CRIA_AUTO_ASSERT(fileName, "Hello, I don't exist");
		if (!fileName)
			return 0;

		/* open file */
		file = fileopen(fileName, "rb");
		CRIA_AUTO_ASSERT(file, "Hey... I'm not good.");
		if (!file)
			return 0;

		do {
			/* reading header + validation */
			if (!ReadFileHeader(file, &header)) {
				CRIA_AUTO_ASSERT(false, "ReadFileHeader failed");
				break;
			}
			CRIA_AUTO_ASSERT(memcmp(&header.Magic[0], "CRAIMATF", 8) == 0 && header.Cols != 0 && header.Rows != 0, "validation failed");
			if (memcmp(&header.Magic[0], "CRAIMATF", 8) != 0 ||
				header.Cols == 0 || header.Rows == 0)
				break;
			
			/* create matrix */
			mat = CRCreateMatrixf(header.Cols, header.Rows);
			CRIA_AUTO_ASSERT(mat, "Matrix creation failed!");
			if (!mat)
				break;

			if (fseek(file, header.DataStart, SEEK_SET)) {
				CRIA_AUTO_ASSERT(false, "fseek failed to set the cursor");
				break;
			}
			if (!ReadData(file, mat->Data, sizeof(float) * header.Cols * header.Rows)) {
				CRIA_AUTO_ASSERT(false, "ReadData failed to load the matrix data");
				break;
			}

			fclose(file);
			return mat;
		} while (false);

		fclose(file);
		if (mat)
			CRFreeMatrixf(mat);
		return 0;
	}

	bool       CRWriteMatrixf(CRMatrixf* mat, char const* fileName, uint decimals)
	{
		using namespace std;

		uint predecimals;
		uint index;
		
		/* validation */
		CRIA_AUTO_ASSERT(VALID_MAT(mat), "The matrix has to be valid!");
		CRIA_AUTO_ASSERT(fileName, "I can't save data to \"null\"");
		if (!VALID_MAT(mat) || !fileName)
			return false;

		/* init format values */
		predecimals = (uint)log10f(
			MAX(CRGetMaxValue(mat), -CRGetMinValue(mat)) /* Getting the longest value */
			) + 2 /* plus one for the sign and one because log10 returns one too short */;
		
		/* opening the file*/
		CreateContainingDir(fileName);
		ofstream file(fileName, ifstream::out | ofstream::binary);
		CRIA_AUTO_ASSERT(file.is_open(), "Failed to create/open the File[%s]", fileName);
		if (!file.is_open())
			return false;

		char* str = new char[1 + predecimals + 1 + decimals + 1 + 1];
		str[1 + predecimals + 1 + decimals + 1] = 0;
		for (index = 0; index < mat->Cols * mat->Rows; index++)
		{
			sprintf(str, "%+*.*f ", predecimals, decimals, mat->Data[index]);
			file << str;
			
			if ((index + 1) % mat->Cols == 0)
				file << std::endl;
		}
		//TODO better error test for fprintf
		delete[] str;

		file.close();
		return true;
	}
	bool       CRWriteMatrixfBmp(CRMatrixf* mat, char const* fileName)
	{
		using namespace bmp_renderer;

		CRIA_AUTO_ASSERT(VALID_MAT(mat), "The matrix is invalid");
		CRIA_AUTO_ASSERT(fileName, "null is not a valid file name, ask your OS");
		if (!VALID_MAT(mat) || !fileName)
			return false;

		Bitmap* bmp = CreateBmp(mat->Cols, mat->Cols);
		CRIA_AUTO_ASSERT(bmp, "The bitmap creation failed, report that to the creator of the BmpRenderer... fuck that's me");
		if (!bmp)
			return false;

		uint8_t color;
		uint row;
		for (uint col = 0; col < mat->Cols; col++)
		{
			for (row = 0; row < mat->Rows; row++)
			{
				color = (uint8_t)(255.0f * mat->Data[row + col * mat->Rows]);
				SetPixel(bmp, row, col, Color(color, color, color));
			}
		}
		
		/* saving the file */
		if (CreateContainingDir(fileName) && !SaveBitmap(bmp, fileName))
		{
			CRIA_AUTO_ASSERT(false, "The bitmap could not be saved sorry!!");
			DeleteBmp(bmp);
			return false;
		}
		
		DeleteBmp(bmp);
		return true;
	}

	bool       CRIsMatValid(CRMatrixf* mat)
	{
		return VALID_MAT(mat);
	}

	void       CRFillMatrixRand(CRMatrixf* mat)
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

	float      CRGetMaxValue(CRMatrixf const* mat)
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
	float      CRGetMinValue(CRMatrixf const* mat)
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
	CRMatrixf* CRClamp(CRMatrixf const* srcMat, float min, float max)
	{
		uint index;
		CRMatrixf* dstMat;

		/* validate */
		CRIA_AUTO_ASSERT(VALID_MAT(srcMat), "The matrix has to be valid.");
		CRIA_AUTO_ASSERT(min < max, "min should be less than max");
		if (!VALID_MAT(srcMat) || min > max)
			return 0;

		/* creating the matrix */
		dstMat = CRCreateMatrixf(srcMat->Cols, srcMat->Rows);
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

	CRMatrixf* CRAdd(CRMatrixf const* a, CRMatrixf const* b)
	{
		CRMatrixf* mat;
		uint index;

		/* matrix validation */
		CRIA_AUTO_ASSERT(VALID_MAT(a) || VALID_MAT(b), "Sorry is wasn't me... I hope");
		CRIA_AUTO_ASSERT(a->Cols == b->Cols || a->Rows == b->Rows, "Different dimensions yay... nooooooooooo");
		if (!VALID_MAT(a) || !VALID_MAT(b) ||
			a->Cols != b->Cols || a->Rows != b->Rows)
			return 0;

		/* matrix creation */
		mat = CRCreateMatrixf(a->Cols, a->Rows);
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
	CRMatrixf* CRSub(CRMatrixf const* a, CRMatrixf const* b)
	{
		CRMatrixf* mat;
		uint index;

		/* matrix validation */
		CRIA_AUTO_ASSERT(VALID_MAT(a) || VALID_MAT(b), "Sorry is wasn't me... I hope");
		CRIA_AUTO_ASSERT(a->Cols == b->Cols || a->Rows == b->Rows, "Different dimensions yay... nooooooooooo");
		if (!VALID_MAT(a) || !VALID_MAT(b) ||
			a->Cols != b->Cols || a->Rows != b->Rows)
			return 0;

		/* matrix creation */
		mat = CRCreateMatrixf(a->Cols, a->Rows);
		CRIA_AUTO_ASSERT(mat, "Matrix creation failed!");
		if (!mat)
			return 0;

		/* adding the numbers */
		for (index = 0; index < a->Cols * a->Rows; index++) {
			mat->Data[index] = a->Data[index] - b->Data[index];
		}

		return mat;
	}
	CRMatrixf* CRMul(CRMatrixf const* a, CRMatrixf const* b)
	{
		CRMatrixf* mat;

		/* validation */
		CRIA_AUTO_ASSERT(VALID_MAT(a) && VALID_MAT(b), "The matrices have to be valid sorry.");
		CRIA_AUTO_ASSERT(a->Rows == b->Cols, "The rows form matrix a and the columns from matrix b have to be the same.");
		if (!VALID_MAT(a) || !VALID_MAT(b) ||
			a->Rows != b->Cols)
			return 0;

		/* create the output matrix */
		mat = CRCreateMatrixf(a->Cols, b->Rows);
		CRIA_AUTO_ASSERT(mat, "The creation of the output matrix failed!")
		if (!mat)
			return 0;

		/* multiplication (not optimized)
		 * a : move along cols (starts at result row)
		 * b : move along rows (starts at result column)
		 */
		uint calCount = a->Cols;
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

	CRMatrixf* CRMul(CRMatrixf const* a, float b)
	{
		CRMatrixf* mat;
		uint index;

		/* matrix validation */
		CRIA_AUTO_ASSERT(VALID_MAT(a), "Sorry is wasn't me... I hope");
		if (!VALID_MAT(a))
			return 0;

		/* matrix creation */
		mat = CRCreateMatrixf(a->Cols, a->Rows);
		CRIA_AUTO_ASSERT(mat, "Matrix creation failed!");
		if (!mat)
			return 0;

		/* adding the numbers */
		for (index = 0; index < a->Cols * a->Rows; index++) {
			mat->Data[index] = a->Data[index] * b;
		}

		return mat;
	}
	CRMatrixf* CRDiv(CRMatrixf const* a, float b)
	{
		CRIA_AUTO_ASSERT(b == 0, "Devision by 0 is undefined behavior");
		if (b == 0)
			return 0;

		return CRMul(a, 1 / b);
	}
}
 