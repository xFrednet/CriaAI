#include <Cria.hpp>
#include <fstream>
#include "Dependencies/BmpRenderer/src/Bitmap.hpp"

using namespace std;
using namespace cria_ai;
using namespace bmp_renderer;

void MatrixTest()
{
	CRMatrixf* a = LoadMatrixf("mat/a.mat");
	CRMatrixf* b = LoadMatrixf("mat/b.mat");
	if (!a)
	{
		a = CreateMatrixf(2, 2);
		FillMatrixRand(a);
		a->Data[0] = 1.0f;
		a->Data[1] = 2.0f;
		a->Data[2] = 3.0f;
		a->Data[3] = 4.0f;
		SaveMatrixf(a, "mat/a.mat");
		WriteMatrixf(a, "mat/a.txt");
		WriteMatrixfBmp(a, "mat/a.bmp");
	}
	if (!b) {
		b = CreateMatrixf(2, 2);
		FillMatrixRand(b);
		b->Data[0] = 0.0f;
		b->Data[1] = 1.0f;
		b->Data[2] = 0.0f;
		b->Data[3] = 0.0f;
		SaveMatrixf(b, "mat/b.mat");
		WriteMatrixf(b, "mat/b.txt");
		WriteMatrixfBmp(b, "mat/b.bmp");
	}

	CRMatrixf* c = Mul(a, b);
	SaveMatrixf(c, "mat/c.mat");
	WriteMatrixf(c, "mat/c.txt");
	WriteMatrixfBmp(c, "mat/c.bmp");

	FreeMatrixf(a);
	FreeMatrixf(b);
	FreeMatrixf(c);
}

int main(int argc, char* argv)
{
	cout << "Hello world" << endl;

	CR_FLOAT_BITMAP* bmp = LoadFBmp("bmptest/test.bmp");

	CR_FLOAT_BITMAP* poolBmp2 = PoolBitmap(bmp, 2);
	CR_FLOAT_BITMAP* poolBmp3 = PoolBitmap(bmp, 3);
	CR_FLOAT_BITMAP* poolBmp9 = PoolBitmap(bmp, 9);

	SaveBitmap(bmp, "bmptest/test2.bmp");
	SaveBitmap(poolBmp2, "bmptest/pool2.bmp");
	SaveBitmap(poolBmp3, "bmptest/pool3.bmp");
	SaveBitmap(poolBmp9, "bmptest/pool9.bmp");

	DeleteFBmp(bmp);
	DeleteFBmp(poolBmp2);
	DeleteFBmp(poolBmp3);
	DeleteFBmp(poolBmp9);

	cin.get();
	return 0;
}