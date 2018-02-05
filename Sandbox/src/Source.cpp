#include <Cria.hpp>
#include <fstream>
#include "Dependencies/BmpRenderer/src/Bitmap.hpp"

using namespace std;
using namespace cria_ai;
using namespace bmp_renderer;

Bitmap* pro1 (Bitmap* bmp)
{
	return bmp;
}
void test()
{
	Bitmap* (**decoder)(Bitmap* bitmap);

	Bitmap* src;

	decoder[0] = pro1;
	Bitmap* bmp = decoder[0](src);
}

int main(int argc, char* argv)
{
	cout << "Hello world" << endl;

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

	cin.get();
	return 0;
}