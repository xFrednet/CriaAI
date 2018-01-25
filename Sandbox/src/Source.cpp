#include <Cria.hpp>

using namespace std;
using namespace cria_ai;

int main(int argc, char* argv)
{
	cout << "Hello world" << endl;
	
	CRMatrixf* mat = LoadMatrixf("Matrix.mat");
	if (!mat)
	{
		mat = CreateMatrixf(4, 4);
		FillMatrixRand(mat);
	}

	SaveMatrixf(mat, "Matrix.mat");
	WriteMatrixf(mat, "Hello.txt");

	FreeMatrixf(mat);

	cin.get();
	return 0;
}