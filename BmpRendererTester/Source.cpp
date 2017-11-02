#include <BmpRenderer.hpp>

#include <iostream>

using namespace std;
using namespace bmp_renderer;

int main()
{
	Window* w1 = Window::CreateInstance("Color", 700, 500, WINDOW_ON_EXIT_DESTROY);

	Bitmap bmp = new Bitmap__(700, 500);
	memset(bmp->Data, 0xff, 700 * 500 * 4);

	for (unsigned index = 0; index < 700 * 500; index += 5)
	{
		bmp->Data[index].G = 0;
		bmp->Data[index].B = 0;
	}

	w1->loadBitmap(bmp);

	while (w1->update()) {
	}

	return 0;
}