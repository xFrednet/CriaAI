#include <BmpRenderer.hpp>

#include <iostream>

using namespace std;
using namespace bmp_renderer;

int main()
{
	Window* w1 = Window::CreateInstance("Color", 700, 500, WINDOW_ON_EXIT_DESTROY);

	Renderer r(700, 500); 

	
	r.clearTarget(Color(0xffffffff));
	r.drawFilledRectangle(100, 100, 600, 400, Color(0xff, 0, 0, 0xff / 4));
	r.drawFilledRectangle(100, 100, 600, 400, Color(0xff, 0, 0, 0xff / 4));
	r.drawFilledRectangle(100, 100, 600, 400, Color(0xff, 0, 0, 0xff / 4));
	r.drawFilledRectangle(100, 100, 600, 400, Color(0xff, 0, 0, 0xff / 4));

	
	for (int y = 0; y <= 500; y += 50)
		for (int x = 0; x <= 700; x += 50)
			r.drawCircle(x, y, 100, Color(0x22000000));

	w1->loadBitmap(r.getRenderTarget());

	while (w1->update()) {
	}
	
	delete w1;
	return 0;
}