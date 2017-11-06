#include <BmpRenderer.hpp>

#include <iostream>

using namespace std;
using namespace bmp_renderer;

int main()
{
	Window* w1 = Window::CreateInstance("Color", 700, 500, WINDOW_ON_EXIT_DESTROY);

	Renderer r(700, 500); 

	
	r.drawRectangle(1, 1, 698, 498, Color(0xff, 0, 0));
	r.drawRectangle(0, 0, 699, 499, Color(0, 0, 0xff));
	
	for (int radius = 0; radius < 100; radius += 10)
		r.drawCircle(200, 200, radius, Color(0xff000000));

	w1->loadBitmap(r.getRenderTarget());

	while (w1->update()) {
	}
	
	delete w1;
	return 0;
}