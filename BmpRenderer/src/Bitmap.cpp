#include "Bitmap.hpp"

#include <string>

namespace bmp_renderer {

	Color::Color(int argb)
		: ARGB(argb)
	{
		
	}
	Color::Color(color_channel r, color_channel g, color_channel b, color_channel a)
		: B(b), G(g), R(r), A(a)
	{
	}

}

namespace bmp_renderer {

	Bitmap__::Bitmap__(unsigned width, unsigned height)
		: WIDTH((int)width), HEIGHT((int)height)
	{
		Data = new Color[width * height];
		memset(Data, BMP_RENDERER_BITMAP_BASE_COLOR, width * height * sizeof(Color));
	}
	Bitmap__::~Bitmap__()
	{
		if (Data)
			delete[] Data;
	}

	Color Bitmap__::getPixel(int xPixel, int yPixel) const
	{
		if (xPixel < 0 || xPixel >= WIDTH || yPixel < 0 || yPixel >= HEIGHT)
			return Color(0, 0, 0, 0);

		return Data[xPixel + yPixel * WIDTH];
	}
	void Bitmap__::setPixel(int xPixel, int yPixel, Color color)
	{
		if (xPixel >= 0 && xPixel < WIDTH && yPixel >= 0 && yPixel < HEIGHT)
			Data[xPixel + yPixel * WIDTH] = color;
	}

	Bitmap__* Bitmap__::createSubBitMap(int xPixel, int yPixel, unsigned width, unsigned height) const
	{
		if (xPixel < 0 || xPixel >= WIDTH || yPixel < 0 || yPixel >= HEIGHT)
			return nullptr;

		if ((xPixel + (int)width) >= WIDTH)
			width = (int)(WIDTH - xPixel - 1);
		if ((yPixel + (int)height) >= HEIGHT)
			height = (int)(HEIGHT - yPixel - 1);

		Bitmap__* copyBitmap = new Bitmap__(width, height);
		for (int yLocation = 0; yLocation < copyBitmap->HEIGHT; yLocation++)
		{
			int yCopySrc = yPixel + yLocation;
			memcpy(&copyBitmap->Data[yLocation * copyBitmap->WIDTH], &this->Data[xPixel + yCopySrc * WIDTH], copyBitmap->WIDTH * sizeof(Color));
		}

		return copyBitmap;
	}
}
