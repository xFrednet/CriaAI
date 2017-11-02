#include "Bitmap.hpp"

#include <string>

namespace bmp_renderer {

	Color::Color(unsigned rgba)
	{
		//the input order of rgba is 0x[a][b][g][r] this means that the channels have the same order in reverse
		color_channel* channals = (color_channel*)&rgba;

		R = channals[2];
		G = channals[1];
		B = channals[0];
		A = channals[3];
	}
	Color::Color(color_channel r, color_channel g, color_channel b, color_channel a)
		: R(r), G(g), B(b), A(a)
	{
	}
}

namespace bmp_renderer {

	Bitmap__::Bitmap__(unsigned width, unsigned height)
		: WIDTH(width), HEIGHT(height)
	{
		Data = new Color[width * height];
		memset(Data, BMP_RENDERER_BITMAP_BASE_COLOR, width * height * sizeof(Color));
	}
	Bitmap__::~Bitmap__()
	{
		if (Data)
			delete[] Data;
	}

	Color Bitmap__::getPixel(unsigned xPixel, unsigned yPixel) const
	{
		if (xPixel >= WIDTH || yPixel >= HEIGHT)
			return Color(0, 0, 0, 0);

		return Data[xPixel + yPixel * WIDTH];
	}
	void Bitmap__::setPixel(unsigned xPixel, unsigned yPixel, Color color)
	{
		if (xPixel < WIDTH && yPixel < HEIGHT)
			Data[xPixel + yPixel * WIDTH] = color;
	}

	Bitmap__* Bitmap__::createSubBitMap(unsigned xPixel, unsigned yPixel, unsigned width, unsigned height) const
	{
		if (xPixel >= WIDTH || yPixel >= HEIGHT)
			return nullptr;

		if ((xPixel + width) >= WIDTH)
			width = WIDTH - xPixel - 1;
		if ((yPixel + height) >= HEIGHT)
			height = HEIGHT - yPixel - 1;

		Bitmap__* copyBitmap = new Bitmap__(width, height);
		for (unsigned yLocation = 0; yLocation < height; yLocation++)
		{
			unsigned yCopySrc = yPixel + yLocation;
			memcpy(&copyBitmap->Data[yLocation * copyBitmap->WIDTH], &this->Data[xPixel + yCopySrc * WIDTH], copyBitmap->WIDTH * sizeof(Color));
		}

		return copyBitmap;
	}
}
