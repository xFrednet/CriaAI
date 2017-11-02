#pragma once

#define BMP_RENDERER_BITMAP_BASE_COLOR 0xffff00ff

namespace bmp_renderer
{
	typedef unsigned char color_channel;

	struct alignas(32) Color
	{
		color_channel B;
		color_channel G;
		color_channel R;
		color_channel A;

		Color(unsigned int rgba = 0xff00000);
		Color(color_channel r, color_channel g, color_channel b, color_channel a = 0xff);
	};

	struct Bitmap__;
	typedef Bitmap__* Bitmap;

	struct Bitmap__
	{
		const unsigned WIDTH;
		const unsigned HEIGHT;
		
		Color* Data;

		Bitmap__(unsigned width, unsigned height);
		~Bitmap__();

		Color getPixel(unsigned xPixel, unsigned yPixel) const;
		void setPixel(unsigned xPixel, unsigned yPixel, Color color);

		Bitmap__* createSubBitMap(unsigned xPixel, unsigned yPixel, unsigned width, unsigned height) const;
	};

}