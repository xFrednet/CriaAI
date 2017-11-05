#pragma once

#define BMP_RENDERER_BITMAP_BASE_COLOR 0xffff00ff

namespace bmp_renderer
{
	typedef unsigned char color_channel;

	struct Color
	{
		union
		{
			int ARGB;
			struct
			{
				color_channel B;
				color_channel G;
				color_channel R;
				color_channel A;
			};
		};

		Color(int argb = 0xff00000);
		Color(color_channel r, color_channel g, color_channel b, color_channel a = 0xff);
	};

	struct Bitmap__;
	typedef Bitmap__* Bitmap;

	struct Bitmap__
	{
		const int WIDTH;
		const int HEIGHT;
		
		Color* Data;

		Bitmap__(unsigned width, unsigned height);
		~Bitmap__();

		Color getPixel(int xPixel, int yPixel) const;
		void setPixel(int xPixel, int yPixel, Color color);

		Bitmap__* createSubBitMap(int xPixel, int yPixel, unsigned width, unsigned height) const;
	};

}