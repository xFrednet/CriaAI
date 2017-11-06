#pragma once

#include <Bitmap.hpp>

namespace bmp_renderer {
	
	class Renderer
	{
	private:
		Bitmap m_RenderTarget;
		bool m_AutoDeleteTarget;

		inline void setPixelWithoutValidation(const int& x, const int& y, const Color& color)
		{
			//TODO add alpha blending out = alpha * new + (1 - alpha) * old
			m_RenderTarget->Data[x + y * m_RenderTarget->WIDTH] = color;
		}

	public:
		Renderer(unsigned width, unsigned height);
		Renderer(Bitmap renderTarget, bool autoDeleteTarget = false);
		~Renderer();

		Bitmap getRenderTarget();
		void setRenderTarget(Bitmap renderTarget, bool autoDeleteTarget = false);
		void clearTarget(Color color = Color(0xffffffff));

		void setPixel(const int& x, const int& y, const Color& color);
		
		/* ======================================*/
		// = lines =
		/* ======================================*/
		void drawLine(int startX, int startY, int endX, int endY, Color color);
		void drawHorizontalLine(int startX, int startY, int length, Color color);
		void drawVerticalLine(int startX, int startY, int length, Color color);

		/* ======================================*/
		// = shapes =
		/* ======================================*/
		void drawRectangle(int x0, int y0, int x1, int y1, Color color);
		void drawFilledRectangle(int x0, int y0, int x1, int y1, Color color);

		void drawCircle(int centerX, int centerY, unsigned radius, Color color);
		void drawFilledCircle(int centerX, int centerY, unsigned radius, Color color);

		/* ======================================*/
		// = bitmap =
		/* ======================================*/
		void drawBitmap(Bitmap bitmap, int destX, int destY);
		void drawBitmap(Bitmap bitmap, int destX1, int destY1, int destX2, int destY2);
		void drawBitmap(Bitmap bitmap, int destX1, int destY1, int destX2, int destY2, int srcX1, int srcY1, int srcX2, int srcY2);
	};

}
