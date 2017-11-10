#include <Renderer.hpp>

#include <string>

#define MAX(x, y)                      (((x) > (y)) ? (x) : (y)) 
#define MIN(x, y)                      (((x) < (y)) ? (x) : (y)) 

namespace bmp_renderer {
	Renderer::Renderer(unsigned width, unsigned height)
		: Renderer(new Bitmap__(width, height), true)
	{
	}
	Renderer::Renderer(Bitmap renderTarget, bool autoDeleteTarget)
		: m_RenderTarget(renderTarget),
		m_AutoDeleteTarget(autoDeleteTarget)
	{
	}

	Renderer::~Renderer()
	{
		if (m_AutoDeleteTarget)
		{
			if (m_RenderTarget)
			{
				delete m_RenderTarget;
				m_RenderTarget = nullptr;
			}
		}
	}

	Bitmap Renderer::getRenderTarget()
	{
		return m_RenderTarget;
	}
	void Renderer::setRenderTarget(Bitmap renderTarget, bool autoDeleteTarget)
	{
		if (m_AutoDeleteTarget) {
			if (m_RenderTarget) {
				delete m_RenderTarget;
				m_RenderTarget = nullptr;
			}
		}

		m_RenderTarget = renderTarget;
		m_AutoDeleteTarget = autoDeleteTarget;
	}

	void Renderer::clearTarget(Color color)
	{
		if (!m_RenderTarget) return;
		for (int pixelNr = 0; pixelNr < m_RenderTarget->WIDTH * m_RenderTarget->HEIGHT; pixelNr++) 
		{
			m_RenderTarget->Data[pixelNr] = color;
		}
	}

	void Renderer::setPixel(const int& x, const int& y, const Color& color)
	{
		if (x < 0 || x >= m_RenderTarget->WIDTH || y < 0 || y >= m_RenderTarget->HEIGHT)
			return;

		setPixelWithoutValidation(x, y, color);
	}


	/* ======================================*/
	// = lines =
	/* ======================================*/
	void Renderer::drawLine(int startX, int startY, int endX, int endY, Color color)
	{
		if (!m_RenderTarget) return;
		if ((startX < 0 && endX < 0) ||
			(startX >= m_RenderTarget->WIDTH && endX >= m_RenderTarget->WIDTH) ||
			(startY < 0 && endY < 0) ||
			(startY >= m_RenderTarget->HEIGHT && endY >= m_RenderTarget->HEIGHT))
			return;

		//Line : a*x + m
		float horizontalDiff = (float)(endX - startX);
		float verticalDiff = (float)(endY - startY);
		bool moveAlongXAxis = abs(horizontalDiff) > abs(verticalDiff);

		// the start has to be lower than the end along the selected axis 
		if ((moveAlongXAxis && endX < startX) || //if alongXAxis and startX is higher 
			(!moveAlongXAxis && endY < startY))  //if alongYAxis and startY is higher
		{
			int oldStartX = startX;
			startX = endX;
			endX = oldStartX;
			
			int oldStartY = startY;
			startY = endY;
			endY = oldStartY;

			verticalDiff *= -1;
			horizontalDiff *= -1;
		}

		if (moveAlongXAxis)
		{
			float incrementer = verticalDiff / horizontalDiff;
			float y = (float)startY + 0.5f;
			for (int x = startX; x < endX; x++, y += incrementer)
			{
				setPixel(x, (int)floor(y), color);
			}
		} else
		{
			float incrementer = horizontalDiff / verticalDiff;
			float x = (float)startX + 0.5f;
			for (int y = startY; y < endY; y++, x += incrementer) {
				setPixel((int)floor(x), y, color);
			}
		}
	}
	void Renderer::drawHorizontalLine(int startX, int drawY, int length, Color color)
	{
		if (!m_RenderTarget) return;
		if (drawY < 0 || drawY >= m_RenderTarget->HEIGHT)
			return;

		if (length < 0)
		{
			length *= -1;
			startX -= length;
		}
		if (startX < 0)
		{
			length += startX;
			startX = 0;
		}
		if (startX + length >= m_RenderTarget->WIDTH)
			length = m_RenderTarget->WIDTH - startX;
		if (length < 0)
			return;

		int drawX;
		for (int xOffset = 0; xOffset < length; xOffset++)
		{
			drawX = startX + xOffset;
			setPixelWithoutValidation(drawX, drawY, color);
		}
	}
	void Renderer::drawVerticalLine(int drawX, int startY, int length, Color color)
	{
		if (!m_RenderTarget) return;
		if (drawX < 0 || drawX >= m_RenderTarget->WIDTH)
			return;

		if (length < 0) {
			length *= -1;
			startY -= length;
		}
		if (startY < 0) {
			length += startY;
			startY = 0;
		}
		if (startY + length >= m_RenderTarget->HEIGHT)
			length = m_RenderTarget->HEIGHT - startY;
		if (length < 0)
			return;

		int drawY;
		for (int yOffset = 0; yOffset < length; yOffset++) {
			drawY = startY + yOffset;
			setPixelWithoutValidation(drawX, drawY, color);
		}
	}

	/* ======================================*/
	// = shapes =
	/* ======================================*/
	void Renderer::drawRectangle(int x0, int y0, int x1, int y1, Color color)
	{
		if (!m_RenderTarget) return;
		
		int width = x1 - x0;
		int height = y1 - y0;
		
		if (width < 0)
		{
			width *= -1;
			int oldX0 = x0;
			x0 = x1;
			x1 = x0;
		} // => x0 < x1
		if (height < 0) {
			height *= -1;
			int oldY0 = y0;
			y0 = y1;
			y1 = oldY0;
		} // => y0 < y1

		if (x1 < 0 || x0 >m_RenderTarget->WIDTH || y1 < 0 || y0 >= m_RenderTarget->HEIGHT)
			return;

		// +---1---+
		// |       |
		// 3       4
		// |       |
		// +---2---+
		drawHorizontalLine(x0, y0, width, color);
		drawHorizontalLine(x0, y1, width + 1, color);

		drawVerticalLine(x0, y0, height, color);
		drawVerticalLine(x1, y0, height, color);
	}
	void Renderer::drawFilledRectangle(int x0, int y0, int x1, int y1, Color color)
	{
		if (!m_RenderTarget) return;
		
		if (x0 > x1)
		{
			int oldX0 = x0;
			x0 = x1;
			x1 = oldX0;
		}
		if (y0 > y1)
		{
			int oldY0 = y0;
			y0 = y1;
			y1 = oldY0;
		}
		
		if (x1 < 0 || x0 >m_RenderTarget->WIDTH || y1 < 0 || y0 >= m_RenderTarget->HEIGHT)
			return;
		
		if (x0 < 0)
			x0 = 0;
		if (x1 >= m_RenderTarget->WIDTH)
			x1 = m_RenderTarget->WIDTH - 1;

		if (y0 < 0)
			y0 = 0;
		if (y1 >= m_RenderTarget->HEIGHT)
			y1 = m_RenderTarget->HEIGHT - 1;

		int drawX;
		for (; y0 <= y1; y0++)
		{
			for (drawX = x0; drawX <= x1; drawX++)
			{
				setPixelWithoutValidation(drawX, y0, color);
			}
		}
	}

	// The circle will be drawn along both axis. The used axis is always the one where the other 
	// axis one has only one matching value. The calculations will be done for one quadrant and applied
	// in the others
	//
	// In the following example an "x"s means that it would be drawn along the horizontal-axis
	// and the "y"s mean that the pixel would be drawn along the vertical-axis.
	//   
	// +-------------x
	// | .           x
	// |   .        x
	// |     .    xx
	// |       . x
	// |      yy
	// |   yyy
	// yyyy
	// 
	// The magic join point is at 3/4 at the radius.
	//
	// The x value is defined by using cos()
	// The y value is defined by using sin()
	// -> y = sin(arccos(x))
	// -> x = cos(arcsin(y))
	//
	// BTW: no I didn't find a better way than copping the loop code to every circle method -.-.(#define [...] looked terrible)
	void Renderer::drawCircle(int centerX, int centerY, unsigned radius, Color color)
	{
		if (!m_RenderTarget) return;
		if ((centerX + (int)radius) < 0 || centerX - (int)radius >= m_RenderTarget->WIDTH ||
			(centerY + (int)radius) < 0 || centerY - (int)radius >= m_RenderTarget->HEIGHT)
			return;

		//this is art
		float fRadius = (float)radius;
		int mainAxisOffset = 0; // offset along the "moveAxis" from origin
		int sideAxisOffset;
		for (float currentUnit = 0, oneRadiusUnit = (1.0f / fRadius); currentUnit <= 0.75f; currentUnit += oneRadiusUnit, mainAxisOffset++)
		{
			//vertical axis
			sideAxisOffset = (int)roundf((fRadius * sinf(acosf(currentUnit)))); // is positive

			setPixel(centerX - mainAxisOffset, centerY + sideAxisOffset, color);
			setPixel(centerX + mainAxisOffset, centerY + sideAxisOffset, color);
			setPixel(centerX + mainAxisOffset, centerY - sideAxisOffset, color);
			setPixel(centerX - mainAxisOffset, centerY - sideAxisOffset, color);
			//TODO the center is overdrawn multiple times this my case issues with transparent colors

			setPixel(centerX - sideAxisOffset, centerY + mainAxisOffset, color);
			setPixel(centerX + sideAxisOffset, centerY + mainAxisOffset, color);
			setPixel(centerX + sideAxisOffset, centerY - mainAxisOffset, color);
			setPixel(centerX - sideAxisOffset, centerY - mainAxisOffset, color);
		}
	}

	void Renderer::drawFilledCircle(int centerX, int centerY, unsigned radius, Color color)
	{
		if (!m_RenderTarget) return;
		if ((centerX + (int)radius) < 0 || centerX - (int)radius >= m_RenderTarget->WIDTH ||
			(centerY + (int)radius) < 0 || centerY - (int)radius >= m_RenderTarget->HEIGHT)
			return;

		//this is art
		float fRadius = (float)radius;
		int mainAxisOffset = 0; // offset along the "moveAxis" from origin
		int sideAxisOffset;
		for (float currentUnit = 0, oneRadiusUnit = (1.0f / fRadius); currentUnit <= 0.75f; currentUnit += oneRadiusUnit, mainAxisOffset++)
		{
			//vertical axis
			sideAxisOffset = (int)(fRadius * sinf(acosf(currentUnit))); // is positive

			drawVerticalLine(centerX - mainAxisOffset, centerY - sideAxisOffset, sideAxisOffset * 2, color);
			drawVerticalLine(centerX + mainAxisOffset, centerY - sideAxisOffset, sideAxisOffset * 2, color);
			//TODO the center is overdrawn multiple times this my case issues with transparent colors

			drawHorizontalLine(centerX - sideAxisOffset, centerY - mainAxisOffset, sideAxisOffset * 2, color);
			drawHorizontalLine(centerX - sideAxisOffset, centerY + mainAxisOffset, sideAxisOffset * 2, color);
		}
	}
}
