#pragma once

#include "../Layer.h"

namespace cria_ai
{

	class CRBmpInLayer : public CRLayer
	{
	public:
		typedef struct CR_BMP_PROCESSOR_NODE_ {
			
			CR_BMP_PROCESSOR_NODE_* Next;
			CR_FLOAT_BITMAP* (*m_BitmapProcessor) (CR_FLOAT_BITMAP* bmp);

		} CR_BMP_PROCESSOR_NODE;
	protected:
		uint32_t m_BitmapWidth;
		uint32_t m_BitmapHeight;

		CR_BMP_PROCESSOR_NODE* m_BmpProcessorsStack;

	public:

	};

}
