#pragma once

#include "../Types.hpp"

#include "Vec2.h"

namespace cria_ai
{

	typedef struct CR_RECT_ {

		union
		{
			struct
			{
				int X;
				int Y;
			};
			CR_VEC2I Pos;
		};
		union
		{
			struct
			{
				uint  Width;
				uint  Height;
			};
			CR_VEC2U Size;
		};

		CR_RECT_(int x, int y, uint width, uint height)
			: X(x), Y(y), Width(width), Height(height)
		{}
		CR_RECT_()
			: CR_RECT_(0, 0, 0, 0)
		{}

		bool contains(const CR_VEC2I& pos) const
		{
			return
				(pos.X >= this->X && pos.X < this->X + ((int)this->Width)) &&
				(pos.Y >= this->Y && pos.Y < this->Y + ((int)this->Height));
		}

	} CR_RECT;

}