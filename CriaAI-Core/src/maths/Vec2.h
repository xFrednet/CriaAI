#pragma once

#include "../Macros.hpp"
#include "../Types.hpp"

#include <math.h>

namespace cria_ai
{
	template <typename _VecT>
	struct CR_VEC2_ {

		typedef CR_VEC2_<_VecT> CR_VEC2;

		_VecT X;
		_VecT Y;

		CR_VEC2_(const _VecT& x, const _VecT& y)
			: X(x), Y(y)
		{}
		CR_VEC2_()
			: CR_VEC2_((_VecT)0, (_VecT)0)
		{}

		inline CR_VEC2* getThis()
		{
			return ((CR_VEC2*)this);
		}

		/*
		 * Utils
		 */
		inline void  scale(const _VecT& scale)
		{
			X *= scale;
			Y *= scale;
		}
		inline void  clamp(const _VecT& min, const _VecT& max)
		{
			CR_CLAMP_VALUE(X, min, max);
			CR_CLAMP_VALUE(Y, min, max);
		}
		inline _VecT getLengthSq() const
		{
			return X * X + Y * Y;
		}
		inline float getLength() const
		{
			return std::sqrtf(getLengthSq());
		}

		/*
		 * Operators
		 */
		inline bool operator==(const CR_VEC2& other) const
		{
			return (X == other.X) &&
				(Y == other.Y);
		}
		inline bool operator!=(const CR_VEC2& other) const
		{
			return (X != other.X) &&
				(Y != other.Y);
		}

		inline CR_VEC2 operator+(const CR_VEC2& other) const
		{
			return CR_VEC2(X + other.X, Y + other.Y);
		}
		inline CR_VEC2 operator-(const CR_VEC2& other) const
		{
			return CR_VEC2(X - other.X, Y - other.Y);
		}
		inline CR_VEC2 operator*(const CR_VEC2& other) const
		{
			return CR_VEC2(X * other.X, Y * other.Y);
		}
		inline CR_VEC2 operator/(const CR_VEC2& other) const
		{
			return CR_VEC2(
				((other.X != 0) ? (X / other.X) : 0),
				((other.Y != 0) ? (Y / other.Y) : 0));
		}

		inline CR_VEC2 operator*(float value) const
		{
			return CR_VEC2((_VecT)(X * value), (_VecT)(Y * value));
		}
		inline CR_VEC2 operator/(float value) const
		{
			if (value == 0)
				return CR_VEC2(0, 0);

			return CR_VEC2((_VecT)(X / value), (_VecT)(Y / value));
		}

		inline CR_VEC2& operator+=(const CR_VEC2& other)
		{
			*getThis() = *getThis() + other;
			return *getThis();
		}
		inline CR_VEC2& operator-=(const CR_VEC2& other)
		{
			*getThis() = *getThis() - other;
			return *getThis();
		}
		inline CR_VEC2& operator*=(const CR_VEC2& other)
		{
			*getThis() = *getThis() * other;
			return *getThis();
		}
		inline CR_VEC2& operator/=(const CR_VEC2& other)
		{
			*getThis() = *getThis() / other;
			return *getThis();
		}
		
		inline CR_VEC2& operator*=(float value)
		{
			*getThis() = *getThis() * value;
			return *getThis();
		}
		inline CR_VEC2& operator/=(float value)
		{
			*getThis() = *getThis() / value;
			return *getThis();
		}
	};

	typedef CR_VEC2_<int> CR_VEC2I;
	typedef CR_VEC2_<uint> CR_VEC2U;
	typedef CR_VEC2_<float> CR_VEC2F;
}
