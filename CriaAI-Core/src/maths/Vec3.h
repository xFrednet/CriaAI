#pragma once

#include "../Macros.hpp"
#include "../Types.hpp"

#include "Vec2.h"

#include <math.h>

namespace cria_ai {
	
	template <typename _VecT>
	struct CR_VEC3_
	{
		typedef CR_VEC3_<_VecT> CR_VEC3;

		_VecT X;
		_VecT Y;
		_VecT Z;

		/*
		 * Constructors
		 */
		CR_VEC3_(_VecT x, _VecT y, _VecT z)
			: X(x), Y(y), Z(z)
		{}
		CR_VEC3_(CR_VEC2_<_VecT> xy, _VecT z = 0)
			: CR_VEC3_(xy.X, xy.Y, z)
		{}
		CR_VEC3_()
			: CR_VEC3_(0, 0, 0)
		{}

		inline CR_VEC3* getThis()
		{
			return ((CR_VEC3*)this);
		}

		/*
		 * Utilities
		 */
		inline CR_VEC3 cross(const CR_VEC3& other) const
		{
			return CR_VEC3(
				this->Y * other.Z - this->Z * other.Y,
				this->Z * other.X - this->X * other.Z,
				this->X * other.Y - this->Y * other.X);
		}
		inline _VecT   getLengthSq() const
		{
			return X * X + Y * Y + Z * Z;
		}
		inline float   getLength() const
		{
			return sqrtf(getLengthSq());
		}
		inline void    scale(_VecT scale)
		{
			X *= scale;
			Y *= scale;
			Z *= scale;
		}
		inline void    clamp(_VecT min, _VecT max)
		{
			CR_CLAMP_VALUE(X, min, max);
			CR_CLAMP_VALUE(Y, min, max);
			CR_CLAMP_VALUE(Z, min, max);
		}
		inline void    normalize()
		{
			float len = getLength();
			if (len == 0)
				return;

			X = (_VecT)(X / len);
			Y = (_VecT)(Y / len);
			Z = (_VecT)(Z / len);
		}

		/*
		 * Operators
		 */ 
		inline bool    operator==(const CR_VEC3& other) const
		{
			return
				(X == other.X) &&
				(Y == other.Y) &&
				(Z == other.Z);
		}
		inline bool    operator!=(const CR_VEC3& other) const
		{
			return
				(X != other.X) &&
				(Y != other.Y) &&
				(Z != other.Z);
		}
		
		inline CR_VEC3 operator+(const CR_VEC3& other) const
		{
			return CR_VEC3(
				X + other.X, 
				Y + other.Y, 
				Z + other.Z);
		}
		inline CR_VEC3 operator-(const CR_VEC3& other) const
		{
			return CR_VEC3(
				X - other.X,
				Y - other.Y,
				Z - other.Z);
		}
		inline CR_VEC3 operator*(const CR_VEC3& other) const
		{
			return CR_VEC3(
				X * other.X,
				Y * other.Y,
				Z * other.Z);
		}
		inline CR_VEC3 operator/(const CR_VEC3& other) const
		{
			return CR_VEC3(
				((other.X != 0) ? (X / other.X) : 0),
				((other.Y != 0) ? (Y / other.Y) : 0),
				((other.Z != 0) ? (Z / other.Z) : 0));
		}

		inline CR_VEC3 operator*(float value) const
		{
			return CR_VEC3((_VecT)(X * value), (_VecT)(Y * value));
		}
		inline CR_VEC3 operator/(float value) const
		{
			if (value == 0)
				return CR_VEC3(0, 0, 0);

			return CR_VEC3((_VecT)(X / value), (_VecT)(Y / value));
		}

		inline CR_VEC3& operator+=(const CR_VEC3& other) const
		{
			*getThis() = *getThis() + other;
			return *getThis();
		}
		inline CR_VEC3& operator-=(const CR_VEC3& other) const
		{
			*getThis() = *getThis() - other;
			return *getThis();
		}
		inline CR_VEC3& operator*=(const CR_VEC3& other) const
		{
			*getThis() = *getThis() * other;
			return *getThis();
		}
		inline CR_VEC3& operator/=(const CR_VEC3& other) const
		{
			*getThis() = *getThis() / other;
			return *getThis();
		}

		inline CR_VEC3& operator*(float value)
		{
			*getThis() = *getThis() * value;
			return *getThis();
		}
		inline CR_VEC3& operator/(float value)
		{
			*getThis() = *getThis() / value;
			return *getThis();
		}
	};

	typedef CR_VEC3_<float>            CR_VEC3F;
	typedef CR_VEC3_<int>              CR_VEC3I;
	typedef CR_VEC3_<uint>             CR_VEC3U;
}
