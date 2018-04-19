#pragma once
#include "../Macros.hpp"
#include <math.h>

namespace cria_ai
{
	template <typename VecT>
	struct CR_VEC2_ {

		VecT X;
		VecT Y;

		CR_VEC2_(const VecT& x, const VecT& y)
			: X(x), Y(y)
		{}
		CR_VEC2_()
			: CR_VEC2_((VecT)0, (VecT)0)
		{}

		inline CR_VEC2_<VecT>* getThis()
		{
			return ((CR_VEC2_<VecT>*)this);
		}

		/*
		 * Utils
		 */
		inline void  scale(const VecT& scale)
		{
			X *= scale;
			Y *= scale;
		}
		inline void  clamp(const VecT& min, const VecT& max)
		{
			CR_CLAMP_VALUE(X, min, max);
			CR_CLAMP_VALUE(Y, min, max);
		}
		inline VecT getLengthSq() const
		{
			return X * X + Y * Y;
		}
		inline VecT getLength() const
		{
			return std::sqrt(getLengthSq());
		}

		/*
		 * Operators
		 */
		inline bool operator==(const CR_VEC2_<VecT>& other) const
		{
			return (X == other.X) &&
				(Y == other.Y);
		}
		inline bool operator!=(const CR_VEC2_<VecT>& other) const
		{
			return (X != other.X) &&
				(Y != other.Y);
		}

		inline CR_VEC2_<VecT> operator+(const CR_VEC2_<VecT>& other) const
		{
			return CR_VEC2_<VecT>(X + other.X, Y + other.Y);
		}
		inline CR_VEC2_<VecT> operator-(const CR_VEC2_<VecT>& other) const
		{
			return CR_VEC2_<VecT>(X - other.X, Y - other.Y);
		}
		inline CR_VEC2_<VecT> operator*(const CR_VEC2_<VecT>& other) const
		{
			return CR_VEC2_<VecT>(X * other.X, Y * other.Y);
		}
		inline CR_VEC2_<VecT> operator/(const CR_VEC2_<VecT>& other) const
		{
			return CR_VEC2_<VecT>(
				((other.X != 0) ? (X / other.X) : 0),
				((other.Y != 0) ? (Y / other.Y) : 0));
		}

		inline CR_VEC2_<VecT> operator*(const VecT& value) const
		{
			return CR_VEC2_<VecT>(X * value, Y * value);
		}
		inline CR_VEC2_<VecT> operator/(const VecT& value) const
		{
			if (value == 0)
				return CR_VEC2_<VecT>(0, 0);

			return CR_VEC2_<VecT>(X / value, Y / value);
		}

		CR_VEC2_<VecT>& operator+=(const CR_VEC2_<VecT>& other)
		{
			*getThis() = *getThis() + other;
			return *getThis();
		}
		CR_VEC2_<VecT>& operator-=(const CR_VEC2_<VecT>& other)
		{
			*getThis() = *getThis() - other;
			return *getThis();
		}
		CR_VEC2_<VecT>& operator*=(const CR_VEC2_<VecT>& other)
		{
			*getThis() = *getThis() * other;
			return *getThis();
		}
		CR_VEC2_<VecT>& operator/=(const CR_VEC2_<VecT>& other)
		{
			*getThis() = *getThis() / other;
			return *getThis();
		}

		CR_VEC2_<VecT>& operator*=(const VecT& value)
		{
			*getThis() = *getThis() * value;
			return *getThis();
		}
		CR_VEC2_<VecT>& operator/=(const VecT& value)
		{
			*getThis() = *getThis() / value;
			return *getThis();
		}
	};

	typedef CR_VEC2_<int> CR_VEC2I;
	typedef CR_VEC2_<uint> CR_VEC2U;
	typedef CR_VEC2_<float> CR_VEC2F;
}
