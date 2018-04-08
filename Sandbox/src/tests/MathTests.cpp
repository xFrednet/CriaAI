#include "MathTests.h"

#define TEST_ASSERT(x) \
if (!(x)) {	\
		printf("=========================================== \n");\
		printf("TEST FAILED: (%s) \n", #x); \
		printf("File: %s \n", __FILE__); \
		printf("Line: %u \n", __LINE__); \
		printf("=========================================== \n");\
		getchar(); \
	return false;\
}

/**
 * \brief This test a selected Vec2 type if everything works. Note that the values of a b and c should be different
 * 
 * \tparam Vec2Type The Vector2 type that should be tested
 * 
 * \param a Vec2 no. 1
 * \param b Vec2 no. 2
 * \param c Vec2 no. 3
 * 
 * \return This returns true when all test pass.
 */
template <typename Vec2Type>
bool TestVec2T(Vec2Type a, Vec2Type b, Vec2Type c)
{
	using namespace cria_ai;

	/*
	 * They have to be different;
	 */
	if ((a.X == b.X && a.Y == b.Y) ||
		(a.X == c.X && a.Y == c.Y) ||
		(b.X == c.X && b.Y == c.Y))
		return false;

	/*
	* Comparison
	*/
	TEST_ASSERT(a == a);
	TEST_ASSERT(b == b && c == c);

	TEST_ASSERT(!(a != a));
	TEST_ASSERT(!(b != b) && !(c != c));

	TEST_ASSERT((a != b));
	TEST_ASSERT((a != c));
	TEST_ASSERT((b != c));

	/*
	 * Operators
	 */
	//TODO "Continue Testing" who gets the reference ?

	return true;
}
bool TestVec2()
{
	using namespace cria_ai;
	
	if (!TestVec2T<CR_VEC2I>(CR_VEC2I(1, 1), CR_VEC2I(-2, 4), CR_VEC2I(6, -7)))
		return false;
	if (!TestVec2T<CR_VEC2F>(CR_VEC2F(1.9f, 1.8f), CR_VEC2F(2.7f, 4.2f), CR_VEC2F(6.3f, 7.5f)))
		return false;
	if (!TestVec2T<CR_VEC2U>(CR_VEC2U(1, 1), CR_VEC2U(2, 4), CR_VEC2U(6, 7)))
		return false;

	return true;
}
