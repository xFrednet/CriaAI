#include "Random.h"

#include <cstdlib>
#include <time.h>

namespace cria_ai
{
	void CRRandSetSeed(long seed)
	{
		srand(seed);
	}
	void CRRandNewSeed()
	{
		CRRandSetSeed((long)time(0));
	}

	float CRRandFloat()
	{
		return (float)rand() / (float)RAND_MAX;
	}
	int32 CRRandInt(int32 cap)
	{
		return rand() % cap;
	}
}
