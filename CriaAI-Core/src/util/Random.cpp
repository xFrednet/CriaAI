#include "Random.h"

#include <cstdlib>
#include <time.h>

namespace cria_ai
{
	void RandSetSeed(long seed)
	{
		srand(seed);
	}
	void RandNewSeed()
	{
		RandSetSeed((long)time(0));
	}

	float RandFloat()
	{
		return (float)rand() / (float)RAND_MAX;
	}
	int32 RandInt(int32 max)
	{
		return rand() % max;
	}
}
