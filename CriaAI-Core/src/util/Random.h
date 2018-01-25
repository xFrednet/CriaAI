#pragma once

#include "../Types.hpp"

/*
 * Note these functions are just wrapper functions for the api specific random function. 
 *
 */
namespace cria_ai
{
	void RandSetSeed(long seed);
	void RandNewSeed();

	float RandFloat();
	int32 RandInt(int32 cap = INT32_MAX);

}