#pragma once

#include "../Types.hpp"

/*
 * Note these functions are just wrapper functions for the api specific random function. 
 *
 */
namespace cria_ai
{
	void CRRandSetSeed(long seed);
	void CRRandNewSeed();

	float CRRandFloat();
	int32 CRRandInt(int32 cap = INT32_MAX);

}