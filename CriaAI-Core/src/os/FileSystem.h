#pragma once

#include "../Common.hpp"

namespace cria_ai
{
	bool CreateContainingDir(const String& fileName);

	bool DoesDirExists(const String& directory);
	bool CreateDir(const String& directory);
}