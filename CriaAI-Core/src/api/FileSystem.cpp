#include "FileSystem.h"

namespace cria_ai
{
	bool CreateContainingDir(const String& fileName)
	{
		if (fileName.find('/') == fileName.npos)
			return true; /* => the file is in no directory */

		String dir = fileName.substr(0, fileName.find_last_of('/'));
		if (DoesDirExists(dir))
			return true;

		return CreateDir(dir);
	}
}

#if defined(_WIN32) || defined(_WIN64)

#include <windows.h>

namespace cria_ai
{	
	bool DoesDirExists(const String& directory)
	{
		struct stat info;

		if (SUCCEEDED(stat(directory.c_str(), &info)))
			return ((info.st_mode & S_IFDIR) != 0);

		return false;
	}
	bool CreateDir(const String& directory)
	{
		return (CreateDirectory(directory.c_str(), nullptr) != 0);
	}
}
#else
#error The function "CreateDir" is not defined for the defined operation system
#endif
