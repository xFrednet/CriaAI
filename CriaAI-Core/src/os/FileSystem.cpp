#include "FileSystem.h"

namespace cria_ai
{
	bool CRCreateContainingDir(const String& fileName)
	{
		if (fileName.find('/') == fileName.npos)
			return true; /* => the file is in no directory */

		String dir = fileName.substr(0, fileName.find_last_of('/'));
		if (CRDoesDirExists(dir))
			return true;

		return CRCreateDir(dir);
	}
}

#if defined(_WIN32) || defined(_WIN64)

#include "win/WinOSContext.h"

namespace cria_ai
{	
	bool CRDoesDirExists(const String& directory)
	{
		struct stat info;

		if (SUCCEEDED(stat(directory.c_str(), &info)))
			return ((info.st_mode & S_IFDIR) != 0);

		return false;
	}
	bool CRCreateDir(const String& directory)
	{
		return (CreateDirectory(directory.c_str(), nullptr) != 0);
	}
}
#else
#error The function "CRCreateDir" is not defined for the defined operation system
#endif

namespace cria_ai {

	CR_BYTE_BUFFER* CRByteBufferCreate(size_t size)
	{
		/*
		 * Validation
		 */
		if (size == 0)
			return nullptr; // Invalid size

		/*
		 * Memory allocation
		 */
		size_t totalSize = sizeof(CR_BYTE_BUFFER) + size;
		CR_BYTE_BUFFER* buffer = (CR_BYTE_BUFFER*)malloc(totalSize);
		if (!buffer)
			return nullptr; // The memory allocation failed

		/*
		 * Filling the fields
		 */
		buffer->Size = size;
		buffer->Data = (byte*)((uintptr)buffer + sizeof(CR_BYTE_BUFFER));
		memset(buffer->Data, 0, sizeof(buffer->Size));

		return buffer;
	}
	void CRByteBufferDelete(CR_BYTE_BUFFER* byteBuffer)
	{
		if (byteBuffer)
			free(byteBuffer);
	}
	bool CRByteBufferValid(CR_BYTE_BUFFER const* byteBuffer)
	{
		if (!byteBuffer)
			return false;

		if (byteBuffer->Size == 0)
			return false;

		if (!byteBuffer->Data)
			return false;

		return true;
	}

	/* //////////////////////////////////////////////////////////////////////////////// */
	// // File related utility //
	/* //////////////////////////////////////////////////////////////////////////////// */
	CR_BYTE_BUFFER* CRFileRead(const String& file, crresult* result)
	{
		/*
		 * Validation
		 */
		if (file.empty())
		{
			if (result)
				*result = CRRES_ERR_INVALID_ARGUMENTS;
			return nullptr;
		}

		/*
		 * open the file 
		 */
		std::ifstream inStream = CROpenFileIn(file);
		if (!inStream.good())
		{
			if (result)
				*result = CRRES_ERR_OS_READ_FROM_FILE_FAILED;
			inStream.close();
			return nullptr;
		}

		/*
		 * create the buffer
		 */
		inStream.seekg(0, inStream.end);
		size_t fileSize = inStream.tellg();
		inStream.seekg(0);
		CR_BYTE_BUFFER* buffer = CRByteBufferCreate(fileSize);
		if (!buffer)
		{
			if (result)
				*result = CRRES_ERR_FAILED_TO_CREATE_BYTE_BUFFER;
			inStream.close();
			return nullptr;
		}

		/*
		 * read file
		 */
		inStream.read((char*)buffer->Data, buffer->Size);
		if (!inStream.good())
		{
			if (result)
				*result = CRRES_ERR_OS_READ_FROM_FILE_FAILED;
			inStream.close();
			CRByteBufferDelete(buffer);
			return nullptr;
		}

		/*
		 * finish and clean up
		 */
		if (result)
			*result = CRRES_OK;
		inStream.close();
		return buffer;
	}
	crresult        CRFileWrite(const String& file, CR_BYTE_BUFFER const* data)
	{
		/*
		 * Validation
		 */
		if (file.empty() || !data)
			return CRRES_ERR_INVALID_ARGUMENTS;

		/*
		 * Making sure the containing directory exists
		 */
		CRCreateContainingDir(file);

		/*
		 * Open stream
		 */
		std::ofstream outStream = CROpenFileOut(file);
		if (!outStream.good())
		{
			outStream.close();
			return CRRES_ERR_OS_FILE_COULD_NOT_BE_OPENED;
		}

		/*
		 * Write the data
		 */
		outStream.write((char const*)data->Data, data->Size);
		if (!outStream.good())
		{
			outStream.close();
			return CRRES_ERR_OS_WRITE_TO_FILE_FAILED;
		}

		/*
		 * Finish and goodbye
		 */
		outStream.close();
		return CRRES_OK;
	}
	
	std::ifstream CROpenFileIn(const String& file)
	{
		return std::ifstream(file.c_str(), std::ios_base::in | std::ios_base::binary);
	}
	std::ofstream CROpenFileOut(const String& file)
	{
		return std::ofstream(file.c_str(), std::ios_base::out | std::ios_base::binary);
	}
	std::fstream  CROpenFile(const String& file)
	{
		return std::fstream(file.c_str(), std::ios_base::in | std::ios_base::out | std::ios_base::binary);
	}

}
