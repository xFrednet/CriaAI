#pragma once

#include "../Types.hpp"
#include "../util/CRResult.h"

namespace cria_ai {

	bool CRCreateContainingDir(const String& fileName);

	bool CRDoesDirExists(const String& directory);
	bool CRCreateDir(const String& directory);

	/* //////////////////////////////////////////////////////////////////////////////// */
	// // File related utility //
	/* //////////////////////////////////////////////////////////////////////////////// */
	typedef struct CR_BYTE_BUFFER_ 
	{
		size_t Size;     /* The total size of this buffer*/
		size_t Position; /* A value storing the current read or write position, this value is not maintained automatically and should not be used for validation */
		byte*  Data;     /* The data of this buffer */
	} CR_BYTE_BUFFER;

	CR_BYTE_BUFFER* CRByteBufferCreate(size_t size);
	void            CRByteBufferDelete(CR_BYTE_BUFFER* byteBuffer);
	bool            CRByteBufferValid(CR_BYTE_BUFFER const* bytebuffer);

	CR_BYTE_BUFFER* CRFileRead(const String& file, crresult* result);
	crresult        CRFileWrite(const String& file, CR_BYTE_BUFFER const* data);
	
	std::ifstream   CROpenFileIn(const String& file);
	std::ofstream   CROpenFileOut(const String& file);
	std::fstream    CROpenFile(const String& file);

}