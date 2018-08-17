#pragma once

#include "../Types.hpp"
#include "../util/CRResult.h"

#ifndef CR_FILE_ENDL
#	define CR_FILE_ENDL "\r\n"
#endif

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
		
		/* A value storing the current read or write position, this value is not 
		 * maintained automatically and should not be used for validation. It can also
		 * be modified all the time because the const keyword should mean that the Data and the Size
		 * are constant. This behavior is especially used by loading functions
		 */
		mutable size_t Position; 
		        byte*  Data;     /* The data of this buffer */
	} CR_BYTE_BUFFER;

	CR_BYTE_BUFFER* CRByteBufferCreate(size_t size);
	void            CRByteBufferDelete(CR_BYTE_BUFFER* byteBuffer);
	bool            CRByteBufferValid(CR_BYTE_BUFFER const* bytebuffer);

	CR_BYTE_BUFFER* CRFileRead(const String& file, crresult* result = nullptr);
	crresult        CRFileWrite(const String& file, CR_BYTE_BUFFER const* data);
	
	std::ifstream   CROpenFileIn(const String& file);
	std::ofstream   CROpenFileOut(const String& file);
	std::fstream    CROpenFile(const String& file);

}