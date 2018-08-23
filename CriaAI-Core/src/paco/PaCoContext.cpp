/******************************************************************************
* Cria  - The worst artificial intelligence on the market.                    *
*         <https://github.com/xFrednet/CriaAI>                                *
*                                                                             *
* =========================================================================== *
* Copyright (C) 2017, 2018, xFrednet <xFrednet@gmail.com>                     *
*                                                                             *
* This software is provided 'as-is', without any express or implied warranty. *
* In no event will the authors be held liable for any damages arising from    *
* the use of this software.                                                   *
*                                                                             *
* Permission is hereby granted, free of charge, to anyone to use this         *
* software for any purpose, including the rights to use, copy, modify,        *
* merge, publish, distribute, sublicense, and/or sell copies of this          *
* software, subject to the following conditions:                              *
*                                                                             *
*   1.  The origin of this software must not be misrepresented; you           *
*       must not claim that you wrote the original software. If you           *
*       use this software in a product, an acknowledgment in the              *
*       product documentation would be greatly appreciated but is not         *
*       required                                                              *
*                                                                             *
*   2.  Altered source versions should be plainly marked as such, and         *
*       must not be misrepresented as being the original software.            *
*                                                                             *
*   3.  This code should not be used for any military or malicious            *
*       purposes.                                                             *
*                                                                             *
*   4.  This notice may not be removed or altered from any source             *
*       distribution.                                                         *
*                                                                             *
******************************************************************************/
#include "PaCoContext.h"

#ifdef CRIA_PACO_CUDA
#	include "cuda/CuContext.cuh"
#endif

#ifdef CRIA_PACO_NULL
#	include "null/NuContext.h"
#endif

namespace cria_ai { namespace paco {
	
	CRPaCoContext* CRPaCoContext::s_Instance = nullptr;

	crresult CRPaCoContext::InitInstance()
	{
		CRPaCoContext* instance = nullptr;

#ifdef CRIA_PACO_CUDA
		instance = new cu::CRCuContext();
#elif defined(CRIA_PACO_NULL)
		instance = new null::CRNuContext();
#endif

		if (!instance)
			return CRRES_ERR_PACO_IS_NOT_SUPPORTED;

		crresult result = instance->init();
		if (CR_FAILED(result))
		{
			delete instance;
			return result;
		}

		s_Instance = instance;
		
		return CRRES_OK;
	}
	crresult CRPaCoContext::TerminateInstance()
	{
		CRPaCoContext* instance = s_Instance;
		s_Instance = nullptr;
		
		delete instance;

		return CRRES_OK;
	}

	void* CRPaCoMalloc(size_t size)
	{
#ifdef CRIA_PACO_CUDA
		return cu::CRCuMalloc(size);
#elif defined(CRIA_PACO_NULL)
		return null::CRNuMalloc(size);
#else
#		error CRPaCoMalloc is not implementet for the current parallel computing API
		return nullptr;
#endif
	}
	void CRPaCoFree(void* mem)
	{
#ifdef CRIA_PACO_CUDA
		return cu::CRCuFree(mem);
#elif defined(CRIA_PACO_NULL)
		return null::CRNuFree(mem);
#else
#		error CRPaCoFree is not implementet for the current parallel computing API
		return nullptr;
#endif
	}

}}