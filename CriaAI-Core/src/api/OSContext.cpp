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
#include "OSContext.h"
#include "win/WinOSContext.h"

namespace cria_ai { namespace api {
	
	CROSContext* CROSContext::s_Instance = nullptr;

	CROSContext::CROSContext()
	{
	}
	CROSContext::~CROSContext()
	{
	}

	crresult CROSContext::InitInstance()
	{
		CROSContext* instance = nullptr;
		
		/*
		 * Create instance
		 */
#ifdef CRIA_OS_WIN
		instance = new win::CRWinOSContext();
#endif
		if (!instance)
		{
			return CRRES_ERR_NEW_FAILED;
		}

		/*
		 * init
		 */
		crresult result = instance->init();
		if (CR_FAILED(result))
		{
			delete instance;
			return result;
		}

		/*
		 * finishing
		 */
		s_Instance = instance;
		return CRRES_OK;
	}

	crresult CROSContext::TerminateInstance()
	{
		/*
		 * validation check
		 */
		if (!s_Instance)
			return CRRES_OK_API_STATIC_INSTANCE_IS_NULL;

		/*
		 * Deleting the instance
		 */
		CROSContext* instance = s_Instance;
		s_Instance = nullptr;
		delete instance;

		/*
		 * Yay return "okay"
		 */
		return CRRES_OK;
	}
}}
