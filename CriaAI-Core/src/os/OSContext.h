﻿/******************************************************************************
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
#pragma once

#include "../Common.hpp"

namespace cria_ai { namespace os {
	
	class CROSContext
	{
	public:
		static CROSContext* s_Instance;
	protected:

		virtual crresult init() = 0;

		virtual void sleep(uint sec, uint ms) = 0;
		virtual CR_VEC2I getMousePos() = 0;
		virtual CR_RECT getVirtualScreenClientArea() = 0;

		CROSContext();
	public:
		virtual ~CROSContext();

		static crresult InitInstance();
		static crresult TerminateInstance();

		static void Sleep(uint sec, uint ms = 0);
		static CR_VEC2I GetMousePos();
		static CR_RECT GetVirtualScreenClientArea();
	};
}}
