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
#pragma once

#include "../Common.hpp"


#define CR_DESTOP_WINDOW_TITLE         ""

namespace cria_ai { namespace os {
	
	class CRWindow;

	typedef cr_ptr<CRWindow> CRWindowPtr;

	class CRWindow
	{
	public:
		static CRWindowPtr CreateDestopWindowInstance(crresult* result = nullptr);
		static CRWindowPtr CreateInstance(const String& title, crresult* result = nullptr);
	protected:
		String  m_Title;
		
		/*
		 * constructor and destructor
		 */
		CRWindow(const String& title);
		virtual crresult init(const String& title) = 0;
	public:
		virtual ~CRWindow();

		inline String getTitle() const
		{
			return m_Title;
		}

		virtual bool isFocussed() const = 0;

		virtual crresult setPos(int x, int y) = 0;
		virtual crresult setSize(uint width, uint height) = 0;
		virtual crresult setClientArea(const CR_RECT& bounds) = 0;

		virtual CR_RECT getClientArea() const = 0;
	};



}}