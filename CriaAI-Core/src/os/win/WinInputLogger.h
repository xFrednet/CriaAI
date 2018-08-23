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

#include "../InputLogger.h"

#ifdef CRIA_OS_WIN

#include "WinOSContext.h"

namespace cria_ai { namespace os { namespace win {
	
	class CRWinInputLogger : public CRInputLogger
	{
	private:
		HHOOK m_KeyboardHook;
		HKL m_KeyLayout;
		
		HHOOK m_MouseHook;

		CR_VEC2I m_OldMousePos;

	public:
		CRWinInputLogger();
		~CRWinInputLogger();
	private:
		static LRESULT CALLBACK HandleKeyboardHook(UINT message, WPARAM wp, LPARAM lp);
		static LRESULT CALLBACK HandleMouseHook(UINT message, WPARAM wp, LPARAM lp);
	protected:
		crresult init() override;

		void update() override;
		
		void processNewMousePos(CR_VEC2I newPos);

	};

}}}

#endif
