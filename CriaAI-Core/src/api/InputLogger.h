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

#include "InputUtil.h"

namespace cria_ai { namespace api {

	typedef void (*cr_logger_key_cb)     (CR_KEY_ID keyID, bool down);
	typedef void (*cr_logger_mbutton_cb) (CR_MOUSE_BUTTON_ID buttonID, bool down);
	typedef void (*cr_logger_mmove_cb)   (int x, int y);

	class CRInputLogger
	{
		/*
		 * Singleton instance
		 */
	protected:
		static CRInputLogger* s_Instance;
	public:
		static crresult InitInstance();

		/*
		 * Class content
		 */
	protected:
		std::list<cr_logger_key_cb>     m_KeyCallbacks;
		std::list<cr_logger_mbutton_cb> m_MButtonCallbacks;
		std::list<cr_logger_mmove_cb>   m_MMoveCallbacks;

		CRInputLogger();
		virtual crresult init() = 0;
	public:
		virtual ~CRInputLogger();
	
		/*
		* Calling callbacks
		*/
	protected:
		void callKeyCBs(CR_KEY_ID keyID, bool down);
		void callMButtonsCBs(CR_MOUSE_BUTTON_ID buttonID, bool down);
		void callMMoveCBs(int x, int y);

		virtual void update() = 0;

		/*
		 * static functions
		 */
	public:
		inline static void Update()
		{
			if (s_Instance)
				s_Instance->update();
		}

		static void AddKeyCallback(cr_logger_key_cb cb)
		{
			if (s_Instance)
				s_Instance->m_KeyCallbacks.push_front(cb);
		}
		static void RemoveKeyCallback(cr_logger_key_cb cb)
		{
			if (s_Instance)
				s_Instance->m_KeyCallbacks.remove(cb);
		}
		static void AddMButtonCallback(cr_logger_mbutton_cb cb)
		{
			if (s_Instance)
				s_Instance->m_MButtonCallbacks.push_front(cb);
		}
		static void RemoveMButtonCallback(cr_logger_mbutton_cb cb)
		{
			if (s_Instance)
				s_Instance->m_MButtonCallbacks.remove(cb);
		}
		static void AddMMoveCallback(cr_logger_mmove_cb cb)
		{
			if (s_Instance)
				s_Instance->m_MMoveCallbacks.push_front(cb);
		}
		static void RemoveMMoveCallback(cr_logger_mmove_cb cb)
		{
			if (s_Instance)
				s_Instance->m_MMoveCallbacks.remove(cb);
		}
	};

}}