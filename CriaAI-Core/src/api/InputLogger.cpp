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
#include "InputLogger.h"

#include "win/WinInputLogger.h"

namespace cria_ai { namespace api {

	/*
	* Singleton instance
	*/
	CRInputLogger* CRInputLogger::s_Instance = nullptr;

	crresult CRInputLogger::InitInstance()
	{
		/*
		 * singleton check
		 */
		if (s_Instance)
			return CRRES_OK_SINGLETON_IS_ALREADY_INITIALIZED;

		/*
		 * create the instance
		 */
#ifdef CRIA_OS_WIN
		s_Instance = new win::CRWinInputLogger();
#endif

		if (!s_Instance)
			return CRRES_ERR_API_OS_UNSUPPORTED;

		/*
		 * Init and return the result
		 */
		return s_Instance->init();
	}

	crresult CRInputLogger::TerminateInstance()
	{
		/*
		 * singleton check
		 */
		if (!s_Instance)
			return CRRES_ERR_API_STATIC_INSTANCE_IS_NULL;

		CRInputLogger* logger = s_Instance;
		s_Instance = nullptr;
		delete logger;

		return CRRES_OK;
	}

	/*
	* Class content
	*/
	CRInputLogger::CRInputLogger()
		: m_MouseWheelPos(0)
	{
		memset(m_KeyStates, false, sizeof(bool) * (CR_KEY_ID_LAST + 1));
		memset(m_MButtonStates, false, sizeof(bool) * (CR_MBUTTON_ID_LAST + 1));
	}
	CRInputLogger::~CRInputLogger()
	{
	}

	/*
	 * Calling callbacks
	 */
	void CRInputLogger::processKey(CR_KEY_ID keyID, bool pressed)
	{
		if (keyID < CR_KEY_ID_LAST)
			m_KeyStates[keyID] = pressed;

		for (cr_logger_key_cb callback : m_KeyCallbacks)
		{
			(*callback)(keyID, pressed);
		}
	}
	void CRInputLogger::processMButton(CR_MBUTTON_ID buttonID, bool pressed)
	{
		if (buttonID <= CR_MBUTTON_ID_LAST)
			m_MButtonStates[buttonID] = pressed;

		for (cr_logger_mbutton_cb callback : m_MButtonCallbacks)
		{
			(*callback)(buttonID, pressed);
		}
	}
	void CRInputLogger::processMMove(CR_VEC2I position, int xMotion, int yMotion)
	{
		m_MousePos = position;

		for (cr_logger_mmove_cb callback : m_MMoveCallbacks)
		{
			(*callback)(position, xMotion, yMotion);
		}
	}
	void CRInputLogger::processMWheel(int change)
	{
		m_MouseWheelPos += change;

		for (cr_logger_mwheel_cb callback : m_MWheelCallbacks)
		{
			(*callback)(change);
		}
	}
}}
