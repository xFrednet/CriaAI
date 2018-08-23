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

#include "../Macros.hpp"
#include "../Types.hpp"

#include <chrono>

#define CR_MS_IN_SEC                   1000

namespace cria_ai
{
	
	/**
	 * \brief This class stops the time from the start point to the end point.
	 * The result is a double representing seconds and fractions of seconds.
	 */
	class StopWatch
	{
		typedef std::chrono::high_resolution_clock::time_point time_point;
	private:
		time_point m_StartTime;
		time_point m_StopTime;

	public:
		StopWatch();

		void start();
		void stop();

		time_point getStart() const;
		time_point getStop() const;

		double getTimeSinceStart() const;
		double getTime() const;

		uint getTimeMSSinceStart() const;
		uint getTimeMS() const;
	};
}
