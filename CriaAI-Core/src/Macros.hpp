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

/*
 * Macros
 */
#define CRIA_API                       __declspec(dllexport)

#if defined(_WIN32) || defined(_WIN64)
#	define CRIA_OS_WIN
#else
#	error The targeted operating system is not supported, sorry!!!
#endif


/*
 * PaCo
 */
#if CR_CUDA
#	define CRIA_PACO_CUDA
#else
#	define CRIA_PACO_NULL
#endif


/*
 * Value helpers
 */
#define MAX(x, y)                      ((x > y) ? x : y)
#define MIN(x, y)                      ((x < y) ? x : y)

#define CR_CLAMP_VALUE(x, min, max) \
if ((min) <= (max)) { \
		if ((x) < (min)) \
			(x) = (min); \
		else if ((x) > (max)) \
			(x) = (max); \
}
#define CR_SWAP_INTS(x, y) \
{\
	int oldValue = x;\
	x = y;\
	y = oldValue; \
}
#define CR_SWAP_FLOATS(x, y) \
{\
	float oldValue = x;\
	x = y;\
	y = oldValue; \
}

#define CR_CAN_UINT32_MUL(a, b)             ((b == 0) || a <= 0xffffffffui32 / (b))

#define CR_IS_FLAG_SET(value, flag)         (((value) & (flag)) != 0)

/*
 * debugging help
 */
#include <cstdio>

#ifdef CRIA_DEBUG_ENABLE_INFO
#	define CRIA_INFO_PRINTF(...)            printf("CRIA [INFO ]: "); printf(__VA_ARGS__); printf("\n");
#	define CRIA_ALERT_PRINTF(...)           printf("CRIA [ALERT]: "); printf(__VA_ARGS__); printf("\n");
#	define CRIA_ERROR_PRINTF(...)           printf("CRIA [ERROR]: "); printf(__VA_ARGS__); printf("\n");
#else
#	define CRIA_INFO_PRINTF(...)
#	define CRIA_ALERT_PRINTF(...)
#	define CRIA_ERROR_PRINTF(...)           printf("CRIA [ERROR]: "); printf(__VA_ARGS__); printf("\n");
#endif

#if defined(CRIA_AUTO_TEST_RESULTS) || defined(CRIA_DEBUG_ENABLE_INFO)
#	define CRIA_AUTO_ASSERT(x, ...) \
		if (!(x)) { \
			CRIA_ERROR_PRINTF("CRIA_AUTO_ASSERT(%s) failed", #x); \
			CRIA_ERROR_PRINTF("File: %s", __FILE__); \
			CRIA_ERROR_PRINTF("Line: %u", __LINE__); \
			CRIA_ERROR_PRINTF(__VA_ARGS__);\
			getchar(); \
		} 
#else
#	define CRIA_AUTO_ASSERT(x, ...)
#endif

/*
 * Other helper
 */
namespace cria_ai {}
