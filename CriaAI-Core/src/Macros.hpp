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
 * Value helpers
 */
#define MAX(x, y)                      ((x > y) ? x : y)
#define MIN(x, y)                      ((x < y) ? x : y)

#define CLAMP_VALUE(x, min, max) \
if (min <= max) {\
		if (x < min)\
			x = min; \
		else if (x > max)\
			x = max; \
}
#define SWAP_INTS(x, y) \
{\
	int oldValue = x;\
	x = y;\
	y = oldValue; \
}
#define SWAP_FLOATS(x, y) \
{\
	float oldValue = x;\
	x = y;\
	y = oldValue; \
}

#define CAN_UINT32_MUL(a, b)             ((b == 0) || a <= 0xffffffffui32 / b)

/*
 * debugging help
 */
#include <stdio.h>

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