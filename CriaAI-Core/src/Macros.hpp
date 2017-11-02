#pragma once

#define CRIA_API                       __declspec(dllexport)

#define MAX(x, y)                      ((x > y) ? x : y)
#define MIN(x, y)                      ((x < y) ? x : y)

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

void LibTest();