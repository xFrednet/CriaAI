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

#include "../Types.hpp"

#define CR_SUCCESS(crresult)           ((crresult).Value >= 0)
#define CR_FAILED(crresult)            ((crresult).Value < 0 )

namespace cria_ai
{
	
	/**
	 * \brief 
	 * 
	 * Structure:
	 * 
	 *       1 1 1 1 1 1 1
	 *       6 5 4 3 2 1 0 9 8 7 6 5 4 3 2 1
	 *      +-+-----+-------+---------------+
	 *      |R|-BUF-|-Source|-----Code------|
	 *      +-+-----+-------+---------------+
	 *    
	 *      R      - Result: 
	 *                 0: success
	 *                 1: error
	 *      
	 *      BUF    - a clear Buffer to make the results more readable
	 *      
	 *      Source - Indicates the source (15 possible sources)
	 *      
	 *      Code   - The error code (256 possible codes)
	 * 
	 */
	typedef struct crresult_ {
		int16_t Value;

		inline bool operator==(const crresult_& other) const
		{
			return Value == other.Value;
		}
		inline bool operator!=(const crresult_& other) const
		{
			return Value != other.Value;
		}
	} crresult;

}

/* //////////////////////////////////////////////////////////////////////////////// */
// // Operators //
/* //////////////////////////////////////////////////////////////////////////////// */
#define CRIA_RES_TYPEDEF(value)                  (cria_ai::crresult{(int16_t)value})

/* //////////////////////////////////////////////////////////////////////////////// */
// // General //
/* //////////////////////////////////////////////////////////////////////////////// */
#define CRRES_ERROR                              CRIA_RES_TYPEDEF(0x8000)
#define CRRES_SUCCESS                            CRIA_RES_TYPEDEF(0x0000)
#define CRRES_OK                                 CRIA_RES_TYPEDEF(0x0000)

/* ********************************************************* */
// * Masks *
/* ********************************************************* */
#define CRRES_MASK_RESULT                        0x8000
#define CRRES_MASK_SOURCE                        0x0f00
#define CRRES_MASK_CODE                          0x00ff

/* ********************************************************* */
// * Sources *
/* ********************************************************* */
#define CRRES_SOURCE_UNKNOWN                     CRIA_RES_TYPEDEF(0x0000)
#define CRRES_SOURCE_UTILS                       CRIA_RES_TYPEDEF(0x0100)

#define CRRES_SOURCE_NN                          CRIA_RES_TYPEDEF(0x0400)

#define CRRES_SOURCE_OS                          CRIA_RES_TYPEDEF(0x0A00)
#define CRRES_SOURCE_WIN                         CRIA_RES_TYPEDEF(0x0B00)
//CRRES_SOURCE_OS                                                 0x0800
//CRRES_SOURCE_WIN                                                0x0900
//CRRES_SOURCE_LINUX                                              0x0A00
//CRRES_SOURCE_MAC                                                0x0B00

#define CRRES_SOURCE_PACO                        CRIA_RES_TYPEDEF(0x0E00)
#define CRRES_SOURCE_CUDA                        CRIA_RES_TYPEDEF(0x0F00)
//CRRES_SOURCE_PACO                                               0x0C00
//CRRES_SOURCE_CUDA                                               0x0D00
//CRRES_SOURCE_PACO_NULL                                          0x0E00
//CRRES_SOURCE_OPENCL                                             0x0F00

/* //////////////////////////////////////////////////////////////////////////////// */
// // ZA_RESULT_SOURCE_NO_SOURCE //
/* //////////////////////////////////////////////////////////////////////////////// */
#define CRRES_OK_SINGLETON_IS_ALREADY_INITIALIZED CRIA_RES_TYPEDEF(0x0001)
#define CRRES_OK_STATIC_INSTANCE_IS_NULL         CRIA_RES_TYPEDEF(0x0002)

#define CRRES_ERR_INVALID_ARGUMENTS              CRIA_RES_TYPEDEF(0x8001)
#define CRRES_ERR_FUNCTION_NOT_IMPLEMENTED       CRIA_RES_TYPEDEF(0x8002)
#define CRRES_ERR_INVALID_DIMENSIONS             CRIA_RES_TYPEDEF(0x8003)
#define CRRES_ERR_MISSING_INFORMATION            CRIA_RES_TYPEDEF(0x8004)
// Yes the original name should be CRRES_ERR_YAY_MULTI_THREADED_TIMING but you know multi threading
#define CRRES_ERR_TIMING_THREADED_YAY_MULTI      CRIA_RES_TYPEDEF(0x8005)
#define CRRES_ERR_MALLOC_FAILED                  CRIA_RES_TYPEDEF(0x8006)
#define CRRES_ERR_NEW_FAILED                     CRIA_RES_TYPEDEF(0x8007)
#define CRRES_ERR_MAKE_SHARED_FAILED             CRIA_RES_TYPEDEF(0x8008)
#define CRRES_ERR_STATIC_VAR_IS_ALREADY_VALID    CRIA_RES_TYPEDEF(0x8009)
#define CRRES_ERR_FAILED_TO_CREATE_BYTE_BUFFER   CRIA_RES_TYPEDEF(0x8010)
#define CRRES_ERR_INVALID_BYTE_BUFFER_SIZE       CRIA_RES_TYPEDEF(0x8011)

/* //////////////////////////////////////////////////////////////////////////////// */
// // CRRES_SOURCE_UTILS //
/* //////////////////////////////////////////////////////////////////////////////// */
#define CRRES_ERR_UTILS_UNKNOWN                  CRIA_RES_TYPEDEF(0x8100)

#define CRRES_ERR_UTILS_FAILED_TO_CREATE_FBMP    CRIA_RES_TYPEDEF(0x8101)

/* //////////////////////////////////////////////////////////////////////////////// */
// // CRRES_SOURCE_NN //
/* //////////////////////////////////////////////////////////////////////////////// */
#define CRRES_ERR_NN_UNKNOWN                     CRIA_RES_TYPEDEF(0x8400)
#define CRRES_OK_NN                              CRIA_RES_TYPEDEF(0x0400)

#define CRRES_ERR_NN_LAYER_INVALID_NEURON_COUNT  CRIA_RES_TYPEDEF(0x8401)
#define CRRES_ERR_NN_LAYER_MATRICIES_NOT_NULL    CRIA_RES_TYPEDEF(0x8402)
#define CRRES_ERR_NN_LAYER_MATRICIES_INIT_FAILED CRIA_RES_TYPEDEF(0x8403)

/* //////////////////////////////////////////////////////////////////////////////// */
// // CRRES_SOURCE_OS //
/* //////////////////////////////////////////////////////////////////////////////// */
#define CRRES_ERR_OS_UNKNOWN                     CRIA_RES_TYPEDEF(0x8A00)
#define CRRES_OK_OS                              CRIA_RES_TYPEDEF(0x0A00)

#define CRRES_OK_OS_INPUTSIM_TARGET_NOT_FOCUSED  CRIA_RES_TYPEDEF(0x0A01)
#define CRRES_OK_OS_INPUTSIM_CURSOR_OUTSIDE      CRIA_RES_TYPEDEF(0x0A02)
#define CRRES_OK_OS_THREAD_IS_ALLREADY_RUNNING   CRIA_RES_TYPEDEF(0x0A03)
#define CRRES_OK_OS_THREAD_IS_ALLREADY_JOINED    CRIA_RES_TYPEDEF(0x0A04)

#define CRRES_ERR_OS_OS_UNSUPPORTED              CRIA_RES_TYPEDEF(0x8A01)
#define CRRES_ERR_OS_STATIC_INSTANCE_IS_NULL     CRIA_RES_TYPEDEF(0x8A02)
#define CRRES_ERR_OS_KEY_OUT_OF_BOUNDS           CRIA_RES_TYPEDEF(0x8A03)
#define CRRES_ERR_OS_BUTTON_OUT_OF_BOUNDS        CRIA_RES_TYPEDEF(0x8A04)
#define CRRES_ERR_OS_INPUTSIM_INIT_FAILED        CRIA_RES_TYPEDEF(0x8A05)
#define CRRES_ERR_OS_WINDOW_INIT_FAILED          CRIA_RES_TYPEDEF(0x8A06)
#define CRRES_ERR_OS_WINDOW_TITLE_NOT_FOUND      CRIA_RES_TYPEDEF(0x8A07)
#define CRRES_ERR_OS_TARGET_IS_NULL              CRIA_RES_TYPEDEF(0x8A08)
#define CRRES_ERR_OS_WINDOW_RESIZE_FAILED        CRIA_RES_TYPEDEF(0x8A09)
#define CRRES_ERR_OS_FAILED_TO_CREATE_DIR        CRIA_RES_TYPEDEF(0x8A0A)

#define CRRES_ERR_OS_FILE_COULD_NOT_BE_OPENED    CRIA_RES_TYPEDEF(0x8A11)
#define CRRES_ERR_OS_WRITE_TO_FILE_FAILED        CRIA_RES_TYPEDEF(0x8A12)
#define CRRES_ERR_OS_READ_FROM_FILE_FAILED       CRIA_RES_TYPEDEF(0x8A13)
#define CRRES_ERR_OS_FILE_FORMAT_UNKNOWN         CRIA_RES_TYPEDEF(0x8A14)

/* //////////////////////////////////////////////////////////////////////////////// */
// // CRRES_SOURCE_WIN //
/* //////////////////////////////////////////////////////////////////////////////// */
#define CRRES_ERR_WIN_UNKNOWN                    CRIA_RES_TYPEDEF(0x8B00)

#define CRRES_OK_WIN                             CRIA_RES_TYPEDEF(0x0B00)

#define CRRES_ERR_WIN_FAILED_TO_RETRIVE_DC       CRIA_RES_TYPEDEF(0x8B01)
#define CRRES_ERR_WIN_FAILED_TO_CREATE_DC        CRIA_RES_TYPEDEF(0x8B02)
#define CRRES_ERR_WIN_FAILED_TO_CREATE_HBMP      CRIA_RES_TYPEDEF(0x8B03)
#define CRRES_ERR_WIN_INPUT_THREAD_BLOCKED       CRIA_RES_TYPEDEF(0x8B04)
#define CRRES_ERR_WIN_SYSTEMPARMINFO_FAILED      CRIA_RES_TYPEDEF(0x8B05)
#define CRRES_ERR_WIN_FAILED_TO_INSTALL_HOCK     CRIA_RES_TYPEDEF(0x8B06)
#define CRRES_ERR_WIN_COULD_NOT_GET_KEY_LAYOUT   CRIA_RES_TYPEDEF(0x8B07)

/* //////////////////////////////////////////////////////////////////////////////// */
// // CRRES_SOURCE_PACO //
/* //////////////////////////////////////////////////////////////////////////////// */
#define CRRES_ERR_PACO                           CRIA_RES_TYPEDEF(0x8E00)

#define CRRES_ERR_PACO_IS_NOT_SUPPORTED          CRIA_RES_TYPEDEF(0x8E01)

/* //////////////////////////////////////////////////////////////////////////////// */
// // CRRES_SOURCE_CUDA //
/* //////////////////////////////////////////////////////////////////////////////// */
#define CRRES_ERR_CUDA_UNKNOWN                   CRIA_RES_TYPEDEF(0x8F00)
/* //////////////////////////////////////////////////////////////////////////////// */
// // CRGetCRResultName //
/* //////////////////////////////////////////////////////////////////////////////// */
#define CRIA_SWITCH_CRRESULT(macro)              case macro.Value: return String(#macro);
namespace cria_ai {

	inline String CRGetCRResultName(const crresult& result)
	{
		switch (result.Value & CRRES_MASK_SOURCE)
		{
			case CRRES_SOURCE_UNKNOWN.Value:
				switch (result.Value)
				{
					CRIA_SWITCH_CRRESULT(CRRES_OK);
					CRIA_SWITCH_CRRESULT(CRRES_ERROR);

					CRIA_SWITCH_CRRESULT(CRRES_OK_SINGLETON_IS_ALREADY_INITIALIZED);
					CRIA_SWITCH_CRRESULT(CRRES_OK_STATIC_INSTANCE_IS_NULL);

					CRIA_SWITCH_CRRESULT(CRRES_ERR_INVALID_ARGUMENTS);
					CRIA_SWITCH_CRRESULT(CRRES_ERR_FUNCTION_NOT_IMPLEMENTED);
					CRIA_SWITCH_CRRESULT(CRRES_ERR_INVALID_DIMENSIONS);
					CRIA_SWITCH_CRRESULT(CRRES_ERR_MISSING_INFORMATION);
					CRIA_SWITCH_CRRESULT(CRRES_ERR_TIMING_THREADED_YAY_MULTI);
					CRIA_SWITCH_CRRESULT(CRRES_ERR_MALLOC_FAILED);
					CRIA_SWITCH_CRRESULT(CRRES_ERR_NEW_FAILED);
					CRIA_SWITCH_CRRESULT(CRRES_ERR_MAKE_SHARED_FAILED);
					CRIA_SWITCH_CRRESULT(CRRES_ERR_STATIC_VAR_IS_ALREADY_VALID);
					CRIA_SWITCH_CRRESULT(CRRES_ERR_FAILED_TO_CREATE_BYTE_BUFFER);
					CRIA_SWITCH_CRRESULT(CRRES_ERR_INVALID_BYTE_BUFFER_SIZE);

					default:
						return String("CRRESULT_UNNAMED_RESULT");
				}
			case CRRES_SOURCE_UTILS.Value:
				switch (result.Value)
				{
					CRIA_SWITCH_CRRESULT(CRRES_ERR_UTILS_UNKNOWN);
					
					CRIA_SWITCH_CRRESULT(CRRES_ERR_UTILS_FAILED_TO_CREATE_FBMP);

					default:
						return String("CRRESULT_UNNAMED_RESULT");
				}
			case CRRES_SOURCE_NN.Value:
				switch (result.Value) 
				{
					CRIA_SWITCH_CRRESULT(CRRES_ERR_NN_UNKNOWN);
					CRIA_SWITCH_CRRESULT(CRRES_OK_NN);

					CRIA_SWITCH_CRRESULT(CRRES_ERR_NN_LAYER_INVALID_NEURON_COUNT);
					CRIA_SWITCH_CRRESULT(CRRES_ERR_NN_LAYER_MATRICIES_NOT_NULL);
					CRIA_SWITCH_CRRESULT(CRRES_ERR_NN_LAYER_MATRICIES_INIT_FAILED);

					default:
						return String("CRRESULT_UNNAMED_RESULT");
				}
			case CRRES_SOURCE_OS.Value:
				switch (result.Value) {
					CRIA_SWITCH_CRRESULT(CRRES_ERR_OS_UNKNOWN);
					CRIA_SWITCH_CRRESULT(CRRES_OK_OS);

					CRIA_SWITCH_CRRESULT(CRRES_OK_OS_INPUTSIM_TARGET_NOT_FOCUSED); 
					CRIA_SWITCH_CRRESULT(CRRES_OK_OS_INPUTSIM_CURSOR_OUTSIDE);
					CRIA_SWITCH_CRRESULT(CRRES_OK_OS_THREAD_IS_ALLREADY_RUNNING);
					CRIA_SWITCH_CRRESULT(CRRES_OK_OS_THREAD_IS_ALLREADY_JOINED);

					CRIA_SWITCH_CRRESULT(CRRES_ERR_OS_OS_UNSUPPORTED);
					CRIA_SWITCH_CRRESULT(CRRES_ERR_OS_STATIC_INSTANCE_IS_NULL);
					CRIA_SWITCH_CRRESULT(CRRES_ERR_OS_KEY_OUT_OF_BOUNDS);
					CRIA_SWITCH_CRRESULT(CRRES_ERR_OS_BUTTON_OUT_OF_BOUNDS);
					CRIA_SWITCH_CRRESULT(CRRES_ERR_OS_INPUTSIM_INIT_FAILED);
					CRIA_SWITCH_CRRESULT(CRRES_ERR_OS_WINDOW_INIT_FAILED);
					CRIA_SWITCH_CRRESULT(CRRES_ERR_OS_WINDOW_TITLE_NOT_FOUND);
					CRIA_SWITCH_CRRESULT(CRRES_ERR_OS_TARGET_IS_NULL);
					CRIA_SWITCH_CRRESULT(CRRES_ERR_OS_WINDOW_RESIZE_FAILED);
					CRIA_SWITCH_CRRESULT(CRRES_ERR_OS_FAILED_TO_CREATE_DIR);

					CRIA_SWITCH_CRRESULT(CRRES_ERR_OS_FILE_COULD_NOT_BE_OPENED);
					CRIA_SWITCH_CRRESULT(CRRES_ERR_OS_WRITE_TO_FILE_FAILED);
					CRIA_SWITCH_CRRESULT(CRRES_ERR_OS_READ_FROM_FILE_FAILED);
					CRIA_SWITCH_CRRESULT(CRRES_ERR_OS_FILE_FORMAT_UNKNOWN);

					default:
						return String("CRRESULT_UNNAMED_RESULT");
				}
			case CRRES_SOURCE_WIN.Value:
				switch (result.Value)
				{
					CRIA_SWITCH_CRRESULT(CRRES_ERR_WIN_UNKNOWN);
					CRIA_SWITCH_CRRESULT(CRRES_OK_WIN);

					CRIA_SWITCH_CRRESULT(CRRES_ERR_WIN_FAILED_TO_RETRIVE_DC);
					CRIA_SWITCH_CRRESULT(CRRES_ERR_WIN_FAILED_TO_CREATE_DC);
					CRIA_SWITCH_CRRESULT(CRRES_ERR_WIN_FAILED_TO_CREATE_HBMP);
					CRIA_SWITCH_CRRESULT(CRRES_ERR_WIN_INPUT_THREAD_BLOCKED);
					CRIA_SWITCH_CRRESULT(CRRES_ERR_WIN_SYSTEMPARMINFO_FAILED);
					CRIA_SWITCH_CRRESULT(CRRES_ERR_WIN_FAILED_TO_INSTALL_HOCK);
					CRIA_SWITCH_CRRESULT(CRRES_ERR_WIN_COULD_NOT_GET_KEY_LAYOUT);

					default:
						return String("CRRESULT_UNNAMED_RESULT");
				}
			case CRRES_ERR_PACO.Value:
				switch (result.Value) {
					CRIA_SWITCH_CRRESULT(CRRES_ERR_PACO);

					CRIA_SWITCH_CRRESULT(CRRES_ERR_PACO_IS_NOT_SUPPORTED);

					default:
						return String("CRRESULT_UNNAMED_RESULT");
				}
			case CRRES_SOURCE_CUDA.Value:
				switch (result.Value) {
					CRIA_SWITCH_CRRESULT(CRRES_ERR_CUDA_UNKNOWN);

					default:
						return String("CRRESULT_UNNAMED_RESULT");
				}
			default:
				return "CRRES_SOURCE_UNKNOWN";
		}
	}
}
#undef CRIA_SWITCH_CRRESULT
