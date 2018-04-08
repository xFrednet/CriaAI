#pragma once

#include "../Types.hpp"

#define CR_SUCCESS(crresult)           ((crresult).Value >= 0)
#define CR_FAILED(crresult)            ((crresult).Value < 0 )

// <Name>
//      ZA_RESULT
//  
// <Structure>
//       1 1 1 1 1 1 1
//       6 5 4 3 2 1 0 9 8 7 6 5 4 3 2 1
//      +-+-------------+---------------+
//      |R|---Source----|-----Code------|
//      +-+-------------+---------------+
//    
//      R      - Result: 
//                 0: success
//                 1: error
//      
//      Source - Indicates the source (128 possible sources)
//      
//      Code   - The error code (256 possible codes)
//
/**
 * \brief 
 * 
 * Structure:
 * 
 *       1 1 1 1 1 1 1
 *       6 5 4 3 2 1 0 9 8 7 6 5 4 3 2 1
 *      +-------+-------+---------------+
 *      |---R---|-Source|-----Code------|
 *      +-------+-------+---------------+
 *    
 *      R      - Result: 
 *                 0: success
 *                 1: error
 *      
 *      Source - Indicates the source (15 possible sources)
 *      
 *      Code   - The error code (256 possible codes)
 * 
 */
namespace cria_ai
{
	
	typedef struct crresult_ {
		int32_t Value;

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
#define CRIA_RES_TYPEDEF(value)                  (cria_ai::crresult{value})

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
#define CRRES_SOURCE_API                         CRIA_RES_TYPEDEF(0x0A00)
#define CRRES_SOURCE_CUDA                        CRIA_RES_TYPEDEF(0x0B00)
#define CRRES_SOURCE_WIN                         CRIA_RES_TYPEDEF(0x0C00)

/* //////////////////////////////////////////////////////////////////////////////// */
// // ZA_RESULT_SOURCE_NO_SOURCE //
/* //////////////////////////////////////////////////////////////////////////////// */
#define CRRES_ERR_INVALUD_ARGUMENTS              CRIA_RES_TYPEDEF(0x8001)
#define CRRES_ERR_FUNCTION_NOT_IMPLEMENTED       CRIA_RES_TYPEDEF(0x8002)
#define CRRES_ERR_INVALID_DIMENSIONS             CRIA_RES_TYPEDEF(0x8003)
#define CRRES_ERR_MISSING_INFORMATION            CRIA_RES_TYPEDEF(0x8004)
// Yes the original name should be CRRES_ERR_YAY_MULTI_THREADED_TIMING but you know multi threading
#define CRRES_ERR_TIMING_THREADED_YAY_MULTI      CRIA_RES_TYPEDEF(0x8005)
#define CRRES_ERR_MALLOC_FAILED                  CRIA_RES_TYPEDEF(0x8006)
#define CRRES_ERR_NEW_FAILED                     CRIA_RES_TYPEDEF(0x8007)

/* //////////////////////////////////////////////////////////////////////////////// */
// // CRRES_SOURCE_API //
/* //////////////////////////////////////////////////////////////////////////////// */
#define CRRES_ERR_API_UNKNOWN                    CRIA_RES_TYPEDEF(0x8A00)
#define CRRES_OK_API                             CRIA_RES_TYPEDEF(0x0A00)

#define CRRES_OK_API_INPUTSIM_TARGET_NOT_FOCUSED CRIA_RES_TYPEDEF(0x0A01)
#define CRRES_OK_API_INPUTSIM_CURSOR_OUTSIDE     CRIA_RES_TYPEDEF(0x0A02)

#define CRRES_ERR_API_KEY_OUT_OF_BOUNDS          CRIA_RES_TYPEDEF(0x8A01)
#define CRRES_ERR_API_BUTTON_OUT_OF_BOUNDS       CRIA_RES_TYPEDEF(0x8A02)
#define CRRES_ERR_API_INPUTSIM_INIT_FAILED       CRIA_RES_TYPEDEF(0x8A03)

/* //////////////////////////////////////////////////////////////////////////////// */
// // CRRES_SOURCE_UTILS //
/* //////////////////////////////////////////////////////////////////////////////// */
#define CRRES_ERR_UTILS_UNKNOWN                  CRIA_RES_TYPEDEF(0x8100)

#define CRRES_ERR_UTILS_FAILED_TO_CREATE_FBMP    CRIA_RES_TYPEDEF(0x8101)

/* //////////////////////////////////////////////////////////////////////////////// */
// // CRRES_SOURCE_CUDA //
/* //////////////////////////////////////////////////////////////////////////////// */
#define CRRES_ERR_CUDA_UNKNOWN                   CRIA_RES_TYPEDEF(0x8B00)

/* //////////////////////////////////////////////////////////////////////////////// */
// // CRRES_SOURCE_WIN //
/* //////////////////////////////////////////////////////////////////////////////// */
#define CRRES_ERR_WIN_UNKNOWN                    CRIA_RES_TYPEDEF(0x8C00)
#define CRRES_OK_WIN                             CRIA_RES_TYPEDEF(0x0C00)

#define CRRES_ERR_WIN_FAILED_TO_RETRIVE_DC       CRIA_RES_TYPEDEF(0x8C01)
#define CRRES_ERR_WIN_FAILED_TO_CREATE_DC        CRIA_RES_TYPEDEF(0x8C02)
#define CRRES_ERR_WIN_FAILED_TO_CREATE_HBMP      CRIA_RES_TYPEDEF(0x8C03)
#define CRRES_ERR_WIN_INPUT_THREAD_BLOCKED       CRIA_RES_TYPEDEF(0x8C04)
#define CRRES_ERR_WIN_SYSTEMPARMINFO_FAILED      CRIA_RES_TYPEDEF(0x8C05)

/* //////////////////////////////////////////////////////////////////////////////// */
// // GetCRResultName //
/* //////////////////////////////////////////////////////////////////////////////// */
#define CRIA_SWITCH_CRRESULT(macro)              case macro.Value: return String(#macro);
namespace cria_ai {

	inline String GetCRResultName(const crresult& result)
	{
		switch (result.Value & CRRES_MASK_SOURCE)
		{
			case CRRES_SOURCE_UNKNOWN.Value:
				switch (result.Value)
				{
					CRIA_SWITCH_CRRESULT(CRRES_OK);
					CRIA_SWITCH_CRRESULT(CRRES_ERROR);

					CRIA_SWITCH_CRRESULT(CRRES_ERR_INVALUD_ARGUMENTS);
					CRIA_SWITCH_CRRESULT(CRRES_ERR_FUNCTION_NOT_IMPLEMENTED);
					CRIA_SWITCH_CRRESULT(CRRES_ERR_INVALID_DIMENSIONS);
					CRIA_SWITCH_CRRESULT(CRRES_ERR_MISSING_INFORMATION);
					CRIA_SWITCH_CRRESULT(CRRES_ERR_TIMING_THREADED_YAY_MULTI);
					CRIA_SWITCH_CRRESULT(CRRES_ERR_MALLOC_FAILED);
					CRIA_SWITCH_CRRESULT(CRRES_ERR_NEW_FAILED);

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
			case CRRES_SOURCE_API.Value:
				switch (result.Value) {
					CRIA_SWITCH_CRRESULT(CRRES_ERR_API_UNKNOWN);
					CRIA_SWITCH_CRRESULT(CRRES_OK_API);

					CRIA_SWITCH_CRRESULT(CRRES_OK_API_INPUTSIM_TARGET_NOT_FOCUSED); 
					CRIA_SWITCH_CRRESULT(CRRES_OK_API_INPUTSIM_CURSOR_OUTSIDE);

					CRIA_SWITCH_CRRESULT(CRRES_ERR_API_KEY_OUT_OF_BOUNDS);
					CRIA_SWITCH_CRRESULT(CRRES_ERR_API_BUTTON_OUT_OF_BOUNDS);
					CRIA_SWITCH_CRRESULT(CRRES_ERR_API_INPUTSIM_INIT_FAILED);

					default:
						return String("CRRESULT_UNNAMED_RESULT");
				}
			case CRRES_SOURCE_CUDA.Value:
				switch (result.Value) {
					CRIA_SWITCH_CRRESULT(CRRES_ERR_CUDA_UNKNOWN);

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

					default:
						return String("CRRESULT_UNNAMED_RESULT");
				}
			default:
				return "CRRES_SOURCE_UNKNOWN";
		}
	}
}
#undef CRIA_SWITCH_CRRESULT
