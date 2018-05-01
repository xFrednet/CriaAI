#pragma once

#include "../Common.hpp"

//TODO conversion functions from os to cr_key_id's

#define CR_KEYBOARD_MAX_KEY_COUNT      254
#define CR_MOUSE_MAX_BUTTON_COUNT      3

/*
 * Keyboard util
 */
namespace cria_ai {
	enum CR_KEY_ID {

		CR_KEY_ID_FIRST                = 0x08,
		
		CR_KEY_BACK                    = 0x08,
		CR_KEY_TAB                     = 0x09,
 		
		CR_KEY_CLEAR                   = 0x0C,
		CR_KEY_RETURN                  = 0x0D,
		
		CR_KEY_SHIFT                   = 0x10,
		CR_KEY_CONTROL                 = 0x11,
		CR_KEY_MENU                    = 0x12,
		CR_KEY_PAUSE                   = 0x13,
		CR_KEY_CAPITAL                 = 0x14,
 		
		CR_KEY_ESCAPE                  = 0x1B,
		
		CR_KEY_CONVERT                 = 0x1C,
		CR_KEY_NONCONVERT              = 0x1D,
		CR_KEY_ACCEPT                  = 0x1E,
		CR_KEY_MODECHANGE              = 0x1F,
		
		CR_KEY_SPACE                   = 0x20,
		CR_KEY_PRIOR                   = 0x21,
		CR_KEY_NEXT                    = 0x22,
		CR_KEY_END                     = 0x23,
		CR_KEY_HOME                    = 0x24,
		
		/*
		 *    [^]
		 * [<][v][>]
		 */
		CR_KEY_LEFT                    = 0x25,
		CR_KEY_UP                      = 0x26,
		CR_KEY_RIGHT                   = 0x27,
		CR_KEY_DOWN                    = 0x28,

		CR_KEY_SELECT                  = 0x29,
		CR_KEY_PRINT                   = 0x2A,
		CR_KEY_EXECUTE                 = 0x2B,
		CR_KEY_SNAPSHOT                = 0x2C,
		CR_KEY_INSERT                  = 0x2D,
		CR_KEY_DELETE                  = 0x2E,
		CR_KEY_HELP                    = 0x2F,

		/*
		* CR_KEY_0 - CR_KEY_9 are the same as ASCII '0' - '9' (0x30 - 0x39)
		* 0x40 : unassigned
		* CR_KEY_A - CR_KEY_Z are the same as ASCII 'A' - 'Z' (0x41 - 0x5A)
		*/
		CR_KEY_0                       = 0x30,
		CR_KEY_1                       = 0x31,
		CR_KEY_2                       = 0x32,
		CR_KEY_3                       = 0x33,
		CR_KEY_4                       = 0x34,
		CR_KEY_5                       = 0x35,
		CR_KEY_6                       = 0x36,
		CR_KEY_7                       = 0x37,
		CR_KEY_8                       = 0x38,
		CR_KEY_9                       = 0x39,
		
		CR_KEY_A                       = 0x41,
		CR_KEY_B                       = 0x42,
		CR_KEY_C                       = 0x43,
		CR_KEY_D                       = 0x44,
		CR_KEY_E                       = 0x45,
		CR_KEY_F                       = 0x46,
		CR_KEY_G                       = 0x47,
		CR_KEY_H                       = 0x48,
		CR_KEY_I                       = 0x49,
		CR_KEY_J                       = 0x4A,
		CR_KEY_K                       = 0x4B,
		CR_KEY_L                       = 0x4C,
		CR_KEY_M                       = 0x4D,
		CR_KEY_N                       = 0x4E,
		CR_KEY_O                       = 0x4F,
		CR_KEY_P                       = 0x50,
		CR_KEY_Q                       = 0x51,
		CR_KEY_R                       = 0x52,
		CR_KEY_S                       = 0x53,
		CR_KEY_T                       = 0x54,
		CR_KEY_U                       = 0x55,
		CR_KEY_V                       = 0x56,
		CR_KEY_W                       = 0x57,
		CR_KEY_X                       = 0x58,
		CR_KEY_Y                       = 0x59,
		CR_KEY_Z                       = 0x5A,

		CR_KEY_WIN                     = 0x5B,
		CR_KEY_APPS                    = 0x5D,

		/*
		* 0x5E : reserved
		*/
		CR_KEY_NUMPAD0                 = 0x60,
		CR_KEY_NUMPAD1                 = 0x61,
		CR_KEY_NUMPAD2                 = 0x62,
		CR_KEY_NUMPAD3                 = 0x63,
		CR_KEY_NUMPAD4                 = 0x64,
		CR_KEY_NUMPAD5                 = 0x65,
		CR_KEY_NUMPAD6                 = 0x66,
		CR_KEY_NUMPAD7                 = 0x67,
		CR_KEY_NUMPAD8                 = 0x68,
		CR_KEY_NUMPAD9                 = 0x69,
		CR_KEY_MULTIPLY                = 0x6A,
		CR_KEY_ADD                     = 0x6B,
		CR_KEY_SEPARATOR               = 0x6C,
		CR_KEY_SUBTRACT                = 0x6D,
		CR_KEY_DECIMAL                 = 0x6E,
		CR_KEY_DIVIDE                  = 0x6F,
		CR_KEY_F1                      = 0x70,
		CR_KEY_F2                      = 0x71,
		CR_KEY_F3                      = 0x72,
		CR_KEY_F4                      = 0x73,
		CR_KEY_F5                      = 0x74,
		CR_KEY_F6                      = 0x75,
		CR_KEY_F7                      = 0x76,
		CR_KEY_F8                      = 0x77,
		CR_KEY_F9                      = 0x78,
		CR_KEY_F10                     = 0x79,
		CR_KEY_F11                     = 0x7A,
		CR_KEY_F12                     = 0x7B,
		CR_KEY_F13                     = 0x7C,
		CR_KEY_F14                     = 0x7D,
		CR_KEY_F15                     = 0x7E,
		CR_KEY_F16                     = 0x7F,
		CR_KEY_F17                     = 0x80,
		CR_KEY_F18                     = 0x81,
		CR_KEY_F19                     = 0x82,
		CR_KEY_F20                     = 0x83,
		CR_KEY_F21                     = 0x84,
		CR_KEY_F22                     = 0x85,
		CR_KEY_F23                     = 0x86,
		CR_KEY_F24                     = 0x87,

		CR_KEY_ID_LAST                 = 0x87
	};

	CR_KEY_ID CRGetCRKeyIDFromWinAsciiKeyID(const uint8 winKeyID);

	inline String CRGetKeyIDName(const CR_KEY_ID& keyID)
	{
		switch (keyID) {
			case CR_KEY_BACK:
				return "[BACK]";
			case CR_KEY_TAB:
				return "[TAB]";


			case CR_KEY_CLEAR:
				return "[CLEAR]";
			case CR_KEY_RETURN:
				return "[ENTER]";


			case CR_KEY_SHIFT: 
				return "[SHIFT]";
			case CR_KEY_CONTROL:
				return "[LCONTROL]";
			case CR_KEY_MENU:
				return "[MENU]";
			case CR_KEY_PAUSE:
				return "[PAUSE]";
			case CR_KEY_CAPITAL:
				return "[CAPITAL]";


			case CR_KEY_ESCAPE:
				return "[ESCAPE]";

		
			case CR_KEY_CONVERT:
				return "[CONVERT]";
			case CR_KEY_NONCONVERT:
				return "[NONCONVERT]";
			case CR_KEY_ACCEPT:
				return "[ACCEPT]";
			case CR_KEY_MODECHANGE:
				return "[MODECHANGE]";
			
			
			case CR_KEY_SPACE:
				return "[SPACE]";
			case CR_KEY_PRIOR:
				return "[PRIOR]";
			case CR_KEY_NEXT:
				return "[NEXT]";
			case CR_KEY_END:
				return "[END]";
			case CR_KEY_HOME:
				return "[HOME]";

				/*
				*    [^]
				* [<][v][>]
				*/
			case CR_KEY_LEFT:
				return "[LEFT]";
			case CR_KEY_UP:
				return "[UP]";
			case CR_KEY_RIGHT:
				return "[RIGHT]";
			case CR_KEY_DOWN:
				return "[DOWN]";


			case CR_KEY_SELECT:
				return "[SELECT]";
			case CR_KEY_PRINT:
				return "[PRINT]";
			case CR_KEY_EXECUTE:
				return "[EXECUTE]";
			case CR_KEY_SNAPSHOT:
				return "[SNAPSHOT]";
			case CR_KEY_INSERT:
				return "[INSERT]";
			case CR_KEY_DELETE:
				return "[DELETE]";
			case CR_KEY_HELP:
				return "[HELP]";


			case CR_KEY_NUMPAD0:
				return "[NUMPAD0]";
			case CR_KEY_NUMPAD1:
				return "[NUMPAD1]";
			case CR_KEY_NUMPAD2:
				return "[NUMPAD2]";
			case CR_KEY_NUMPAD3:
				return "[NUMPAD3]";
			case CR_KEY_NUMPAD4:
				return "[NUMPAD4]";
			case CR_KEY_NUMPAD5:
				return "[NUMPAD5]";
			case CR_KEY_NUMPAD6:
				return "[NUMPAD6]";
			case CR_KEY_NUMPAD7:
				return "[NUMPAD7]";
			case CR_KEY_NUMPAD8:
				return "[NUMPAD8]";
			case CR_KEY_NUMPAD9:
				return "[NUMPAD9]";
			case CR_KEY_MULTIPLY:
				return "[MULTIPLY]";
			case CR_KEY_ADD:
				return "[ADD]";
			case CR_KEY_SEPARATOR:
				return "[SEPARATOR]";
			case CR_KEY_SUBTRACT:
				return "[SUBTRACT]";
			case CR_KEY_DECIMAL:
				return "[DECIMAL]";
			case CR_KEY_DIVIDE:
				return "[DIVIDE]";


			case CR_KEY_WIN:
				return "[WIN]";
			case CR_KEY_APPS:
				return "[APPS]";


			case CR_KEY_F1:
				return "[F1]";
			case CR_KEY_F2:
				return "[F2]";
			case CR_KEY_F3:
				return "[F3]";
			case CR_KEY_F4:
				return "[F4]";
			case CR_KEY_F5:
				return "[F5]";
			case CR_KEY_F6:
				return "[F6]";
			case CR_KEY_F7:
				return "[F7]";
			case CR_KEY_F8:
				return "[F8]";
			case CR_KEY_F9:
				return "[F9]";
			case CR_KEY_F10:
				return "[F10]";
			case CR_KEY_F11:
				return "[F11]";
			case CR_KEY_F12:
				return "[F12]";
			case CR_KEY_F13:
				return "[F13]";
			case CR_KEY_F14:
				return "[F14]";
			case CR_KEY_F15:
				return "[F15]";
			case CR_KEY_F16:
				return "[F16]";
			case CR_KEY_F17:
				return "[F17]";
			case CR_KEY_F18:
				return "[F18]";
			case CR_KEY_F19:
				return "[F19]";
			case CR_KEY_F20:
				return "[F20]";
			case CR_KEY_F21:
				return "[F21]";
			case CR_KEY_F22:
				return "[F22]";
			case CR_KEY_F23:
				return "[F23]";
			case CR_KEY_F24:
				return "[F24]";


			default:
				if ((keyID >= CR_KEY_0 && keyID <= CR_KEY_9) || 
					(keyID >= CR_KEY_A && keyID <= CR_KEY_Z))
					return String() + (char)keyID;

				return "[KEY_ID: " + std::to_string(keyID) + "]";
		}
	}
}

/*
 * Mouse util
 */
namespace cria_ai
{
	enum CR_MBUTTON_ID {

		CR_MBUTTON_ID_FIRST  = 0,

		CR_MBUTTON_LEFT      = 0,
		CR_MBUTTON_1         = 0,

		CR_MBUTTON_MIDDLE    = 1,
		CR_MBUTTON_2         = 1,

		CR_MBUTTON_RIGHT     = 2,
		CR_MBUTTON_3         = 2,

		CR_MBUTTON_ID_LAST   = 2

	};
}