#include "InputUtil.h"

#ifdef CRIA_OS_WIN

#include "win/WinOSContext.h"

namespace cria_ai
{
	CR_KEY_ID CRGetCRKeyIDFromWinAsciiKeyID(const uint8 winKeyID)
	{
		switch (winKeyID)
		{
			case VK_LSHIFT:
			case VK_RSHIFT:
				return CR_KEY_SHIFT;
			case VK_LCONTROL:
			case VK_RCONTROL:
				return CR_KEY_CONTROL;
			case VK_LMENU:
			case VK_RMENU:
				return CR_KEY_MENU;
			case VK_LWIN:
			case VK_RWIN:
				return CR_KEY_WIN;
			default:
				return (CR_KEY_ID)winKeyID; // The rest of the key IDs are the same
		}
	}
}
#endif //CRIA_OS_WIN
