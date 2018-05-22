#include "WinInputSimulator.h"
#include "../InputUtil.h"

#ifdef CRIA_OS_WIN
/*
 * The windows screen has a virtual size of 65536 x 65536 to support diagonal 
 * movement better etc. at least that is how I've understood it. Anyways it does
 * not matter why. This value is needed for the MOUSEEVENTF_ABSOLUTE flag.
 * 
 * My documentation is 10/10 as always ... sure... I will just keep telling
 * that to my self. ~xFrednet 2018
 */
#define CRIA_WIN_VIRTUAL_SCREEN_SIZE             65536

namespace cria_ai { namespace os { namespace win {
	
	bool GetMouseAccellState()
	{
		int mouseParams[3];
		memset(mouseParams, 0, sizeof(int) * 3);

		// Get the current values.
		SystemParametersInfo(SPI_GETMOUSE, 0, mouseParams, 0);

		return mouseParams[2] == 1;
	}
	crresult SetMouseAccellState(bool state)
	{
		int mouseParams[3];

		// Get the current values.
		if (!SystemParametersInfo(SPI_GETMOUSE, 0, mouseParams, 0))
			return CRRES_ERR_WIN_SYSTEMPARMINFO_FAILED;

		// Modify the acceleration value as directed.
		mouseParams[2] = ((state) ? 1 : 0);

		// Update the system setting.
		if (!SystemParametersInfo(SPI_SETMOUSE, 0, mouseParams, SPIF_SENDCHANGE))
			return CRRES_ERR_WIN_SYSTEMPARMINFO_FAILED;

		return CRRES_OK;
	}
	
	//TODO the "source engine] "arma" and probably some other engines do not work with this virtual input method 
	crresult CRWinInputSimulator::sendInputMessage(INPUT* message) const
	{
		if (m_TargetWindow->isFocussed())
			return CRRES_OK_OS_INPUTSIM_TARGET_NOT_FOCUSED;

		CR_RECT mBounds = m_TargetWindow->getClientArea();
		CR_VEC2I pos = getMousePos();
		if (pos.X < 0 || pos.X >= (int)mBounds.Width ||
			pos.Y < 0 || pos.Y >= (int)mBounds.Height)
			return CRRES_OK_OS_INPUTSIM_CURSOR_OUTSIDE;

		if (!SendInput(1, message, sizeof(INPUT)))
			return CRRES_ERR_WIN_INPUT_THREAD_BLOCKED;

		return CRRES_OK_OS;
	}

	CRWinInputSimulator::CRWinInputSimulator()
		: m_OriginalMouseAccellState(true),
		m_MouseSetMultiplayer(0.0f, 0.0f)
	{
	}

	CRWinInputSimulator::~CRWinInputSimulator()
	{
		if (m_OriginalMouseAccellState)
			SetMouseAccellState(m_OriginalMouseAccellState);
	}

	crresult CRWinInputSimulator::init()
	{
		/*
		 * Disable windows mouse movement thing, it's not fun
		 */
		m_OriginalMouseAccellState = GetMouseAccellState();
		if (CR_FAILED(SetMouseAccellState(false)))
		{
			return CRRES_ERR_OS_INPUTSIM_INIT_FAILED;
		}

		/*
		 * m_MouseSetMultiplayer
		 */
		m_MouseSetMultiplayer.X = ((float)CRIA_WIN_VIRTUAL_SCREEN_SIZE / (float)GetSystemMetrics(SM_CXSCREEN));
		m_MouseSetMultiplayer.Y = ((float)CRIA_WIN_VIRTUAL_SCREEN_SIZE / (float)GetSystemMetrics(SM_CYSCREEN));
		
		/*
		 * return
		 */
		return CRRES_OK;
	}

	/*
	* keyboard interaction
	*/
	crresult CRWinInputSimulator::simulateKeyPress(uint key)
	{
		INPUT input;
		memset(&input, 0, sizeof(INPUT));
		input.type = INPUT_KEYBOARD;
		
		input.ki.wVk         = key;
		input.ki.dwFlags     = 0;
		
		return sendInputMessage(&input);
	}
	crresult CRWinInputSimulator::simulateKeyRelease(uint key)
	{
		INPUT input;
		memset(&input, 0, sizeof(INPUT));
		input.type = INPUT_KEYBOARD;

		input.ki.wVk         = key;
		input.ki.dwFlags     = KEYEVENTF_KEYUP;

		return sendInputMessage(&input);
	}

	/*
	* Mouse button interaction
	*/
	crresult CRWinInputSimulator::simulateButtonPress(uint button)
	{
		INPUT input;
		memset(&input, 0, sizeof(INPUT));
		input.type = INPUT_MOUSE;
		
		switch (button)
		{
			case CR_MBUTTON_LEFT:
				input.mi.dwFlags = MOUSEEVENTF_LEFTDOWN;
				break;
			case CR_MBUTTON_MIDDLE:
				input.mi.dwFlags = MOUSEEVENTF_MIDDLEDOWN;
				break;
			case CR_MBUTTON_RIGHT:
				input.mi.dwFlags = MOUSEEVENTF_RIGHTDOWN;
				break;
			default:
				return  CRRES_ERR_OS_BUTTON_OUT_OF_BOUNDS;
		}

		return sendInputMessage(&input);
	}
	crresult CRWinInputSimulator::simulateButtonRelease(uint button)
	{
		INPUT input;
		memset(&input, 0, sizeof(INPUT));
		input.type = INPUT_MOUSE;

		switch (button) {
			case CR_MBUTTON_LEFT:
				input.mi.dwFlags = MOUSEEVENTF_LEFTUP;
				break;
			case CR_MBUTTON_MIDDLE:
				input.mi.dwFlags = MOUSEEVENTF_MIDDLEUP;
				break;
			case CR_MBUTTON_RIGHT:
				input.mi.dwFlags = MOUSEEVENTF_RIGHTUP;
				break;
			default:
				return  CRRES_ERR_OS_BUTTON_OUT_OF_BOUNDS;
		}

		return sendInputMessage(&input);
	}

	/*
	* Mouse movement interaction
	*/
	crresult CRWinInputSimulator::simulateMouseMove(CR_VEC2I motion)
	{
		INPUT input;
		memset(&input, 0, sizeof(INPUT));
		input.type = INPUT_MOUSE;

		input.mi.dx      = motion.X;
		input.mi.dy      = motion.Y;
		input.mi.dwFlags = MOUSEEVENTF_MOVE;

		return sendInputMessage(&input);
	}
	crresult CRWinInputSimulator::simulateMouseSet(CR_VEC2I pos)
	{
		INPUT input;
		memset(&input, 0, sizeof(INPUT));
		input.type = INPUT_MOUSE;

		CR_VEC2I windowPos = m_TargetWindow->getClientArea().Pos;
		input.mi.dx = (int)((float)(pos.X + windowPos.X) * m_MouseSetMultiplayer.X);
		input.mi.dy = (int)((float)(pos.Y + windowPos.Y) * m_MouseSetMultiplayer.Y);
		input.mi.dwFlags = MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE;

		return sendInputMessage(&input);
	}

	crresult CRWinInputSimulator::simulateMouseScroll(int amount)
	{
		INPUT input;
		memset(&input, 0, sizeof(INPUT));
		input.type = INPUT_MOUSE;

		input.mi.mouseData = amount * WHEEL_DELTA;
		input.mi.dwFlags   = MOUSEEVENTF_WHEEL;

		return sendInputMessage(&input);
	}

	CR_VEC2I CRWinInputSimulator::getMousePos() const
	{
		POINT pos;
		GetCursorPos(&pos);

		return (CR_VEC2I(pos.x, pos.y) - m_TargetWindow->getClientArea().Pos);
	}
}}}

#endif