#include "InputSimulator.h"

#include "win/WinInputSimulator.h"

namespace cria_ai { namespace api {
	CRInputSimulator* CRInputSimulator::GetInstance(const CRWindowPtr& targetWindow, crresult* result)
	{
		CRInputSimulator* inputSim = nullptr;

#ifdef CRIA_OS_WIN
		inputSim = new win::CRWinInputSimulator();
#endif

		/*
		 * Creation validation
		 */
		if (!inputSim)
		{
			if (result)
				*result = CRRES_ERR_NEW_FAILED;
			return nullptr;
		}

		/*
		 * init and error check
		 */
		crresult res = inputSim->init();
		if (CR_FAILED(res))
		{
			*result = res;
			
			delete inputSim;
			return inputSim;
		}
		/*
		 * target the selected window
		 */
		res = inputSim->setWindowTarget(targetWindow);
		if (CR_FAILED(res))
		{
			*result = res;

			delete inputSim;
			return inputSim;
		}

		/*
		 * return
		 */
		if (result)
			*result = CRRES_OK;
		return inputSim;
	}

	crresult CRInputSimulator::newTargetWindow()
	{
		return CRRES_OK;
	}
	CRInputSimulator::CRInputSimulator()
	{
		memset(m_KeyBoardKeyStates, (int)false, sizeof(bool) * CR_KEYBOARD_MAX_KEY_COUNT);
		memset(m_MouseButtonState, (int)false, sizeof(bool) * CR_MOUSE_MAX_BUTTON_COUNT);
	}

	CRInputSimulator::~CRInputSimulator()
	{
	}

	void CRInputSimulator::update()
	{
		//TODO Check is keys and buttons should be released
	}

	/*
	* Window target
	*/
	crresult CRInputSimulator::setWindowTarget(CRWindowPtr targetWindow)
	{
		crresult result;

		if (!targetWindow.get())
		{
			targetWindow = CRWindow::CreateDestopWindowInstance(&result);
		
			if (CR_FAILED(result))
				return result;
		}

		CR_RECT size = targetWindow->getClientArea();
		setMouse(CR_VEC2I(size.Width / 2, size.Height / 2));

		return CRRES_OK;
	}

	/*
	* keyboard interaction
	*/
	void CRInputSimulator::toggleKey(uint key)
	{
		if (key >= CR_KEYBOARD_MAX_KEY_COUNT)
			return;

		if (m_KeyBoardKeyStates[key])
			simulateKeyRelease(key);
		else
			simulateKeyPress(key);
	}
	void CRInputSimulator::clickKey(uint key)
	{
		if (key >= CR_KEYBOARD_MAX_KEY_COUNT)
			return;

		simulateKeyPress(key);
		if (CR_INPUTSIM_CLICK_TIME_MS == 0)
			simulateKeyRelease(key);
	}
	void CRInputSimulator::setKeyState(uint key, bool state)
	{
		if (key >= CR_KEYBOARD_MAX_KEY_COUNT)
			return;

		if (state)
			simulateKeyPress(key);
		else
			simulateKeyRelease(key);
	}
	bool CRInputSimulator::getKeyState(uint key)
	{
		if (key >= CR_KEYBOARD_MAX_KEY_COUNT)
			return false;

		return m_KeyBoardKeyStates[key];
	}

	/*
	* Mouse buttonID interaction
	*/
	void CRInputSimulator::toggleButton(uint buttonID)
	{
		if (buttonID >= CR_MOUSE_MAX_BUTTON_COUNT)
			return;

		if (m_MouseButtonState[buttonID])
			simulateButtonRelease(buttonID);
		else
			simulateButtonPress(buttonID);
	}
	void CRInputSimulator::clickButton(uint buttonID)
	{
		if (buttonID >= CR_MOUSE_MAX_BUTTON_COUNT)
			return;

		simulateButtonPress(buttonID);
		if (CR_INPUTSIM_CLICK_TIME_MS == 0)
			simulateButtonRelease(buttonID);
	}
	void CRInputSimulator::setButtonState(uint buttonID, bool state)
	{
		if (buttonID >= CR_MOUSE_MAX_BUTTON_COUNT)
			return;

		if (state)
			simulateButtonPress(buttonID);
		else
			simulateButtonRelease(buttonID);
	}
	bool CRInputSimulator::getButtonState(uint buttonID)
	{
		if (buttonID >= CR_MOUSE_MAX_BUTTON_COUNT)
			return false;

		return m_KeyBoardKeyStates[buttonID];
	}

	void CRInputSimulator::scrollMouse(int amount)
	{
		simulateMouseScroll(amount);
	}

	/*
	* Mouse movement interaction
	*/
	void CRInputSimulator::moveMouse(CR_VEC2I motion)
	{
		CR_VEC2I mousePos = getMousePos();
		CR_RECT bounderies = ((m_TargetWindow.get()) ? m_TargetWindow->getClientArea() : CR_RECT(0, 0, 0, 0));
		
		CR_CLAMP_VALUE(motion.X, -mousePos.X, (int)bounderies.Width - mousePos.X - 1);
		CR_CLAMP_VALUE(motion.Y, -mousePos.Y, (int)bounderies.Height - mousePos.Y - 1);

		simulateMouseMove(motion);
	}
	void CRInputSimulator::setMouse(CR_VEC2I pos)
	{
		CR_RECT bounderies = ((m_TargetWindow.get()) ? m_TargetWindow->getClientArea() : CR_RECT(0, 0, 0, 0));
		CR_CLAMP_VALUE(pos.X, 0, (int)bounderies.Width - 1);
		CR_CLAMP_VALUE(pos.Y, 0, (int)bounderies.Height - 1);

		simulateMouseSet(pos);
	}
}}
