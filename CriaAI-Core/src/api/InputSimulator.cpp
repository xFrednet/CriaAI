#include "InputSimulator.h"

#include "win/WinInputSimulator.h"

namespace cria_ai { namespace api {
	CRInputSimulator* CRInputSimulator::GetInstance(const String& target, crresult* result)
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
		if (result)
			*result = res;
		if (CR_FAILED(res))
		{
			delete inputSim;
			inputSim = nullptr;
		}
		/*
		 * target the selected window
		 */
		//inputSim->m_TargetWindowTitle.clear();
		inputSim->setNewWindowTarget(target);

		/*
		 * return
		 */
		return inputSim;
	}

	CRInputSimulator::CRInputSimulator()
		: m_TargetWindowTitle("")
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
	void CRInputSimulator::setNewWindowTarget(const String& newWindowTarget)
	{
		if (!m_TargetWindowTitle.empty() && m_TargetWindowTitle == newWindowTarget)
			return; /* the window title is the same */

		String oldTitle = m_TargetWindowTitle;
		m_TargetWindowTitle = newWindowTarget;

		newTargetWindowTitle(oldTitle);
		
		CRIA_INFO_PRINTF("The new target window hat the boundaries: X: %i, Y: %i, Width: %i, Height: %i (Title %s)",
			m_MouseBounderies.X, m_MouseBounderies.Y, m_MouseBounderies.Width, m_MouseBounderies.Height, m_TargetWindowTitle.c_str());

		setMouse(CR_VEC2I(m_MouseBounderies.Width / 2, m_MouseBounderies.Height / 2));
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
	* Mouse button interaction
	*/
	void CRInputSimulator::toggleButton(uint button)
	{
		if (button >= CR_MOUSE_MAX_BUTTON_COUNT)
			return;

		if (m_MouseButtonState[button])
			simulateButtonRelease(button);
		else
			simulateButtonPress(button);
	}
	void CRInputSimulator::clickButton(uint button)
	{
		if (button >= CR_MOUSE_MAX_BUTTON_COUNT)
			return;

		simulateButtonPress(button);
		if (CR_INPUTSIM_CLICK_TIME_MS == 0)
			simulateButtonRelease(button);
	}
	void CRInputSimulator::setButtonState(uint button, bool state)
	{
		if (button >= CR_MOUSE_MAX_BUTTON_COUNT)
			return;

		if (state)
			simulateButtonPress(button);
		else
			simulateButtonRelease(button);
	}
	bool CRInputSimulator::getButtonState(uint button)
	{
		if (button >= CR_MOUSE_MAX_BUTTON_COUNT)
			return false;

		return m_KeyBoardKeyStates[button];
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
		CR_CLAMP_VALUE(motion.X, -mousePos.X, (int)m_MouseBounderies.Width - mousePos.X - 1);
		CR_CLAMP_VALUE(motion.Y, -mousePos.Y, (int)m_MouseBounderies.Height - mousePos.Y - 1);

		simulateMouseMove(motion);
	}
	void CRInputSimulator::setMouse(CR_VEC2I pos)
	{
		CR_CLAMP_VALUE(pos.X, 0, (int)m_MouseBounderies.Width - 1);
		CR_CLAMP_VALUE(pos.Y, 0, (int)m_MouseBounderies.Height - 1);

		simulateMouseSet(pos);
	}
}}
