#pragma once

#include "../Common.hpp"

#include "InputUtil.h"
#include "Window.h"

#ifndef CR_INPUTSIM_CLICK_TIME
#	define CR_INPUTSIM_CLICK_TIME_MS             0
#endif

namespace cria_ai { namespace api {
	
	class CRInputSimulator
	{
	public:
		static CRInputSimulator* GetInstance(const CRWindowPtr& targetWindow = nullptr, crresult* result = nullptr);
	protected:
		/**
		 * \brief This arrays are maintained by the api specific sub class.
		 */
		bool m_KeyBoardKeyStates[CR_KEYBOARD_MAX_KEY_COUNT]{};
		/**
		* \brief This arrays are maintained by the api specific sub class. (at least they should be ;P)
		*/
		bool m_MouseButtonState[CR_MOUSE_MAX_BUTTON_COUNT]{};

		CRWindowPtr m_TargetWindow;

		/*
		 * Init
		 */
		CRInputSimulator();
		virtual crresult init() = 0;

		/*
		 * Virtual methods
		 */
		virtual crresult simulateKeyPress(uint key) = 0;
		virtual crresult simulateKeyRelease(uint key) = 0;

		virtual crresult simulateButtonPress(uint button) = 0;
		virtual crresult simulateButtonRelease(uint button) = 0;

		virtual crresult simulateMouseScroll(int amount) = 0;
		
		virtual crresult simulateMouseMove(CR_VEC2I motion) = 0;
		virtual crresult simulateMouseSet(CR_VEC2I pos) = 0;

		virtual crresult newTargetWindow();

	public:
		virtual ~CRInputSimulator();

		virtual void update();

		/*
		* Window target
		*/
		crresult setWindowTarget(CRWindowPtr targetWindow);

		/*
		 * keyboard interaction
		 */
		void toggleKey(uint key);
		void clickKey(uint key);
		void setKeyState(uint key, bool state);
		bool getKeyState(uint key);

		/*
		 * Mouse button interaction
		 */
		void toggleButton(uint buttonID);
		void clickButton(uint buttonID);
		void setButtonState(uint buttonID, bool state);
		bool getButtonState(uint buttonID);
		void scrollMouse(int amount);

		/*
		 * Mouse movement interaction
		 */
		void moveMouse(CR_VEC2I motion);
		void setMouse(CR_VEC2I pos);
		virtual CR_VEC2I getMousePos() const = 0;//TODO
	};

}}