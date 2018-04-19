#pragma once

#include "../Common.hpp"

#ifndef CR_INPUTSIM_CLICK_TIME
#	define CR_INPUTSIM_CLICK_TIME_MS             0
#endif

#define CR_KEYBOARD_MAX_KEY_COUNT      254
#define CR_MOUSE_MAX_BUTTON_COUNT      3

namespace cria_ai { namespace api {
	
	class CRInputSimulator
	{
	public:
		static CRInputSimulator* GetInstance(const String& target = "", crresult* result = nullptr);
	protected:
		String m_TargetWindowTitle;
		/**
		 * \brief This arrays are maintained by the api specific sub class.
		 */
		bool m_KeyBoardKeyStates[CR_KEYBOARD_MAX_KEY_COUNT]{};
		/**
		* \brief This arrays are maintained by the api specific sub class. (at least they should be ;P)
		*/
		bool m_MouseButtonState[CR_MOUSE_MAX_BUTTON_COUNT]{};
		/**
		 * \brief These client position is relative to the client window. 
		 * The offset are the X and Y coordinates of the m_MouseBounderies.
		 */
		CR_RECT m_MouseBounderies;

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

		virtual void newTargetWindowTitle(const String& oldTitle) = 0;

	public:
		virtual ~CRInputSimulator();

		virtual void update();

		/*
		* Window target
		*/
		/**
		 * \brief 
		 * 
		 * \param newWindowTarget 
		 * 
		 */
		void setNewWindowTarget(const String& newWindowTarget);

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
		void toggleButton(uint key);
		void clickButton(uint key);
		void setButtonState(uint button, bool state);
		bool getButtonState(uint key);
		void scrollMouse(int amount);

		/*
		 * Mouse movement interaction
		 */
		void moveMouse(CR_VEC2I motion);
		void setMouse(CR_VEC2I pos);
		virtual CR_VEC2I getMousePos() const = 0;//TODO
	};

}}