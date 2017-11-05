#include "Window.hpp"

#if defined(_WIN32) || defined(_WIN64)
#	include "API\win\WinWindow.hpp"
#endif

namespace bmp_renderer {

	Window* Window::CreateInstance(char const* name, unsigned width, unsigned height, WINDOW_ON_EXIT_ACTION onExit)
	{
#if defined(_WIN32) || defined(_WIN64)
		return new api::WinWindow(name, width, height, onExit);
#endif
		return nullptr;
	}

	Window::Window(WINDOW_ON_EXIT_ACTION onExit)
		: m_OnExitAction(onExit)
	{
		
	}

	WINDOW_ON_EXIT_ACTION Window::getOnExitAction() const
	{
		return m_OnExitAction;
	}
	void Window::setOnExitAction(WINDOW_ON_EXIT_ACTION action)
	{
		m_OnExitAction = action;
	}

}
