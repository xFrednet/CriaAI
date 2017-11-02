#pragma once

#include "Bitmap.hpp"

namespace bmp_renderer
{
	enum WINDOW_ON_EXIT_ACTION
	{
		WINDOW_ON_EXIT_DESTROY = 0,
		WINDOW_ON_EXIT_HIDE,
		WINDOW_ON_EXIT_MINIMIZE,
		WINDOW_ON_EXIT_DO_NOTHING 
	};

	class Window
	{
	public:
		static Window* CreateInstance(char const* name, unsigned width, unsigned height, WINDOW_ON_EXIT_ACTION onExit = WINDOW_ON_EXIT_DESTROY);

	protected:
		WINDOW_ON_EXIT_ACTION m_OnExitAction;
		
		Window(WINDOW_ON_EXIT_ACTION onExit);
	public:
		virtual ~Window() {/* bye bye */}

		virtual bool update() = 0;
		virtual void loadBitmap(const Bitmap bitmap) = 0;

		virtual void setVisibility(bool visible) = 0;
		virtual bool getVisibility() const = 0;

		virtual void destroy() = 0;
		virtual bool isValid() const = 0;

		WINDOW_ON_EXIT_ACTION getOnExitAction() const;
		void setOnExitAction(WINDOW_ON_EXIT_ACTION action);
	};
}
