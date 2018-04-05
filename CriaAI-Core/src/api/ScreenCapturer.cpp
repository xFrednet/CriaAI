#include "ScreenCapturer.h"

#include "win/WinScreenCapturer.h"

namespace cria_ai { namespace api {

	/*
	 * static CR_RECT GetClientArea(const String& windowTitle);
	 * static uint8 GetDisplayNo(const String& windowTitle);
	 * static uint8 GetDisplayCount();
	 * 
	 * These functions are defined inside API specific screen capturer .cpp file.
	 */

	CRScreenCapturer* CRScreenCapturer::CreateInstance(const CR_RECT& cArea, uint8 displayNo, crresult* result)
	{
		CRScreenCapturer* instance = nullptr;

		/*
		 * creating the instance
		 */
		instance = new win::CRWinScreenCapturer();

		/*
		 * init
		 */
		crresult initRes = instance->init(cArea, displayNo);
		if (result)
			*result = initRes;

		return instance;
	}

	CRScreenCapturer::CRScreenCapturer()
		: m_Area({0, 0, 0, 0}),
		m_LastFrame(nullptr)
	{
	}

	CRScreenCapturer::~CRScreenCapturer()
	{
		if (m_LastFrame)
			DeleteFBmp(m_LastFrame);
	}

	CR_FLOAT_BITMAP* CRScreenCapturer::getLastFrame()
	{
		return m_LastFrame;
	}
	CR_FLOAT_BITMAP const* CRScreenCapturer::getLastFrame() const
	{
		return m_LastFrame;
	}

}}