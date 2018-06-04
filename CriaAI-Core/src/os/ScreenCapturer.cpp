#include "ScreenCapturer.h"

#include "win/WinScreenCapturer.h"

namespace cria_ai { namespace os {

	CRScreenCapturer* CRScreenCapturer::CreateInstance(CRWindowPtr target, crresult* result)
	{
		CRScreenCapturer* instance = nullptr;

		/*
		 * creating the instance
		 */
		instance = new win::CRWinScreenCapturer();

		/*
		 * init
		 */
		crresult initRes = instance->init(target);
		if (result)
			*result = initRes;

		return instance;
	}

	CRScreenCapturer::CRScreenCapturer()
		: m_LastFrame(nullptr)
	{
	}

	CRScreenCapturer::~CRScreenCapturer()
	{
		if (m_LastFrame)
		{
			CRDeleteFBmp(m_LastFrame);
			m_LastFrame = nullptr;
		}
	}

	crresult CRScreenCapturer::setTarget(CRWindowPtr target)
	{
		if (!target.get())
			return CRRES_ERR_OS_TARGET_IS_NULL;

		/*
		 * Create new bmp
		 */
		CR_RECT tSize = target->getClientArea();
		if (m_LastFrame)
		{
			CR_FLOAT_BITMAP* oldBmp = m_LastFrame;
			m_LastFrame = nullptr;
			CRDeleteFBmpNormal(oldBmp);
		}
		
		/*
		 * updating class members
		 */
		m_Target = target;
		m_LastFrame = CRCreateFBmpNormal(tSize.Width, tSize.Height, CR_SCREENCAP_CHANNEL_COUNT);
		// if I use CRCreateFBmp it always crashes for some weired reason. but it works with malloc this means CRCreateFBmpNormal

		/*
		 * return
		 */
		return newTarget(target);
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
