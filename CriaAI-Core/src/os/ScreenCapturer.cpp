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
			CRDeleteFBmp(m_LastFrame);
	}

	crresult CRScreenCapturer::setTarget(CRWindowPtr target)
	{
		if (!target.get())
			return CRRES_ERR_API_TARGET_IS_NULL;

		/*
		 * Create new bmp
		 */
		CR_RECT tSize = target->getClientArea();
		CR_FLOAT_BITMAP* newBmp = CRCreateFBmp(tSize.Width, tSize.Height, CR_SCREENCAP_CHANNEL_COUNT);
		CR_FLOAT_BITMAP* oldBmp = m_LastFrame;
		
		/*
		 * updating class members
		 */
		m_Target = target;
		m_LastFrame = newBmp;
		CRDeleteFBmp(oldBmp);

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
