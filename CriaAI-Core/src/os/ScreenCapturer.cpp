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
		crresult res = instance->init();
		if (result)
			*result = res;

		if (CR_FAILED(res))
		{
			delete instance;
			return nullptr;
		}

		/*
		 * set target
		 */
		res = instance->setTarget(target);
		if (result)
			*result = res;

		if (CR_FAILED(res)) {
			delete instance;
			return nullptr;
		}

		return instance;
	}

	CRScreenCapturer::CRScreenCapturer()
		: m_Frame(nullptr),
		m_FrameSize(0, 0, 0, 0),
		m_ContinueCapture(false)
	{
	}

	CRScreenCapturer::~CRScreenCapturer()
	{
		if (isCaptureThreadRunnning())
			stopCaptureThread();

		m_FrameLock.lock();
		if (m_Frame) {
			CRDeleteFBmpNormal(m_Frame);
			m_Frame = nullptr;
		}
		m_FrameLock.unlock();
	}

	crresult CRScreenCapturer::setTarget(CRWindowPtr& target)
	{
		if (!target.get())
			return CRRES_ERR_OS_TARGET_IS_NULL;

		CR_RECT area = target->getClientArea();
		/*
		 * Create new bmp
		 */
		m_FrameLock.lock();
		// delete old frame
		if (m_Frame)
		{
			CRDeleteFBmpNormal(m_Frame);
			m_Frame = nullptr;
		}
		// Frame size
		m_FrameSize.Width = area.Width;
		m_FrameSize.Height = area.Height;

		//update m_Target
		m_Target = target;

		//unlock
		m_FrameLock.unlock();

		/*
		 * return
		 */
		return newTarget(target);
	}

	/*
	 * Capture thread
	 */
	crresult CRScreenCapturer::runCaptureThread()
	{
		if (isCaptureThreadRunnning())
			return CRRES_OK_OS_THREAD_IS_ALLREADY_RUNNING;

		m_ContinueCapture = true;
		m_CaptureThread   = std::thread([this]()
		{
			crresult result;
			while (this->m_ContinueCapture) {
				result = this->grabFrame();
				
				if (CR_FAILED(result))
				{
					CRIA_ALERT_PRINTF("CRScreenCapturer: this->grabFrame() failed: %16i", result.Value);
					
					if (CR_SCREENCAP_THREAD_BREAK_ON_FAIL)
					{
						break;
					}
				}
			}
		});

		CRIA_INFO_PRINTF("CRScreenCapturer::runCaptureThread: started Thread!");

		return CRRES_OK;
	}
	crresult CRScreenCapturer::stopCaptureThread()
	{
		if (!isCaptureThreadRunnning())
			return CRRES_OK_OS_THREAD_IS_ALLREADY_JOINED;

		m_ContinueCapture = false;
		m_CaptureThread.join();

		return CRRES_OK;
	}

	bool CRScreenCapturer::isCaptureThreadRunnning() const
	{
		return m_ContinueCapture;
	}

	CR_FLOAT_BITMAP* CRScreenCapturer::getFrame()
	{
		m_FrameLock.lock();
		CR_FLOAT_BITMAP* frame = m_Frame;
		m_Frame = nullptr;
		m_FrameLock.unlock();

		return frame;
	}

}}
