#pragma once

#include "../Common.hpp"
#include "Window.h"

#define CR_SCREENCAP_CHANNEL_COUNT     4

namespace cria_ai { namespace api {

	class CRScreenCapturer 
	{
	public:
		static CRScreenCapturer* CreateInstance(CRWindowPtr target, crresult* result = nullptr);
		
		/*
		 * * These functions are API specific.
	     */
	protected:
		CRWindowPtr m_Target;
		CR_FLOAT_BITMAP* m_LastFrame;

		CRScreenCapturer();
		virtual crresult init(CRWindowPtr target) = 0;
		virtual crresult newTarget(CRWindowPtr target) = 0;
	public:
		virtual ~CRScreenCapturer();

		virtual crresult setTarget(CRWindowPtr target);
		virtual crresult grabFrame() = 0;

		CR_FLOAT_BITMAP* getLastFrame();
		CR_FLOAT_BITMAP const* getLastFrame() const;
	};

}}