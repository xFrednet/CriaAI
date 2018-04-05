#pragma once

#include "../Common.hpp"

#define CR_SCREENCAP_CHANNEL_COUNT     4

namespace cria_ai { namespace api {

	class CRScreenCapturer 
	{
	public:
		static CRScreenCapturer* CreateInstance(const CR_RECT& cArea = {0, 0, 0, 0}, uint8 displayNo = 0, crresult* result = nullptr);
		
		/*
		 * * These functions are defined inside API specific screen capturer .cpp file.
	     */
		static CR_RECT GetClientArea(const String& windowTitle);
		static uint8 GetDisplayNo(const String& windowTitle);
		static uint8 GetDisplayCount();
	protected:
		CR_RECT m_Area;
		CR_FLOAT_BITMAP* m_LastFrame;

		CRScreenCapturer();
		virtual crresult init(const CR_RECT& cArea = {0, 0, 0, 0}, uint8 displayNo = 0) = 0;
	public:
		virtual ~CRScreenCapturer();

		virtual crresult updateRectangle(CR_RECT area = {0, 0, 0, 0}) = 0;
		virtual crresult grabFrame() = 0;

		CR_FLOAT_BITMAP* getLastFrame();
		CR_FLOAT_BITMAP const* getLastFrame() const;
	};

}}