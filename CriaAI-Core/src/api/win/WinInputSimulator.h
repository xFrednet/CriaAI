#pragma once

#include "../../Common.hpp"
#include "../InputSimulator.h"

#ifdef CRIA_OS_WIN

#include "WinOSContext.h"

namespace cria_ai { namespace api { namespace win {
	
	class CRWinInputSimulator : public CRInputSimulator
	{
		friend class CRInputSimulator;
	private:
		bool m_OriginalMouseAccellState;
		CR_VEC2F m_MouseSetMultiplayer;

		crresult sendInputMessage(INPUT* message) const;
	
	protected:
		CRWinInputSimulator();
	public:
		~CRWinInputSimulator();

	protected:
		crresult init() override;
		
		crresult simulateKeyPress(uint key) override;
		crresult simulateKeyRelease(uint key) override;
		
		crresult simulateButtonPress(uint button) override;
		crresult simulateButtonRelease(uint button) override;
		
		crresult simulateMouseScroll(int amount) override;
		
		crresult simulateMouseMove(CR_VEC2I motion) override;
		crresult simulateMouseSet(CR_VEC2I pos) override;

	public:
		CR_VEC2I getMousePos() const override;
	};

}}}

#endif