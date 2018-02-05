#pragma once

#include "../Common.hpp"
#include "../maths/Matrixf.hpp"

namespace cria_ai
{
	
	typedef enum CRIA_LAYER_TYPE_ {
		CRIA_LAYER_INPUT_LAYER   = 0x0100,
		
		CRIA_LAYER_HIDDEN_LAYER  = 0x0200,
		CRIA_LAYER_MERGING_LAYER = 0x0201,

		CRIA_LAYER_OUTPUT_LAYER  = 0x0400,
	} CRIA_LAYER_TYPE;

	class CRLayer
	{
	protected:
		CRMatrixf* m_LastOutput;

		uint32_t m_InNodesCount;
		uint32_t m_OutNodesCount;
	public:

		virtual ~CRLayer() {}

		/**
		 * \brief This method is used by input layers. The input info is interpreted 
		 * by the layer. for a CRBitmapInLayer this would be the bitmap file name.
		 * 
		 * \param inputInfo Relevant information for input layers.
		 * 
		 * \return This returns true if the loading was successful or if the layer is no
		 * input layer. 
		 */
		virtual bool loadInput(const String& inputInfo)
		{
			return true;
		}

		virtual CRIA_LAYER_TYPE getType() const = 0;
		inline  bool isType(const CRIA_LAYER_TYPE& type) const
		{
			return (type & getType()) == 0;
		}

		/*
		 * getters
		 */
		CRMatrixf* getLastOutput()
		{
			return m_LastOutput;
		}
		CRMatrixf const* getLastOutput() const
		{
			return m_LastOutput;
		}

		uint32_t getInNodesCount() const
		{
			return m_InNodesCount;
		}
		uint32_t getOutNodesCount() const 
		{
			return m_OutNodesCount;
		}
	};

}
