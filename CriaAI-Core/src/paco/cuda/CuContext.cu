#include "CuContext.cuh"

#ifdef CRIA_PACO_CUDA 

namespace cria_ai { namespace paco { namespace cu {
	
	crresult CRCuContext::init()
	{
		return CRRES_OK;
	}

	void* CRCuMalloc(size_t size)
	{
		void* mem = nullptr;
		cudaMallocManaged(&mem, size);
		return mem;
	}
	void CRCuFree(void* mem)
	{
		if (mem)
			cudaFree(mem);
	}
}}}

#endif //CRIA_PACO_CUDA 
