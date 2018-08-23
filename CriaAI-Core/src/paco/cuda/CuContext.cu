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
		cudaError res = cudaMallocManaged(&mem, size);
		
		CRIA_AUTO_ASSERT(res == cudaSuccess, "Target size: %llu, Cuda Error code: %i", size, res);
		if (res != cudaSuccess)
			return nullptr;

		return mem;
	}
	void CRCuFree(void* mem)
	{
		if (mem)
			cudaFree(mem);
	}
}}}

#endif //CRIA_PACO_CUDA 
