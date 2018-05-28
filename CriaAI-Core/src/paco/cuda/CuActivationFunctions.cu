#include "../ActivationFunctions.h"

#ifdef CRIA_PACO_CUDA 

#include "CuContext.cuh"

#define CR_CUDA_AF_BLOCK_COUNT         1
#define CR_CUDA_AF_THREAD_COUNT        256

namespace cria_ai { namespace paco {
	
	/**
	* \brief A activation function
	*
	* Equation:     1 / (1 + e^-x) = r
	* Output Range: (0 < x < 1)
	*
	* \param input  A matrix containing values for processing.
	* \param output A matrix that holds the output values.
	*/
	__global__ void CRCuSigmoid(CRNWMat const* input, CRNWMat* output);
	void CRSigmoid(CRNWMat const* input, CRNWMat* output)
	{
		CRIA_SIGMOID_VALIDATION_CHECK(input, output);

		CRCuSigmoid<<<CR_CUDA_AF_BLOCK_COUNT, CR_CUDA_AF_THREAD_COUNT >>>(input, output);
		cudaDeviceSynchronize();
	}
	__global__ void CRCuSigmoid(CRNWMat const* input, CRNWMat* output)
	{
		int startIndex = blockIdx.x * blockDim.x + threadIdx.x;
		int stride = blockDim.x * gridDim.x;

		for (uint index = startIndex; index < input->Cols * input->Rows; index += stride)
		{
			output->Data[index] = 1.0f / (1.0f + __expf(-input->Data[index]));
		}
	}

	/**
	* \brief A inverse activation function
	*
	* Equation:    -ln((1/r) - 1) = x
	* Input Range: (0 < r < 1)
	*
	* \param input  A matrix containing values for processing.
	* \param output A matrix that holds the output values.
	*/
	__global__ void CRCuSigmoidInv(CRNWMat const* input, CRNWMat* output);
	void CRSigmoidInv(CRNWMat const* input, CRNWMat* output)
	{
		CRIA_SIGMOID_VALIDATION_CHECK(input, output);

		CRCuSigmoidInv<<<CR_CUDA_AF_BLOCK_COUNT, CR_CUDA_AF_THREAD_COUNT>>>(input, output);
		cudaDeviceSynchronize();
	}
	__global__ void CRCuSigmoidInv(CRNWMat const* input, CRNWMat* output)
	{
		int startIndex = blockIdx.x * blockDim.x + threadIdx.x;
		int stride = blockDim.x * gridDim.x;

		for (uint index = startIndex; index < input->Cols * input->Rows; index += stride) {

			if (input->Data[index] > 0 || input->Data[index] < 1)
				output->Data[index] = -__logf((1 / input->Data[index]) - 1);
			else 
				output->Data[index] = 0;

		}
	}
}}

#endif //CRIA_PACO_CUDA 
