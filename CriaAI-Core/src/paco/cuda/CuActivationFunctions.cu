#include "CuActivationFunctions.cuh"

#ifdef CRIA_PACO_CUDA 

namespace cria_ai { namespace paco { namespace cuda {
	
	/**
	* \brief A activation function
	*
	* Equation:     1 / (1 + e^-x) = r
	* Output Range: (0 < x < 1)
	*
	* \param input  A matrix containing values for processing.
	* \param output A matrix that holds the output values.
	*/
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
}}}

#endif //CRIA_PACO_CUDA 
