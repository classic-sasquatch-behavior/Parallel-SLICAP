#pragma once

#pragma region includes

//opencv
#include<opencv2/opencv.hpp>

//cuda
#include<cuda.h>
#include<device_launch_parameters.h>
#include<cuda_runtime_api.h>

//stdlib
#include<string>
#include<vector>
using std::string;
using std::vector;

#pragma endregion----------[5]

#pragma region typedefs

typedef cv::Mat h_Mat;			//host matrix container
typedef cv::cuda::GpuMat d_Mat; //device matrix container

typedef cv::cuda::PtrStepSzi int_ptr;   //the pointer through which d_Mat is accessed from within the kernel
typedef cv::cuda::PtrStepSzf float_ptr; //same as above, although this one is for floats.

#pragma endregion

#pragma region device macros

//aquire the coordinates of the thread. works on 2d kernels as well as 1d, if youre okay with ignoring one of the dimensions.
#define GET_DIMS(_y_dim_, _x_dim_)							\
	int _x_dim_ = (blockIdx.x * blockDim.x) + threadIdx.x;	\
	int _y_dim_ = (threadIdx.y * blockDim.y) + threadIdx.y;	\
	int & _X_ = _x_dim_;									\
	int & _Y_ = _y_dim_;										

//check if the thread is within the bounds of the d_Mat given as shape, and if not, return the thread.
#define CHECK_BOUNDS(_shape_) if((_X_ >= _shape_.cols)||(_Y_ >= _shape_.rows)){return;} 

#pragma endregion

#pragma region global macros

//virtually transforms a 2d tensor into a smaller 2d tensor, and obtains the resulting coordinates
#define CAST_DOWN(_old_coord_, _new_max_) \
	((_old_coord_ - (_old_coord_ % _new_max_ ))/ _new_max_)

//virtually transforms a 2d tensor into a larger 2d tensor, and obtains the resulting coordinates. 
//specifically, it tries to point to the element of the larger array which correlates to the geometric center of 
//the smaller array, if one were to imagine the two tensors superimposed. 
#define CAST_UP(_old_coord_, _old_max_, _new_max_) \
	((_old_coord_*(_new_max_/_old_max_))+(((_new_max_/_old_max_)-((_new_max_/_old_max_)%2))/2))

//iterates through the elements directly adjacent to the given coordinates.
#define FOR_NEIGHBOR(_base_y_dim_, _base_x_dim_, _parent_, _y_dim_, _x_dim_, _content_)					 \
	for(uint _y_dim_ = -1; _y_dim_ < 2; _y_dim_++){														 \
		for (uint _x_dim_ = -1; _x_dim_ < 2; _x_dim_++) {											     \
			if(((_y_dim_ + _base_y_dim_) < 0)||((_x_dim_ + _base_x_dim_) < 0)							 \
			||((_y_dim_ + _base_y_dim_) >= _parent_.rows)||((_x_dim_ + _base_x_dim_) >= _parent_.cols )) \
			{continue;}																					 \
			_content_;																					 \
		}																								 \
	}

//virtually transform a 2d tensor into a 1d tensor, and obtain the resulting id of the element pointed to by the given coordinates
#define LINEAR_CAST(_y_dim_, _x_dim_, _x_max_) \
((_y_dim_ * _x_max_) + _x_dim_)

#pragma endregion

#pragma region host macros

//launch a cuda function (such as those required for memory allocation), and make sure it went through without error
#define CUDA_FUNCTION_CALL(_function_)													  \
{																						  \
	cudaError_t error = (_function_);													  \
	if(error != cudaSuccess) {															  \
		std::cout << "Error at " << __FILE__ << ":" << __LINE__ << " - "				  \
			<< cudaGetErrorName(error) << ", " << cudaGetErrorString(error) << std::endl; \
		abort();																		  \
	}																					  \
}

//synchronize host with device, check for errors in the kernel
#define SYNC_AND_CHECK_FOR_ERRORS(_kernel_)											 \
{																					 \
	cudaError_t error = cudaGetLastError();											 \
	if(error != cudaSuccess) {														 \
		std::cout << "Error in kernel " << #_kernel_								 \
		<< " at " << __FILE__ << ":" << __LINE__ << ": "							 \
		<< cudaGetErrorName(error) << ":" << cudaGetErrorString(error) << std::endl; \
		abort();													  				 \
		}																			 \
}			

//launch kernel
#define LAUNCH_KERNEL(_kernel_name_, _configure_kernel_, _kernel_arguments_) \
	_configure_kernel_;														 \
	_kernel_name_ <<<num_blocks, threads_per_block>>> _kernel_arguments_;	 \
	SYNC_AND_CHECK_FOR_ERRORS(_kernel_name_);

//create the necessary host and device pointers to allocate memory to the device. written as a macro to reduce boilerplate.
#define DECLARE_HOST_AND_DEVICE_POINTERS(_type_, _pointer_name_) \
	_type_ _pointer_name_;										 \
	_type_* h_##_pointer_name_ = &_pointer_name_;				 \
	_type_* d_##_pointer_name_;

#pragma endregion
