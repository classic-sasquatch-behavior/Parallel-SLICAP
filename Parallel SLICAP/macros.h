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

typedef cv::cuda::PtrStepSzi int_ptr;   //the pointer by means of which d_Mat is accessed in the kernel
typedef cv::cuda::PtrStepSzf float_ptr; //same as above, although this one is for floats.

#pragma endregion----------[4]

#pragma region device macros

#define GET_DIMS(_y_dim_, _x_dim_)							\
	int _x_dim_ = (blockIdx.x * blockDim.x) + threadIdx.x;	\
	int _y_dim_ = (threadIdx.y * blockDim.y) + threadIdx.y;	\
	int & _X_ = _x_dim_;									\
	int & _Y_ = _y_dim_;										

#define CHECK_BOUNDS(_shape_) if((_X_ >= _shape_.cols)||(_Y_ >= _shape_.rows)){return;} 

#pragma endregion-----[3]

#pragma region global macros

//virtually transform a 2d tensor into a smaller 2d tensor, and obtain the resulting coordinates
#define CAST_DOWN(_old_coord_, _new_max_) \
	((_old_coord_ - (_old_coord_ % _new_max_ ))/ _new_max_)

//virtually transform a 2d tensor into a larger 2d tensor, and obtain the resulting coordinates. 
//specifically, it tries to point to the element of the larger array which correlates to the geometric center of 
//the smaller array, if one were to imagine the two tensors superimposed. 
//if _new_max_ is not divisible by _old_max_ (as would happen if the two matrices being operated upon are not similar), 
//then the behavior of this macro is undefined (though it should still provide an answer that is close).
//if the common denominator of the new dimension and the old dimension is even, it is obviously impossible to return a
//true center point. in this case, the macro just tries to get reasonably close.
#define CAST_UP(_old_coord_, _old_max_, _new_max_) \
	((_old_coord_*(_new_max_/_old_max_))+(((_new_max_/_old_max_)-((_new_max_/_old_max_)%2))/2))

//iterates through the elements directly adjacent to the given coordinates.
#define FOR_NEIGHBOR(_base_y_dim_, _base_x_dim_, _parent_, _y_dim_, _x_dim_, _content_)					 \
	for(uint _y_dim_ = -1; _y_dim_ < 2; _y_dim_++){														 \
		for (uint _x_dim_ = -1; _x_dim_ < 2; _x_dim_++) {											     \
			if(((_y_dim_ + _base_y_dim_) < 0)||((_x_dim_ + _base_x_dim_) < 0)							 \
			||((_y_dim_ + _base_y_dim_) >= _parent_.rows)||((_x_dim_ + _base_x_dim_) >= _parent_.cols )) \
			{ break;}																					 \
			_content_;																					 \
		}																								 \
	}

//virtually transform a 2d tensor into a 1d tensor, and obtain the resulting id of the element pointed to by the given coordinates
#define LINEAR_CAST(_y_dim_, _x_dim_, _x_max_) \
((_y_dim_ * _x_max_) + _x_dim_)

#define GET_CARTESIAN_COORDS(_y_dim_, _x_dim_, _id_, _new_x_max_) \
int _x_dim_ = _id_ % _new_x_max_;								  \
int _y_dim_ = (_id_ - _x_dim_) / _new_x_max_;

#define REPEAT_UNTIL_CONVERGENCE(_define_convergence_, _content_) \
	converged = false;				\
	while(!converged) {				\
		_content_;					\
		if (_define_convergence_) {	\
			converged = true;		\
		}							\
	}								



#pragma endregion-----[2]

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

//create the necessary host and device pointers to allocate memory to the device. written as a macro to reduce boilerplate.
#define DECLARE_HOST_AND_DEVICE_POINTERS(_type_, _pointer_name_) \
	_type_ _pointer_name_;										 \
	_type_* h_##_pointer_name_ = &_pointer_name_;				 \
	_type_* d_##_pointer_name_;

//declare the channels of the vector<d_Mat*> within the local scope. written as a macro to reduce boilerplate
#define UNPACK(_source_, _first_, _second_, _third_) \
d_Mat& _first_ = *(_source_[0]);					 \
d_Mat& _second_ = *(_source_[1]);					 \
d_Mat& _third_ = *(_source_[2]);


#define XML(_content_)				\
{									\
std::filesystem.open("sneaky.xml") << #_content_;		\
}				




#pragma endregion-------[1]

#pragma region readme
// some notes on the standard followed in this document:
//
// • references passed to functions invariably represent the result of that function.
// • similarly, the results of a function or kernel are always the last arguments provided to the function or kernel.
// • 'function' will only ever refer to host functions. Kernels are called 'kernels' throughout the document.
// • the pointers required for memory allocation are declared in a macro. this produces h_[pointer_name] as a host pointer,
// and d_[pointer_name] as a device pointer. 
// • the terms 'center', 'cluster' and 'superpixel' are used interchangably when discussing SLIC
// • all user-controlled parameters are declared as const
// • OpenCV uses row major notation, therefore matrices are always treated as row major in the document
#pragma endregion------------[0]
