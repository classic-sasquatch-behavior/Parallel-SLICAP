#pragma once



#pragma region includes

//opencv
#include<opencv4/opencv2/opencv.hpp>

//cuda
#include<cuda.h>
#include<device_launch_parameters.h>
#include<cuda_runtime_api.h>

//stdlib
#include<string>
#include<vector>
using std::string;
using std::vector;

#pragma endregion

#pragma region typedefs

typedef unsigned int uint;

typedef cv::Mat h_Mat;			//host matrix container
typedef cv::cuda::GpuMat d_Mat; //device matrix container

typedef cv::cuda::PtrStepSzi int_ptr;   //the pointer through which d_Mat is accessed from within the kernel
typedef cv::cuda::PtrStepSzf float_ptr; //same as above, although this one is for floats.

#pragma endregion