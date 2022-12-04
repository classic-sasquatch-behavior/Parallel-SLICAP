#pragma once

#include"external_includes.h"

#pragma region device macros

//aquire the coordinates of the thread. works on 2d kernels as well as 1d, if youre okay with ignoring one of the dimensions.
#define GET_DIMS(_y_dim_, _x_dim_)							\
	int _x_dim_ = (blockIdx.x * blockDim.x) + threadIdx.x;	\
	int _y_dim_ = (blockIdx.y * blockDim.y) + threadIdx.y;	\
	int & _X_ = _x_dim_;									\
	int & _Y_ = _y_dim_;										

//check if the thread is within the bounds of the d_Mat given as shape, and if not, return the thread.
#define CHECK_BOUNDS(_max_rows_, _max_cols_) if((_X_ >= _max_cols_)||(_Y_ >= _max_rows_)){return;} 

#pragma endregion

#pragma region global macros

//virtually transforms a 2d tensor into a smaller 2d tensor, and obtains the resulting coordinates
#define CAST_DOWN(_old_coord_, _new_max_) \
	((_old_coord_ - (_old_coord_ % _new_max_ ))/ _new_max_)

//virtually transforms a 2d tensor into a larger 2d tensor, and obtains the resulting coordinates. 
#define CAST_UP(_old_coord_, _old_max_, _new_max_) \
	((_old_coord_*(_new_max_/_old_max_))+(((_new_max_/_old_max_)-((_new_max_/_old_max_)%2))/2))

//iterates through the elements directly adjacent to the given coordinates.
#define FOR_NEIGHBOR(_new_y_dim_, _new_x_dim_, _parent_y_max_, _parent_x_max_, _base_y_dim_, _base_x_dim_, _content_)     \
	for(int _y_dim_ = -1; _y_dim_ < 2; _y_dim_++){																	      \
		for (int _x_dim_ = -1; _x_dim_ < 2; _x_dim_++) {															      \
			int _new_y_dim_ = _base_y_dim_ + _y_dim_;																      \
			int _new_x_dim_ = _base_x_dim_ + _x_dim_;																      \
			if((_new_y_dim_ < 0)||(_new_x_dim_ < 0) || (_new_y_dim_ >= _parent_y_max_)||(_new_x_dim_ >= _parent_x_max_ )) \
			{continue;}																								      \
			_content_;																								      \
		}																											      \
	}

//virtually transform a 2d tensor into a 1d tensor, and obtain the resulting id of the element pointed to by the given coordinates
#define LINEAR_CAST(_y_dim_, _x_dim_, _x_max_) ((_y_dim_ * _x_max_) + _x_dim_)

#pragma endregion

#pragma region host macros



#define PRINT_MAT(_input_)\
std::cout << std::endl << #_input_ << std::endl;\
std::cout << "cols: " << _input_.cols << ", rows: " << _input_.rows << std::endl;

//create the necessary host and device pointers to allocate memory to the device. written as a macro to reduce boilerplate.
#define DECLARE_HOST_AND_DEVICE_POINTERS(_type_, _pointer_name_) \
	_type_ _pointer_name_;										 \
	_type_* h_##_pointer_name_ = &_pointer_name_;				 \
	_type_* d_##_pointer_name_;

#pragma endregion
