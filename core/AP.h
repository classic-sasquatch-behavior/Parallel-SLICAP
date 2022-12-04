#pragma once

#include"AP_kernels.cuh"
#include"SLICAP.h"



struct AP {

	//values
	cv::Size AP_matrix_size;
	const int AP_matrix_type = CV_32FC1;

	SLICAP* parent;


	d_Mat similarity_matrix, responsibility_matrix, availibility_matrix, critereon_matrix, exemplars;
	d_Mat average_superpixel_color_vectors, pixels_per_superpixel;

	int difference_in_exemplars;
	int* h_difference_in_exemplars = &difference_in_exemplars;
	int* d_difference_in_exemplars;

	int constant_cycles = 0;

	AP(SLICAP* _parent);

	void calculate_average_color_vectors();

	void generate_similarity_matrix();

	void update_responsibility_matrix();

	void update_availibility_matrix();

	void update_critereon_matrix();

	void extract_and_examine_exemplars();

	void segment_image_using_exemplars();

};

