#pragma once

#include"external_includes.h"
#include"Matrix.cuh"
#include"macros.h"




__global__ static void AP_condense_color_vectors(int_ptr L_src, int_ptr A_src, int_ptr B_src, int_ptr labels, int N, int_ptr color_vectors, int_ptr pixels_per_superpixel) {
	GET_DIMS(row, col);
	CHECK_BOUNDS(labels.rows, labels.cols);

	int label = labels(row, col);

	int L = L_src(row, col);
	int A = A_src(row, col);
	int B = B_src(row, col);

	atomicAdd(&color_vectors(0, label), L);
	atomicAdd(&color_vectors(1, label), A);
	atomicAdd(&color_vectors(2, label), B);

	atomicAdd(&pixels_per_superpixel(0, label), 1);
}

__global__ static void AP_calculate_average_color_vectors(int_ptr color_vectors, int_ptr pixels_per_superpixel) {
	GET_DIMS(NA, id);
	CHECK_BOUNDS(pixels_per_superpixel.rows, pixels_per_superpixel.cols);

	int N = pixels_per_superpixel(0, id);

	int L = color_vectors(0, id) / N;
	int A = color_vectors(1, id) / N;
	int B = color_vectors(2, id) / N;

	color_vectors(0, id) = L;
	color_vectors(1, id) = A;
	color_vectors(2, id) = B;
}

#pragma endregion

#pragma region similarity matrix
__global__ static void AP_generate_similarity_matrix(int_ptr color_vectors, int_ptr pixels_per_superpixel, float_ptr similarity_matrix) {
	GET_DIMS(data, exemplar);
	CHECK_BOUNDS(similarity_matrix.rows, similarity_matrix.cols);

	int data_N = pixels_per_superpixel(0, data);

	int data_L = color_vectors(0, data) / data_N;
	int data_A = color_vectors(1, data) / data_N;
	int data_B = color_vectors(2, data) / data_N;
	int data_channels[3] = { data_L, data_A, data_B };

	int exemplar_N = pixels_per_superpixel(0, exemplar);

	int exemplar_L = color_vectors(0, exemplar) / exemplar_N;
	int exemplar_A = color_vectors(1, exemplar) / exemplar_N;
	int exemplar_B = color_vectors(2, exemplar) / exemplar_N;
	int exemplar_channels[3] = { exemplar_L, exemplar_A, exemplar_B };

	int sum = 0;
	for (int i = 0; i < 3; i++) {
		int data_channel = data_channels[i];
		int exemplar_channel = exemplar_channels[i];
		sum += (data_channel - exemplar_channel) * (data_channel - exemplar_channel);
	}
	float result = -sqrtf(sum);

	similarity_matrix(data, exemplar) = result;
}

__global__ static void AP_scan_for_lowest_value(float_ptr similarity_matrix, float_ptr lowest_values, float* result) {
	GET_DIMS(zero, id);
	CHECK_BOUNDS(similarity_matrix.rows, similarity_matrix.cols); //technically works to check bounds with a square mat, because the kernel is one dimensional

	float lowest_value = 1000000.0f;
	for (int i_data = 0; i_data < similarity_matrix.rows; i_data++) {
		float value = similarity_matrix(i_data, id);
		if (value < lowest_value) {
			lowest_value = value;
		}
	}
	lowest_values(0, id) = lowest_value;

	if (id != 0) { return; }
	for (int i = 0; i < similarity_matrix.rows; i++) {
		float value = lowest_values(0, i);
		if (value < lowest_value) {
			lowest_value = value;
		}
	}

	*result = lowest_value;

}

__global__ static void AP_set_preference_values(float* preference_value, float_ptr similarity_matrix) {
	GET_DIMS(zero, id);
	CHECK_BOUNDS(similarity_matrix.rows, similarity_matrix.cols); //technically works to check bounds with a square mat, because the kernel is one dimensional

	similarity_matrix(id, id) = *preference_value;

}
#pragma endregion

#pragma region responsibility matrix

__global__ static void AP_update_responsibility_matrix(float_ptr similarity_matrix, float_ptr availibility_matrix, float damping_factor, float_ptr responsibility_matrix) {
	GET_DIMS(data, exemplar);
	CHECK_BOUNDS(responsibility_matrix.rows, responsibility_matrix.cols);

	float similarity = similarity_matrix(data, exemplar);

	float max_value = -1000000;
	for (int i_exemplar = 0; i_exemplar < responsibility_matrix.cols; i_exemplar++) {
		if (i_exemplar = exemplar) { continue; }
		float value = availibility_matrix(data, i_exemplar) + similarity_matrix(data, i_exemplar);
		if (value > max_value) {
			max_value = value;
		}
	}

	float result = similarity - max_value;

	result = (damping_factor * (responsibility_matrix(data, exemplar))) + ((1.0 - damping_factor) * (result));

	responsibility_matrix(data, exemplar) = result;
}

#pragma endregion

#pragma region availibility matrix

__global__ static void AP_update_availibility_matrix(float_ptr responsibility_matrix, float_ptr availibility_matrix, float damping_factor) {
	GET_DIMS(data, exemplar);
	CHECK_BOUNDS(availibility_matrix.rows, availibility_matrix.cols);

	float sum = 0;
	for (int i_data = 0; i_data < availibility_matrix.rows; i_data++) {
		if ((i_data == data) || (i_data == exemplar)) { continue; }
		sum += fmaxf(0, responsibility_matrix(i_data, exemplar));
	}

	float result = responsibility_matrix(data, exemplar) + sum;
	result = (damping_factor * (availibility_matrix(data, exemplar))) + ((1.0 - damping_factor) * (result));
	result = fminf(0, result);
	availibility_matrix(data, exemplar) = result;

}

#pragma endregion

#pragma region critereon matrix

__global__ static void AP_calculate_critereon_matrix(float_ptr availibility_matrix, float_ptr responsiblity_matrix, float_ptr critereon_matrix) {
	GET_DIMS(data, exemplar);
	CHECK_BOUNDS(critereon_matrix.rows, critereon_matrix.cols);
	critereon_matrix(data, exemplar) = availibility_matrix(data, exemplar) + responsiblity_matrix(data, exemplar);
}

__global__ static void AP_extract_and_examine_exemplars(float_ptr critereon_matrix, int_ptr exemplars, int* difference) {
	GET_DIMS(zero, id);
	CHECK_BOUNDS(exemplars.rows, exemplars.cols);

	float max_value = -1000000.0f;
	int index_of_exemplar = -1;
	for (int i_exemplar = 0; i_exemplar < exemplars.cols; i_exemplar++) {

		float value = critereon_matrix(id, i_exemplar);
		if (value > max_value) {
			max_value = value;
			index_of_exemplar = i_exemplar;
		}
	}

	int previous_exemplar = exemplars(0, id);
	if (previous_exemplar != index_of_exemplar) {
		atomicAdd(difference, 1);
	}

	exemplars(0, id) = index_of_exemplar;
}

#pragma endregion

#pragma endregion
