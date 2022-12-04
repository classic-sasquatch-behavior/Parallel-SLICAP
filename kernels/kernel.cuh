#pragma once

#include"macros.h"



#pragma region AP kernels

#pragma region calculate color vectors

__global__ void AP_condense_color_vectors(int_ptr L_src, int_ptr A_src, int_ptr B_src, int_ptr labels, int N, int_ptr color_vectors, int_ptr pixels_per_superpixel) {
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

__global__ void AP_calculate_average_color_vectors(int_ptr color_vectors, int_ptr pixels_per_superpixel) {
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
__global__ void AP_generate_similarity_matrix(int_ptr color_vectors, int_ptr pixels_per_superpixel, float_ptr similarity_matrix) {
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

__global__ void AP_scan_for_lowest_value(float_ptr similarity_matrix, float_ptr lowest_values, float* result) {
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

__global__ void AP_set_preference_values(float* preference_value, float_ptr similarity_matrix) {
	GET_DIMS(zero, id);
	CHECK_BOUNDS(similarity_matrix.rows, similarity_matrix.cols); //technically works to check bounds with a square mat, because the kernel is one dimensional

	similarity_matrix(id, id) = *preference_value;

}
#pragma endregion

#pragma region responsibility matrix

__global__ void AP_update_responsibility_matrix(float_ptr similarity_matrix, float_ptr availibility_matrix, float damping_factor, float_ptr responsibility_matrix) {
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

__global__ void AP_update_availibility_matrix(float_ptr responsibility_matrix, float_ptr availibility_matrix, float damping_factor) {
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

__global__ void AP_calculate_critereon_matrix(float_ptr availibility_matrix, float_ptr responsiblity_matrix, float_ptr critereon_matrix) {
	GET_DIMS(data, exemplar);
	CHECK_BOUNDS(critereon_matrix.rows, critereon_matrix.cols);
	critereon_matrix(data, exemplar) = availibility_matrix(data, exemplar) + responsiblity_matrix(data, exemplar);
}

__global__ void AP_extract_and_examine_exemplars(float_ptr critereon_matrix, int_ptr exemplars, int* difference) {
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

#pragma region SLIC kernels

#pragma region initialize centers

__global__ void SLIC_initialize_centers(int_ptr source, int_ptr center_rows, int_ptr center_cols) {
	GET_DIMS(row, col);
	CHECK_BOUNDS(center_rows.rows, center_rows.cols );
	center_rows(row, col) = CAST_UP(row, center_rows.rows, source.rows);
	center_cols(row, col) = CAST_UP(col, center_cols.cols, source.cols);
}

#pragma endregion

#pragma region assign pixels to centers

__global__ void SLIC_assign_pixels_to_centers(int_ptr L_src, int_ptr A_src, int_ptr B_src, int density_modifier, int_ptr K_rows, int_ptr K_cols, int_ptr labels) {
	GET_DIMS(row, col);
	CHECK_BOUNDS(L_src.rows, L_src.cols);

	int sector_row = CAST_DOWN(row, K_rows.rows);
	int sector_col = CAST_DOWN(col, K_rows.cols);

	int self_channels[3] = { L_src(row, col), A_src(row, col), B_src(row, col) };
	int self_coordinates[2] = { row, col };

	int min_distance = 1000000; //arbitrarily large number for comparison
	int closest_center = -1;

	FOR_NEIGHBOR(neighbor_row, neighbor_col, K_rows.rows, K_rows.cols, row, col,
		int actual_neighbor_row = K_rows(neighbor_row, neighbor_col);
		int actual_neighbor_col = K_cols(neighbor_row, neighbor_col);

		int neighbor_channels[3] = { (L_src(actual_neighbor_row, actual_neighbor_col), A_src(actual_neighbor_row, actual_neighbor_col), B_src(actual_neighbor_row, actual_neighbor_col)) };
		int neighbor_coordinates[2] = { (actual_neighbor_row, actual_neighbor_col) };
		int neighbor_label = LINEAR_CAST(neighbor_row, neighbor_col, K_rows.cols);

		int color_distance = 0;
		for (int i = 0; i < 3; i++) {
			int self_channel = self_channels[i];
			int neighbor_channel = neighbor_channels[i];
			color_distance += (self_channel - neighbor_channel) * (self_channel - neighbor_channel);
		}
		color_distance = sqrtf(color_distance);

		int spatial_distance = 0;
		for (int i = 0; i < 2; i++) {
			int self_coordinate = self_coordinates[i];
			int neighbor_coordinate = neighbor_coordinates[i];
			spatial_distance += (self_coordinate - neighbor_coordinate) * (self_coordinate - neighbor_coordinate);
		}
		spatial_distance = sqrtf(spatial_distance);

		int total_distance = color_distance + (density_modifier * spatial_distance);
		if (total_distance < min_distance) {
			min_distance = total_distance;
			closest_center = neighbor_label;
		}
	);

	labels(row, col) = closest_center;

}

#pragma endregion

#pragma region update centers

__global__ void SLIC_condense_labels(int_ptr labels, int_ptr row_sums, int_ptr col_sums, int_ptr num_instances) {
	GET_DIMS(row, col);
	CHECK_BOUNDS(labels.rows, labels.cols);

	int label = labels(row, col);
	atomicAdd(&row_sums(0, label), row);
	atomicAdd(&col_sums(0, label), col);
	atomicAdd(&num_instances(0, label), 1);
}

__global__ void SLIC_update_centers(int_ptr row_sums, int_ptr col_sums, int_ptr num_instances, int_ptr center_rows, int_ptr center_cols) {
	GET_DIMS(row, col);
	CHECK_BOUNDS(row_sums.rows, row_sums.cols);
	uint id = LINEAR_CAST(row, col, row_sums.cols);

	int row_sum = row_sums(0, id);
	int col_sum = col_sums(0, id);
	int N = num_instances(0, id);

	int new_row = row_sum / N;
	int new_col = col_sum / N;

	center_rows(row, col) = new_row;
	center_cols(row, col) = new_col;
}

#pragma endregion

#pragma region separate blobs

__global__ void SLIC_separate_blobs(int_ptr original_labels, int_ptr working_labels, uint* flag) {
	GET_DIMS(row, col);
	CHECK_BOUNDS(original_labels.rows, original_labels.cols);
	uint id = LINEAR_CAST(row, col, original_labels.cols);

	int original_label = original_labels(row, col);
	//MAKE SURE TO INITIALIZE WORKING LABELS AS ALL 0s!

	int working_label = working_labels(row, col);
	if (working_label < id) { working_label = id; }

	FOR_NEIGHBOR(irow, icol, original_labels.rows, original_labels.cols, row, col,
		int neighbor_label = working_labels(irow, icol);
		int neighbor_original_label = original_labels(irow, icol);
		if (neighbor_original_label == original_label) {
			if (neighbor_label > working_label) {
				working_label = neighbor_label;
				*flag = 1;
			}
		}
	);

	working_labels(row, col) = working_label;
}

#pragma endregion

#pragma region absorb small blobs

__global__ void SLIC_find_sizes(int_ptr labels, int_ptr label_sizes) {
	GET_DIMS(row, col);
	CHECK_BOUNDS(labels.rows, labels.cols);

	int label = labels(row, col);
	atomicAdd(&label_sizes(0, label), 1);
}

__global__ void SLIC_find_weak_labels(int_ptr label_sizes, int_ptr label_strengths, int threshold) {
	GET_DIMS(row, col);
	CHECK_BOUNDS(label_strengths.rows, label_strengths.cols);
	uint id = LINEAR_CAST(row, col, label_strengths.cols);
	int label_size = label_sizes(0, id);
	if (label_size <= threshold) { label_strengths(row, col) = 1; }
}

//look over again, somethings wrong in here I can tell 
__global__ void SLIC_absorb_small_blobs(int_ptr original_labels, int_ptr label_strengths, int_ptr label_sizes, int_ptr working_labels, uint* flag) {
	GET_DIMS(row, col);
	CHECK_BOUNDS(original_labels.rows, original_labels.cols);
	int original_label = original_labels(row, col);
	int working_label = working_labels(row, col);
	int temp_label = working_label;
	bool weak = (bool)label_strengths(row, col);
	if (weak) {
		int size = label_sizes(0, original_label);
		FOR_NEIGHBOR(neighbor_row, neighbor_col, original_labels.rows, original_labels.cols, row, col,
			int neighbor_label = working_labels(neighbor_row, neighbor_col);
			int neighbor_size = label_sizes(0, neighbor_label);

			if (neighbor_size > size) {
				temp_label = neighbor_label;
			}
		)
		if (working_label != temp_label) {
			working_labels(row, col) = temp_label;
			*flag = 1;
		}
	}
	else { return; }
}

#pragma endregion

#pragma region produce ordered labels

__global__ void SLIC_raise_flags(int_ptr labels, int_ptr flags) {
	GET_DIMS(col, row);
	CHECK_BOUNDS(labels.rows, labels.cols);

	int id = labels(row, col);
	flags(1, id) = 1;
}

__global__ void SLIC_init_map(int_ptr flags, int_ptr sum_flags, int_ptr map) {
	GET_DIMS(id, NA);
	CHECK_BOUNDS(flags.rows, flags.cols);
	if (flags(0, id) == 0) { return; }
	int condensed_id = sum_flags(0, id);
	map(0, condensed_id) = id;
}

__global__ void SLIC_invert_map(int_ptr condensed_map, int_ptr useful_map) {
	GET_DIMS(NA, id);
	CHECK_BOUNDS(condensed_map.rows, condensed_map.cols);
	int value = condensed_map(0, id);
	useful_map(0, value) = id;
}

__global__ void SLIC_assign_new_labels(int_ptr labels, int_ptr map) {
	GET_DIMS(row, col);
	CHECK_BOUNDS(labels.rows, labels.cols);
	int orginal_label = labels(row, col);
	labels(row, col) = map(0, orginal_label);
}

#pragma endregion

#pragma endregion