#pragma once

#include"external_includes.h"
#include"Matrix.cuh"
#include"macros.h"



#pragma region SLIC kernels

#pragma region initialize centers

__global__ static void SLIC_initialize_centers(int_ptr source, int_ptr center_rows, int_ptr center_cols) {
	GET_DIMS(row, col);
	CHECK_BOUNDS(center_rows.rows, center_rows.cols );
	center_rows(row, col) = CAST_UP(row, center_rows.rows, source.rows);
	center_cols(row, col) = CAST_UP(col, center_cols.cols, source.cols);
}

#pragma endregion

#pragma region assign pixels to centers

__global__ static void SLIC_assign_pixels_to_centers(int_ptr L_src, int_ptr A_src, int_ptr B_src, int density_modifier, int_ptr K_rows, int_ptr K_cols, int_ptr labels) {
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

__global__ static void SLIC_condense_labels(int_ptr labels, int_ptr row_sums, int_ptr col_sums, int_ptr num_instances) {
	GET_DIMS(row, col);
	CHECK_BOUNDS(labels.rows, labels.cols);

	int label = labels(row, col);
	atomicAdd(&row_sums(0, label), row);
	atomicAdd(&col_sums(0, label), col);
	atomicAdd(&num_instances(0, label), 1);
}

__global__ static void SLIC_update_centers(int_ptr row_sums, int_ptr col_sums, int_ptr num_instances, int_ptr center_rows, int_ptr center_cols) {
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

__global__ static void SLIC_separate_blobs(int_ptr original_labels, int_ptr working_labels, uint* flag) {
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

__global__ static void SLIC_find_sizes(int_ptr labels, int_ptr label_sizes) {
	GET_DIMS(row, col);
	CHECK_BOUNDS(labels.rows, labels.cols);

	int label = labels(row, col);
	atomicAdd(&label_sizes(0, label), 1);
}

__global__ static void SLIC_find_weak_labels(int_ptr label_sizes, int_ptr label_strengths, int threshold) {
	GET_DIMS(row, col);
	CHECK_BOUNDS(label_strengths.rows, label_strengths.cols);
	uint id = LINEAR_CAST(row, col, label_strengths.cols);
	int label_size = label_sizes(0, id);
	if (label_size <= threshold) { label_strengths(row, col) = 1; }
}

//look over again, somethings wrong in here I can tell 
__global__ static void SLIC_absorb_small_blobs(int_ptr original_labels, int_ptr label_strengths, int_ptr label_sizes, int_ptr working_labels, uint* flag) {
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

__global__ static void SLIC_raise_flags(int_ptr labels, int_ptr flags) {
	GET_DIMS(col, row);
	CHECK_BOUNDS(labels.rows, labels.cols);

	int id = labels(row, col);
	flags(1, id) = 1;
}

__global__ static void SLIC_init_map(int_ptr flags, int_ptr sum_flags, int_ptr map) {
	GET_DIMS(id, NA);
	CHECK_BOUNDS(flags.rows, flags.cols);
	if (flags(0, id) == 0) { return; }
	int condensed_id = sum_flags(0, id);
	map(0, condensed_id) = id;
}

__global__ static void SLIC_invert_map(int_ptr condensed_map, int_ptr useful_map) {
	GET_DIMS(NA, id);
	CHECK_BOUNDS(condensed_map.rows, condensed_map.cols);
	int value = condensed_map(0, id);
	useful_map(0, value) = id;
}

__global__ static void SLIC_assign_new_labels(int_ptr labels, int_ptr map) {
	GET_DIMS(row, col);
	CHECK_BOUNDS(labels.rows, labels.cols);
	int orginal_label = labels(row, col);
	labels(row, col) = map(0, orginal_label);
}

#pragma endregion

#pragma endregion