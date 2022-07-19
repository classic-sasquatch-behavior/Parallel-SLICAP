#include"macros.h"

#pragma region general kernels

__global__ void exclusive_scan_upsweep(int N, int step, int_ptr source, int_ptr buffer, int* sum) {
	GET_DIMS(zero, id);
	if (id >= N) { return; }
	int a = step * (2 * id + 1) - 1;
	int b = step * (2 * id + 2) - 1;
	buffer(0, b) += source(0, a);

	if (N == 1) {
		if (id != 0) { return; }
		*sum = buffer(0, source.cols - 1);
		buffer(0, source.cols - 1) = 0;
	}


}

__global__ void exclusive_scan_downsweep(int N, int step, int_ptr source, int_ptr buffer) {
	GET_DIMS(zero, id);
	if (id >= N) { return; }
	int a = step * (2 * id + 1) - 1;
	int b = step * (2 * id + 2) - 1;

	float t = source(0, a);
	buffer(0, a) = source(0, b);
	buffer(0, b) += t;
}

#pragma endregion

#pragma region AP kernels

#pragma region calculate color vectors

__global__ void AP_condense_color_vectors(int_ptr L_src, int_ptr A_src, int_ptr B_src, int_ptr labels, int N, int_ptr color_vectors, int_ptr pixels_per_superpixel) {
	GET_DIMS(row, col);
	CHECK_BOUNDS(labels);

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
	CHECK_BOUNDS(pixels_per_superpixel);

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
	CHECK_BOUNDS(similarity_matrix);

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
	CHECK_BOUNDS(similarity_matrix); //technically works to check bounds with a square mat, because the kernel is one dimensional

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
	CHECK_BOUNDS(similarity_matrix); //technically works to check bounds with a square mat, because the kernel is one dimensional

	similarity_matrix(id, id) = *preference_value;

}
#pragma endregion

#pragma region responsibility matrix

__global__ void AP_update_responsibility_matrix(float_ptr similarity_matrix, float_ptr availibility_matrix, float damping_factor, float_ptr responsibility_matrix) {
	GET_DIMS(data, exemplar);
	CHECK_BOUNDS(responsibility_matrix);

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
	CHECK_BOUNDS(availibility_matrix);

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
	CHECK_BOUNDS(critereon_matrix);
	critereon_matrix(data, exemplar) = availibility_matrix(data, exemplar) + responsiblity_matrix(data, exemplar);
}

__global__ void AP_extract_and_examine_exemplars(float_ptr critereon_matrix, int_ptr exemplars, int* difference) {
	GET_DIMS(zero, id);
	CHECK_BOUNDS(exemplars);

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
	CHECK_BOUNDS(center_rows);
	center_rows(row, col) = CAST_UP(row, center_rows.rows, source.rows);
	center_cols(row, col) = CAST_UP(col, center_cols.cols, source.cols);
}

#pragma endregion

#pragma region assign pixels to centers

__global__ void SLIC_assign_pixels_to_centers(int_ptr L_src, int_ptr A_src, int_ptr B_src, int density_modifier, int_ptr K_rows, int_ptr K_cols, int_ptr labels) {
	GET_DIMS(row, col);
	CHECK_BOUNDS(L_src);

	int sector_row = CAST_DOWN(row, K_rows.rows);
	int sector_col = CAST_DOWN(col, K_rows.cols);

	int self_channels[3] = { L_src(row, col), A_src(row, col), B_src(row, col) };
	int self_coordinates[2] = { row, col };

	int min_distance = 1000000; //arbitrarily large number for comparison
	int closest_center = -1;

	FOR_NEIGHBOR(row, col, K_rows, irow, icol,
		int neighbor_row = irow + row;
		int neighbor_col = icol + col;

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
	CHECK_BOUNDS(labels);

	int label = labels(row, col);
	atomicAdd(&row_sums(0, label), row);
	atomicAdd(&col_sums(0, label), col);
	atomicAdd(&num_instances(0, label), 1);
}

__global__ void SLIC_update_centers(int_ptr row_sums, int_ptr col_sums, int_ptr num_instances, int_ptr center_rows, int_ptr center_cols) {
	GET_DIMS(row, col);
	CHECK_BOUNDS(row_sums);
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
	CHECK_BOUNDS(original_labels);
	uint id = LINEAR_CAST(row, col, original_labels.cols);

	int original_label = original_labels(row, col);
	//MAKE SURE TO INITIALIZE WORKING LABELS AS ALL 0s!

	int working_label = working_labels(row, col);
	if (working_label < id) { working_label = id; }

	FOR_NEIGHBOR(row, col, original_labels, irow, icol,
		int neighbor_label = working_labels(row, col);
		int neighbor_original_label = original_labels(row, col);
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
	CHECK_BOUNDS(labels);

	int label = labels(row, col);
	atomicAdd(&label_sizes(0, label), 1);
}

__global__ void SLIC_find_weak_labels(int_ptr label_sizes, int_ptr label_strengths, int threshold) {
	GET_DIMS(row, col);
	CHECK_BOUNDS(label_strengths);
	uint id = LINEAR_CAST(row, col, label_strengths.cols);
	int label_size = label_sizes(0, id);
	if (label_size <= threshold) { label_strengths(row, col) = 1; }
}

//look over again, somethings wrong in here I can tell 
__global__ void SLIC_absorb_small_blobs(int_ptr original_labels, int_ptr label_strengths, int_ptr label_sizes, int_ptr working_labels, uint* flag) {
	GET_DIMS(row, col);
	CHECK_BOUNDS(original_labels);
	int original_label = original_labels(row, col);
	int working_label = working_labels(row, col);
	int temp_label = working_label;
	bool weak = (bool)label_strengths(row, col);
	if (weak) {
		int size = label_sizes(0, original_label);
		FOR_NEIGHBOR(row, col, original_labels, irow, icol,
			int neighbor_row = row + irow;
		int neighbor_col = col + icol;

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
	CHECK_BOUNDS(labels);

	int id = labels(row, col);
	flags(1, id) = 1;
}

__global__ void SLIC_init_map(int_ptr flags, int_ptr sum_flags, int_ptr map) {
	GET_DIMS(id, NA);
	CHECK_BOUNDS(flags);
	if (flags(0, id) == 0) { return; }
	int condensed_id = sum_flags(0, id);
	map(0, condensed_id) = id;
}

__global__ void SLIC_invert_map(int_ptr condensed_map, int_ptr useful_map) {
	GET_DIMS(NA, id);
	CHECK_BOUNDS(condensed_map);
	int value = condensed_map(0, id);
	useful_map(0, value) = id;
}

__global__ void SLIC_assign_new_labels(int_ptr labels, int_ptr map) {
	GET_DIMS(row, col);
	CHECK_BOUNDS(labels);
	int orginal_label = labels(row, col);
	labels(row, col) = map(0, orginal_label);
}

#pragma endregion

#pragma endregion

namespace SLICAP {

	#pragma region declarations

	uint block_size_x; //threads per block in the x dimension
	uint block_size_y; //threads per block in the y dimension
	uint grid_size_x;  //blocks within grid in the x dimension
	uint grid_size_y;  //blocks within grid in the y dimension
	dim3 threads_per_block; //the actual threads per block to be passed into the kernel 
	dim3 num_blocks; //the actual number of blocks to be passed into the kernel

	h_Mat result;
	h_Mat source;
	vector<d_Mat> CIELAB_planes;
	d_Mat L_src;
	d_Mat A_src;
	d_Mat B_src;

	DECLARE_HOST_AND_DEVICE_POINTERS(uint, flag);

	int num_superpixels;
	d_Mat labels;

	#pragma endregion

	#pragma region functions

	vector<d_Mat> split_into_channels(h_Mat input) {
		vector<h_Mat> split;
		cv::split(input, split);
		d_Mat d_L, d_A, d_B;
		d_L.upload(split[0]);
		d_A.upload(split[1]);
		d_B.upload(split[2]);
		return { d_L, d_A, d_B };
	}

	void load_image(string source_filename) {
		h_Mat CIELAB_src;
		source = cv::imread(source_filename, cv::IMREAD_COLOR);
		cv::cvtColor(source, CIELAB_src, cv::COLOR_BGR2Lab);
		CIELAB_planes = split_into_channels(CIELAB_src);
		d_Mat& L_src = CIELAB_planes[0];
		d_Mat& A_src = CIELAB_planes[1];
		d_Mat& B_src = CIELAB_planes[2];
	}

	void kernel_1d(cv::Size shape) {
		uint src_x = shape.width;
		uint src_y = shape.height;
		block_size_x = 1024;
		block_size_y = 1;
		grid_size_x = ((src_x - (src_x % block_size_x)) / block_size_x) + 1;
		grid_size_y = ((src_y - (src_y % block_size_y)) / block_size_y) + 1;
		threads_per_block = { block_size_x, block_size_y, 1 };
		num_blocks = { grid_size_x, grid_size_y, 1 };
	}

	void kernel_2d(cv::Size shape) {
		uint src_x = shape.width;
		uint src_y = shape.height;
		block_size_x = 32;
		block_size_y = 32;
		grid_size_x = ((src_x - (src_x % block_size_x)) / block_size_x) + 1;
		grid_size_y = ((src_y - (src_y % block_size_y)) / block_size_y) + 1;
		threads_per_block = { block_size_x, block_size_y, 1 };
		num_blocks = { grid_size_x, grid_size_y, 1 };
	}

	int closest_greater_power_of_two(int input) {
		return powf(2, ceil(log2(input)));
	}

	d_Mat exclusive_scan(d_Mat input, int& sum) {
		DECLARE_HOST_AND_DEVICE_POINTERS(int, true_K);
		int closest_greater_power = closest_greater_power_of_two(num_superpixels);
		int N = closest_greater_power;
		d_Mat results = input;
		d_Mat buffer = results;

		int step_up = 1;
		for (int i = 0; i < log2(closest_greater_power); i++) {
			buffer = results;
			N /= 2;
			step_up *= 2;
			LAUNCH_KERNEL(exclusive_scan_upsweep, kernel_1d(cv::Size(N, 1)), (N, step_up, results, buffer, d_true_K));
			results = buffer;
		}

		N = 1;
		int step_down = closest_greater_power;
		for (int i = log2(closest_greater_power); i > 0; i--) {
			buffer = results;
			N *= 2;
			step_down /= 2;
			LAUNCH_KERNEL(exclusive_scan_downsweep, kernel_1d(cv::Size(N, 1)), (N, step_down, results, buffer));
			results = buffer;
		}
	}

	void display_result() {
		cv::imshow("source image", source);
		cv::imshow("segmented image", result);
		cv::waitKey(0);
	}

	void reset_flag() {
		flag = 0;
		CUDA_FUNCTION_CALL(cudaMalloc((void**)&d_flag, sizeof(uint)));
		CUDA_FUNCTION_CALL(cudaMemcpy(d_flag, h_flag, sizeof(uint), cudaMemcpyHostToDevice));
	}

	void read_flag() {
		CUDA_FUNCTION_CALL(cudaMemcpy(d_flag, h_flag, sizeof(uint), cudaMemcpyHostToDevice));
		CUDA_FUNCTION_CALL(cudaFree(d_flag));
	}

	#pragma endregion

	namespace SLIC {

		#pragma region SLIC declarations

		const int displacement_threshold = 1;
		const float density = 0.5;
		const int superpixel_size_factor = 10;
		const int size_threshold = (superpixel_size_factor * superpixel_size_factor) / 2;
		
		int source_rows, source_cols, num_pixels;
		int SP_rows, SP_cols, num_superpixels;
		int space_between_centers;
		int density_modifier;


		d_Mat center_rows, center_cols, center_grid;
		d_Mat row_sums, col_sums, num_instances;

		DECLARE_HOST_AND_DEVICE_POINTERS(int, displacement);
		#pragma endregion

		void initialize() {

			//initialize values
			source_rows = source.rows;
			source_cols = source.cols;
			num_pixels = source_rows * source_cols; 

			space_between_centers = sqrt(num_pixels) / superpixel_size_factor;
			density_modifier = density / space_between_centers;

			SP_rows = floor(source_rows/space_between_centers);
			SP_cols = floor(source_cols/space_between_centers);
			num_superpixels = SP_rows * SP_cols;

			d_Mat labels(cv::Size(source_rows, source_cols), CV_32SC1);

			d_Mat center_rows(cv::Size(SP_cols, SP_rows), CV_32SC1);
			d_Mat center_cols(cv::Size(SP_cols, SP_rows), CV_32SC1);
			d_Mat center_grid(cv::Size(SP_cols, SP_rows), CV_32SC1);

			d_Mat row_sums(cv::Size(num_pixels, 1), CV_32SC1);
			d_Mat col_sums(cv::Size(num_pixels, 1), CV_32SC1);
			d_Mat num_instances(cv::Size(num_pixels, 1), CV_32SC1);

		}

		void sample_centers() {

			LAUNCH_KERNEL(SLIC_initialize_centers, kernel_2d(center_grid.size()), (L_src, center_rows, center_cols));

		}

		void assign_pixels_to_centers() {

			LAUNCH_KERNEL(SLIC_assign_pixels_to_centers, kernel_2d(labels.size()), (L_src, A_src, B_src, density_modifier, center_rows, center_cols, labels));

		}

		void reset_displacement() {
			displacement = 0;
			CUDA_FUNCTION_CALL(cudaMalloc((void**)&d_displacement, sizeof(int)));
			CUDA_FUNCTION_CALL(cudaMemcpy(d_displacement, h_displacement, sizeof(int), cudaMemcpyHostToDevice));
		}

		void read_displacement() {
			CUDA_FUNCTION_CALL(cudaMemcpy(d_displacement, h_displacement, sizeof(int), cudaMemcpyHostToDevice));
			CUDA_FUNCTION_CALL(cudaFree(d_displacement));
		}

		void update_centers() {
			LAUNCH_KERNEL(SLIC_condense_labels, kernel_2d(labels.size()), (labels, row_sums, col_sums, num_instances));
			//first we sum the rows, columns of each set of pixels which shares a label, and we record how many pixels there were of each label.

			reset_displacement();
			LAUNCH_KERNEL(SLIC_update_centers, kernel_1d(num_instances.size()), (row_sums, col_sums, num_instances, center_rows, center_cols));
			//then we derive the average row and column for each label, and move the center correspoding to that label to that space. additionally, 
			//we take note of how far it moved to monitor convergence.
			read_displacement();
		}

		void separate_blobs() {
			d_Mat working_labels = labels;

			while (flag != 0) {
				reset_flag();

				LAUNCH_KERNEL(SLIC_separate_blobs, kernel_2d(labels.size()), (labels, working_labels, d_flag));
				//here we are using a cellular automaton. first we assign each pixel its linear ID from 0 - num_pixels. then each pixel looks at the neighbors with which it
				//originally shared a label, and adopts the numerically highest label that it sees. This repeats until the image no longer changes.

				read_flag();
			}

		}

		void absorb_small_blobs() {		
			d_Mat cluster_sizes(cv::Size(num_superpixels, 1), CV_32SC1);
			d_Mat cluster_strengths(cv::Size(num_superpixels, 1), CV_32SC1);
			d_Mat working_labels = labels;
			
			LAUNCH_KERNEL(SLIC_find_sizes, kernel_2d(labels.size()), (labels, cluster_sizes));
			//first we simply record the size of each blob in pixels


			LAUNCH_KERNEL(SLIC_find_weak_labels, kernel_1d(cluster_sizes.size()), (cluster_sizes, cluster_strengths, size_threshold));
			//then, we compare each blob size to a prescribed threshold, and assign a binary flag indicating whether that blob surpasses or falls short of the threshold.


			while (flag != 0) {
				reset_flag();

				LAUNCH_KERNEL(SLIC_absorb_small_blobs, kernel_2d(labels.size()), (labels, cluster_strengths, cluster_sizes, working_labels, d_flag));
				//we are using a celluar automaton in almost the exact same way as we did in separate_blobs, except this time we are looking at the sizes of the blobs,
				//and whether they fall below the prescribed size threshold.

				read_flag();
			}

		}

		void produce_ordered_labels() {

			d_Mat flags(cv::Size(num_pixels, 1), CV_32SC1, cv::Scalar{ 0 });
			LAUNCH_KERNEL(SLIC_raise_flags, kernel_2d(labels.size()), (labels, flags));
			//first, note that all our possible labels lie within 0 - num_pixels. To count them, we will first have each pixel raise a binary flag at the index of a 0 - num_pixels list
			//correlating to the value of its label. Many threads will write to the same flag, but this is ok because we're only raising them.

			int true_K;
			d_Mat exclusive_scan_flags = exclusive_scan(flags, true_K);
			num_superpixels = true_K;
			//since the number of ones in the array correlates to the position and number of labels, running it through an exclusive scan will yield both the true number of labels, 
			//and an array which contains the information concerning their positions within the array of 0 - num_pixels.

			d_Mat condensed_map(cv::Size(true_K, 1), CV_32SC1);
			LAUNCH_KERNEL(SLIC_init_map, kernel_1d(flags.size()), (flags, exclusive_scan_flags, condensed_map));
			//this is where it gets a bit tricky. since every element of the exclusive scan array contains the sum of all elements before it, and since our original array of flags 
			//contained only ones, we can effectively "pop" all of the unused labels. this leaves us with a 0 - true_K map, with the new labels as indices and the old labels as values.

			d_Mat useful_map = flags;
			LAUNCH_KERNEL(SLIC_invert_map, kernel_1d(useful_map.size()), (condensed_map, useful_map));
			//we take the condensed map, and we invert it so that the indices now correlate to the old labels, and the values correlate to their new 0 - true_K labels. We will use this
			//map to assign each old label (which fell sparsely between 0 - num_pixels) to their new labels, which fit perfectly between 0 - true_K.

			LAUNCH_KERNEL(SLIC_assign_new_labels, kernel_1d(labels.size()), (labels, useful_map));
			//now we simply apply these new labels according to the old labels and the map that we produced.
		}

		void enforce_connectivity() {
			//after the main body of the algorithm has converged, we are left with an image of the superpixels written in labels 0-K. Or at least, we would be if we were perfectly
			//confident that all superpixels intialized at the beginning of the algorithm survived, contiuous and unatrophied. In reality, we cannot assume that, so we must assure it.

			separate_blobs();
			//the first step is to make sure that all pixels which share a label fall in the same spatially connected region. This is accomplished by relabeling all superpixels to a 
			//sparse set of values between 0 - num_pixels, such that each actually contiguous region has a unique label. We will call these uniquely-labeled continuous regions 'blobs'.

			absorb_small_blobs();
			//next, we must account for any newly created blobs which are too small to be useful as superpixels. We will do this by simply allowing sufficiently large blobs to absorb their
			//smaller neighbors. As this part is currently structured, if a small blob has several large neighbors they compete for all or nothing. It may be better to divide the territory evenly.

			produce_ordered_labels();
			//now that the actual blobs are in their final form, we must count and relabel them. This process is tricky, so it is detailed further in produce_ordered_labels.

		}

	}

	namespace AP {

		#pragma region AP declarations
		const float damping_factor = 0.5f;
		const uint difference_threshold = 10;
		const uint num_constant_cycles_for_convergence = 3;

		//values
		cv::Size AP_matrix_size;
		const int AP_matrix_type = CV_32FC1;


		d_Mat similarity_matrix, responsibility_matrix, availibility_matrix, critereon_matrix, exemplars;
		d_Mat average_superpixel_color_vectors, pixels_per_superpixel;

		DECLARE_HOST_AND_DEVICE_POINTERS(int, difference_in_exemplars);
		int constant_cycles = 0;


		#pragma endregion

		void initialize() {
			cv::Size AP_matrix_size(num_superpixels, num_superpixels);

			d_Mat similarity_matrix(AP_matrix_size, AP_matrix_type);
			d_Mat responsibility_matrix(AP_matrix_size, AP_matrix_type);
			d_Mat availibility_matrix(AP_matrix_size, AP_matrix_type, cv::Scalar{ 0 });
			d_Mat critereon_matrix(AP_matrix_size, AP_matrix_type);
			d_Mat exemplars(cv::Size(num_superpixels, 1), CV_32SC1);

			d_Mat average_superpixel_color_vectors(cv::Size(num_superpixels, 3), CV_32SC1);
			d_Mat pixels_per_superpixel(cv::Size(num_superpixels, 1), CV_32SC1);
		}

		void calculate_average_color_vectors() {
			LAUNCH_KERNEL(AP_condense_color_vectors, kernel_2d(labels.size()), (L_src, A_src, B_src, labels, num_superpixels, average_superpixel_color_vectors, pixels_per_superpixel));

			LAUNCH_KERNEL(AP_calculate_average_color_vectors, kernel_1d(pixels_per_superpixel.size()), (average_superpixel_color_vectors, pixels_per_superpixel));
		}

		void generate_similarity_matrix() {
			LAUNCH_KERNEL(AP_generate_similarity_matrix, kernel_2d(similarity_matrix.size()), (average_superpixel_color_vectors, pixels_per_superpixel, similarity_matrix));
			
			d_Mat lowest_values(cv::Size(num_superpixels, 1), CV_32SC1);		  
			DECLARE_HOST_AND_DEVICE_POINTERS(float, preference);				  
			CUDA_FUNCTION_CALL(cudaMalloc((void**)&d_preference, sizeof(float))); 
			LAUNCH_KERNEL(AP_scan_for_lowest_value, kernel_1d(lowest_values.size()), (similarity_matrix, lowest_values, d_preference));

			LAUNCH_KERNEL(AP_set_preference_values, kernel_1d(cv::Size(num_superpixels, 1)), (d_preference, similarity_matrix));
		}

		void update_responsibility_matrix() {
			LAUNCH_KERNEL(AP_update_responsibility_matrix, kernel_2d(responsibility_matrix.size()), (similarity_matrix, availibility_matrix, damping_factor, responsibility_matrix));
		}

		void update_availibility_matrix() {
			LAUNCH_KERNEL(AP_update_availibility_matrix, kernel_2d(availibility_matrix.size()), (responsibility_matrix, availibility_matrix, damping_factor));
		}

		void update_critereon_matrix() {
			LAUNCH_KERNEL(AP_calculate_critereon_matrix, kernel_2d(critereon_matrix.size()), (availibility_matrix, responsibility_matrix, critereon_matrix));
		}

		void extract_and_examine_exemplars() {
			LAUNCH_KERNEL(AP_extract_and_examine_exemplars, kernel_1d(exemplars.size()), (critereon_matrix, exemplars, d_difference_in_exemplars));
		}

		void segment_image_using_exemplars() {
			h_Mat h_exemplars;
			h_Mat h_labels;
			h_Mat h_average_superpixel_color_vectors;
			h_Mat h_region_colors(cv::Size(num_superpixels, 3), CV_32SC1);
			h_Mat h_region_num_superpixels(cv::Size(num_superpixels, 1), CV_32SC1);
			h_Mat result(labels.size(), CV_8UC3);

			exemplars.download(h_exemplars);
			labels.download(h_labels);
			average_superpixel_color_vectors.download(h_average_superpixel_color_vectors);

			for (int label = 0; label < exemplars.cols; label++) {
				int exemplar_label = h_exemplars.at<int>(0, label);
				h_region_colors.at<int>(0, exemplar_label) += h_average_superpixel_color_vectors.at<int>(0, label);
				h_region_colors.at<int>(1, exemplar_label) += h_average_superpixel_color_vectors.at<int>(1, label);
				h_region_colors.at<int>(2, exemplar_label) += h_average_superpixel_color_vectors.at<int>(2, label);
				h_region_num_superpixels.at<int>(0, exemplar_label)++;
			}
			for (int i = 0; i < exemplars.cols; i++) {
				h_region_colors.at<int>(0, i) /= h_region_num_superpixels.at<int>(0, i);
				h_region_colors.at<int>(1, i) /= h_region_num_superpixels.at<int>(0, i);
				h_region_colors.at<int>(2, i) /= h_region_num_superpixels.at<int>(0, i);
			}

			for (int row = 0; row < result.rows; row++) {
				for (int col = 0; col < result.cols; col++) {
					int label = h_labels.at<int>(row, col);
					int region = h_exemplars.at<int>(0, label);
					int L = h_region_colors.at<int>(0, region);
					int A = h_region_colors.at<int>(1, region);
					int B = h_region_colors.at<int>(2, region);

					cv::Vec3b color = {(uchar)L, (uchar)A, (uchar)B};
					result.at<cv::Vec3b>(row, col) = color;
				}
			}

			cv::cvtColor(result, result, cv::COLOR_Lab2BGR);

		}

	}

	void initialize(string source) {
		load_image(source);
		SLIC::initialize();
		AP::initialize();
	}

}

void SLIC() { using namespace SLICAP;
	//SLIC is an algorithm which oversegments an image into regularly spaced clusters of pixels with similar characteristics, known as superpixels. It is quite similar to the popular K-means algorithm, 
	//the main difference being that it explicitly takes into account spatial proximity, and seeds regularly spaced centers along a virtual grid, so as to ultimately produce a field of superpixels. 

	SLIC::sample_centers(); 
	//first, we sample a number of points from the source image to serve as centers, at regular intervals S.

	do {
		SLIC::assign_pixels_to_centers();
		//for each pixel, given the 9 centers closest to it in space, we determine which of these centers is the nearest to it according to the distance formula given by the paper.
		//this formula consists of the euclidean distance between the color vectors of the pixel and the center, plus the distance between their coordinates tempered by the 'density' parameter. 

		SLIC::update_centers();
		//for each center, given all pixels which are labeled with its ID, we calculate the geometric mean of these pixels and shift the center to this new poisition.
		//we then record the average distance that the centers were displaced, for the purpose of checking for convergence.

	} while (SLIC::displacement > SLIC::displacement_threshold);
	//once the average displacement of the centers falls below the threshold, the algorithm has converged.

	SLIC::enforce_connectivity(); 
	//after producing the superpixels, we need to ensure that regions with the same label are connected, and that the values of these labels run sequentially from 0 - K with no missed values.
	//the exact process is somewhat complex, and detailed further within the enforce_connectivity function.
}
	
void AP() { using namespace SLICAP;
	//Affinity Propagation is a clustering algorithm which associates data points by recursively passing messages between them. The task of affinity propagation is to associate each data point with
	//another data point, known as its exemplar, which fairly represents that point and others like it. The fun part is that the reulting number of clusters are not predetermined, but emerge organically.

	AP::calculate_average_color_vectors();
	//before we begin Affinity Propagation, we must process the data we would like to compare. In this case, that means taking the average CIELAB vector of each superpixel determined previosuly by SLIC,
	//and assigning to it the average of the CIELAB vectors of its constituent pixels in the original image. 

	AP::generate_similarity_matrix();
	//before we begin passing messages, we must determine the baseline similarity between each pair of data points. here the negative euclidean distance between their color vectors. after finding the similarity 
	//for each pair, we must set all elements along the diagonal to the 'preference value'. This value influences how many clusters are generated by the algorithm. We are using the lowest value in the matrix.

	while (AP::constant_cycles < AP::num_constant_cycles_for_convergence) {
		AP::update_responsibility_matrix();
		//the responsibility matrix represents the messages sent from each data point to each potential exemplar. These messages reflect how well-suited that particular potential exemplar is  
		//to serve as the exemplar for a particular data point, taking into account other potential exemplars for that data point. 

		AP::update_availibility_matrix();
		//the availibility matrix is the second set of messages, and sort of the inverse of the first. The availbility communicates from each potential exemplar to each data point how appropriate 
		//it would be for the data point to choose that potential exemplar, taking into account the support from other data points for the potential exemplar.

		AP::update_critereon_matrix();
		//the critereon matrix is simply the sum of the availibility and responsibility matrices - or, in other words, the synthesis of their messages. Each data point is assigned the greatest-valued 
		//potential exemplar. If this happens to be the data point itself, this means that the data point is in fact an exemplar.

		AP::extract_and_examine_exemplars();
		//here we actuallly carry out the process described above of extracting the exemplars from the critereon matrix. Then we check the new exemplars against the ones determined in the previous 
		//iteration. We will make note of how many have changed between the two lists, and declare convergence once no change occurs for a certain number of iterations.
	}

	AP::segment_image_using_exemplars();
	//now that we have the exemplars, we will create the final segmented image to be displayed as the result.

}

int main() {
	SLICAP::initialize("example.png");

	SLIC(); //Simple Linear Iterative Clustering: an algorithm to oversegment the image into a field of reguarly sized clusters, fittingly called superpixels.
	AP(); //Affinity Propagation: a message passing algorithm which groups data points under their 'exemplars': suitable representatives of a large number of other data points.

	SLICAP::display_result(); //we will use Affinity Propagation to associate the superpixels produced by SLIC into larger regions based on color distance, producing a segmentation of the original image. 

	return 0;
}