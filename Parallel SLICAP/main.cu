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

	atomicAdd(color_vectors(0, label), L);
	atomicAdd(color_vectors(1, label), A);
	atomicAdd(color_vectors(2, label), B);

	atomicAdd(num_labels(0, label), 1);
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
__global__ void AP_generate_similarity_matrix(int_ptr color_vectors, int_ptr pixels_per_superpixel, int_ptr similarity_matrix) {
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
	int result = -sqrtf(sum);

	similarity_matrix(data, exemplar) = result;
}

__global__ void AP_scan_for_lowest_value(int_ptr similarity_matrix, int_ptr lowest_values, float* result) {
	GET_DIMS(zero, id);
	CHECK_BOUNDS(similarity_matrix); //technically works to check bounds with a square mat, because the kernel is one dimensional

	float lowest_value = 1000000;
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

__global__ void AP_set_preference_values(float preference_value, int_ptr similarity_matrix) {
	GET_DIMS(zero, id);
	CHECK_BOUNDS(similarity_matrix); //technically works to check bounds with a square mat, because the kernel is one dimensional

	similarity_matrix(id, id) = preference_value;

}
#pragma endregion

#pragma region responsibility matrix

__global__ void AP_update_responsibility_matrix(int_ptr similarity_matrix, int_ptr availibility_matrix, float damping_factor, int_ptr responsibility_matrix) {
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

__global__ void AP_update_availibility_matrix(int_ptr responsibility_matrix, int_ptr availibility_matrix, int damping_factor) {
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

__global__ void AP_calculate_critereon_matrix(int_ptr availibility_matrix, int_ptr responsiblity_matrix, int_ptr critereon_matrix) {
	GET_DIMS(data, exemplar);
	CHECK_BOUNDS(critereon_matrix);
	critereon_matrix(data, exemplar) = availibility_matrix(data, exemplar) + responsiblity_matrix(data, exemplar);
}

__global__ void AP_extract_and_examine_exemplars(int_ptr critereon_matrix, int_ptr exemplars, int* difference) {
	GET_DIMS(zero, id);
	CHECK_BOUNDS(exemplars);

	float max_value = -1000000;
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

//__global__ void SLIC_gradient_descent() {
//	GET_DIMS(row, col);
//	CHECK_BOUNDS();
//
//}

#pragma endregion

#pragma region assign pixels to centers

__global__ void SLIC_assign_pixels_to_centers(int_ptr L_src, int_ptr A_src, int_ptr B_src, int density_modifier, int_ptr K_rows, int_ptr K_cols, int_ptr labels) {
	GET_DIMS(row, col);
	CHECK_BOUNDS(L_src);

	int sector_row = CAST_DOWN(row, K_rows.rows);
	int sector_col = CAST_DOWN(col, K_rows.cols);

	int self_L = L_src(row, col);
	int self_A = L_src(row, col);
	int self_B = L_src(row, col);
	int self_channels[3] = { self_L, self_A, self_B };

	int self_coordinates[2] = { row, col };

	int min_distance = 1000000; //arbitrarily large number for comparison
	int closest_center = -1;

	FOR_NEIGHBOR(row, col, K_row, irow, icol,
		int neighbor_row = irow + row;
	int neighbor_col = icol + col;

	int actual_neighbor_row = K_rows(neighbor_row, neighbor_col);
	int actual_neighbor_col = K_cols(neighbor_row, neighbor_col);

	int neighbor_L = L_src(actual_neighbor_row, actual_neighbor_col);
	int neighbor_A = A_src(actual_neighbor_row, actual_neighbor_col);
	int neighbor_B = B_src(actual_neighbor_row, actual_neighbor_col);
	int neighbor_channels[3] = { neighbor_L, neighbor_A, neighbor_B };

	int neighbor_coordinates[2] = { actual_neighbor_row, actual_neighbor_col };
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
	atomicAdd(row_sums(0, label), row);
	atomicAdd(col_sums(0, label), col);
	atomicAdd(num_instances(0, label), 1);
}

__global__ void SLIC_update_centers(int_ptr row_sums, int_ptr col_sums, int_ptr num_instances, int_ptr center_rows, int_ptr center_cols) {
	GET_DIMS(row, col);
	CHECK_BOUNDS(row_sums);
	GET_LINEAR_ID(row, col, id, row_sums.cols);

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

__global__ void SLIC_separate_blobs(int_ptr original_labels, int_ptr working_labels, int* flag) {
	GET_DIMS(row, col);
	CHECK_BOUNDS(original_labels);
	GET_LINEAR_ID(row, col, id, original_labels.cols);

	int original_label = original_labels(row, col);
	//MAKE SURE TO INITIALIZE WORKING LABELS AS ALL 0s!

	int working_label = working_labels(row, col);
	if (working_label < id) { working_label = id; }

	FOR_NEIGHBOR(irow, icol, original_labels, row, col,
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
	atomicAdd(label_sizes(0, label), 1);
}

__global__ void SLIC_find_weak_labels(int_ptr label_sizes, int_ptr label_strengths, int threshold) {
	GET_DIMS(row, col);
	CHECK_BOUNDS(label_strengths);
	GET_LINEAR_ID(row, col, id, label_strengths.cols);
	int label_size = label_sizes(0, id);
	if (label_size <= threshold) { label_strengths(row, col) = 1; }
}

//look over again, somethings wrong in here I can tell 
__global__ void SLIC_absorb_small_blobs(int_ptr original_labels, int_ptr label_strengths, int_ptr label_sizes, int_ptr working_labels, int* flag) {
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

__global__ void SLIC_invert_map(int_ptr old_map, int_ptr new_map) {
	GET_DIMS(id, NA);
	CHECK_BOUNDS(old_map);
	int value = old_map(0, id);
	new_map(0, value) = id;
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

	h_Mat source;
	vector<d_Mat> CIELAB_planes;
	d_Mat& L_src = CIELAB_planes[0];
	d_Mat& A_src = CIELAB_planes[1];
	d_Mat& B_src = CIELAB_planes[2];

	DECLARE_HOST_AND_DEVICE_POINTERS(uint, flag);


	#pragma endregion

	#pragma region functions

	vector<d_Mat*> split_into_channels(h_Mat input) {
		vector<h_Mat> split;
		cv::split(input, split);
		d_Mat d_L, d_A, d_B;
		d_L.upload(split[0]);
		d_A.upload(split[1]);
		d_B.upload(split[2]);
		return { &d_L, &d_A, &d_B };
	}

	void load_image(string source) {
		h_Mat CIELAB_src;
		BGR_src = cv::imread(source, cv::IMREAD_COLOR);
		cv::cvtColor(BGR_src, CIELAB_src, cv::COLOR_BGR2Lab);
		CIELAB_planes = split_into_channels(CIELAB_src);
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
		d_Mat exclusive_sum_flags = flags;
		DECLARE_HOST_AND_DEVICE_POINTERS(int, true_K);
		int closest_greater_power = closest_greater_power_of_two(num_superpixels);
		int N = closest_greater_power;
		d_Mat results = exclusive_sum_flags;
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
		cv::imshow("source image", source_image);
		cv::imshow("segmented image", segmented_image);
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
		
		int source_rows, source_cols, num_pixels;
		int SP_rows, SP_cols, num_superpixels;
		int space_between_centers;
		int density_modifier;

		d_Mat labels;
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

		void reset_displacement() {
			displacement = 0;
			CUDA_FUNCTION_CALL(cudaMalloc((void**)&d_displacement, sizeof(int)));
			CUDA_FUNCTION_CALL(cudaMemcpy(d_displacement, h_displacement, sizeof(int), cudaMemcpyHostToDevice));
		}

		void read_displacement() {
			CUDA_FUNCTION_CALL(cudaMemcpy(d_displacement, h_displacement, sizeof(int), cudaMemcpyHostToDevice));
			CUDA_FUNCTION_CALL(cudaFree(d_displacement));
		}

		void sample_centers() {

			LAUNCH_KERNEL(SLIC_initialize_centers, kernel_2d(center_grid.size()), (source, center_rows, center_cols));

		}

		void assign_pixels_to_centers() {

			LAUNCH_KERNEL(SLIC_assign_pixels_to_centers, kernel_2d(labels.size()), (L_src, A_src, B_src, center_rows, center_cols, labels));

		}

		void update_centers() {
			LAUNCH_KERNEL(SLIC_condense_labels, kernel_2d(labels.size()), (labels, row_sums, col_sums, num_instances));

			reset_displacement();
			LAUNCH_KERNEL(SLIC_update_centers, kernel_1d(num_instances.size()), (row_sums, col_sums, num_instances, center_rows, center_cols));
			read_displacement();
		}

		void separate_blobs() {
			d_Mat working_labels = labels;

			while (flag != 0) {
				reset_flag();

				LAUNCH_KERNEL(SLIC_separate_blobs, kernel_2d(labels.size()), (labels, working_labels, d_flag));

				read_flag();
			}

		}

		void absorb_small_blobs() {		
			d_Mat cluster_sizes(cv::Size(num_superpixels, 1), CV_32SC1);
			d_Mat cluster_strengths(cv::Size(num_superpixels, 1), CV_32SC1);
			d_Mat working_labels = labels;
			
			LAUNCH_KERNEL(SLIC_find_sizes, kernel_2d(labels.size()), (labels, cluster_sizes));

			LAUNCH_KERNEL(SLIC_find_weak_labels, kernel_1d(cluster_sizes.size()), (cluster_sizes, cluster_strengths, size_threshold));

			while (flag != 0) {
				reset_flag();
				LAUNCH_KERNEL(SLIC_absorb_small_blobs, kernel_2d(labels.size()), (labels, cluster_strengths, cluster_sizes, working_labels, flag))
				read_flag();
			}

		}

		void produce_ordered_labels() {
			d_Mat flags(cv::Size(num_pixels, 1), CV_32SC1, cv::Scalar{ 0 });
			int true_K;

			LAUNCH_KERNEL(SLIC_raise_flags, kernel_2d(labels.size()), (labels, flags));

			d_Mat exclusive_scan_flags = exclusive_scan(flags, true_K);

			d_Mat condensed_map(cv::Size(true_K, 1), CV_32SC1);
			LAUNCH_KERNEL(SLIC_init_map, kernel_1d(condensed_map.size()), (flags, results, condensed_map));

			d_Mat useful_map = flags;
			LAUNCH_KERNEL(SLIC_invert_map, kernel_1d(useful_map.size()), (condensed_map, useful_map));

			LAUNCH_KERNEL(SLIC_assign_new_labels, kernel_1d(labels.size()), (labels, useful_map));
			
		}

		void enforce_connectivity() {

			separate_blobs();

			absorb_small_blobs();

			produce_ordered_labels();

		}

	}

	namespace AP {

		#pragma region AP declarations



		#pragma endregion

		void initialize() {
			bool initialized;
			//parameters
			const float damping_factor = 0.5f;
			const uint difference_threshold = 10;
			//const uint num_constant_cycles_for_convergence = 3;

			//values
			cv::Size AP_matrix_size(num_superpixels, num_superpixels);
			const int AP_matrix_type = CV_32FC1;

			//matrices
			d_Mat similarity_matrix(AP_matrix_size, AP_matrix_type);
			d_Mat responsibility_matrix(AP_matrix_size, AP_matrix_type);
			d_Mat availibility_matrix(AP_matrix_size, AP_matrix_type, cv::Scalar{ 0 });
			d_Mat critereon_matrix(AP_matrix_size, AP_matrix_type);
			d_Mat exemplars(cv::Size(num_superpixels, 1), CV_32SC1);


			DECLARE_HOST_AND_DEVICE_POINTERS(int, difference_in_exemplars);
			//int constant_cycles = 0;
			bool converged = false;



			result = segment_image_using_exemplars(exemplars, labels, average_superpixel_color_vectors);
		}

		void AP_calculate_average_color_vectors() {
			UNPACK(src_planes, L_src, A_src, B_src);

			d_Mat average_superpixel_color_vectors(cv::Size(num_superpixels, 3), CV_32SC1);
			d_Mat pixels_per_superpixel(cv::Size(num_superpixels, 1), CV_32SC1);

			conf_2d_kernel(labels.size());
			AP_condense_color_vectors << <num_blocks, threads_per_block >> > \
				(L_src, A_src, B_src, labels, num_superpixels, average_superpixel_color_vectors, pixels_per_superpixel);
			SYNC_AND_CHECK_FOR_ERRORS(AP_condense_color_vectors);

			AP_calculate_average_color_vectors << <num_blocks, threads_per_block >> > \
				(average_superpixel_color_vectors, pixels_per_superpixel);
			SYNC_AND_CHECK_FOR_ERRORS(AP_calculate_average_color_vectors);
		}

		void AP_generate_similarity_matrix() {
			//initialize similarity matrix
			conf_2d_kernel(similarity_matrix.size());										  //defined in [3]
			AP_generate_similarity_matrix << <num_blocks, threads_per_block >> >				  /**/ \
				(average_superpixel_color_vectors, pixels_per_superpixel, similarity_matrix); //
			SYNC_AND_CHECK_FOR_ERRORS(AP_generate_similarity_matrix);						  //

			d_Mat lowest_values(cv::Size(num_superpixels, 1), CV_32SC1);		  //
			DECLARE_HOST_AND_DEVICE_POINTERS(float, preference);				  //
			CUDA_FUNCTION_CALL(cudaMalloc((void**)&d_preference, sizeof(float))); //
			conf_1d_kernel(lowest_values.size());								  //
			AP_scan_for_lowest_value << <num_blocks, threads_per_block >> >		  /**/ \
				(similarity_matrix, lowest_values, d_preference);				  //
			SYNC_AND_CHECK_FOR_ERRORS(AP_scan_for_lowest_value);				  //

			//set diagonal of similarity matrix
			conf_1d_kernel(cv::Size(num_superpixels, 1));				 //
			AP_set_preference_values << <num_blocks, threads_per_block >> > /**/ \
				(d_preference, similarity_matrix);						 //
			SYNC_AND_CHECK_FOR_ERRORS(AP_set_preference_values);		 //
		}

		void AP_update_responsibility_matrix() {
			conf_2d_kernel(responsibility_matrix.size());										 //
			AP_update_responsibility_matrix << <num_blocks, threads_per_block >> >					 /**/ \
				(similarity_matrix, availibility_matrix, damping_factor, responsibility_matrix); //
			SYNC_AND_CHECK_FOR_ERRORS(AP_update_responsibility_matrix);
		}

		void AP_update_availibility_matrix() {
			conf_2d_kernel(availibility_matrix.size());						  //
			AP_update_availibility_matrix << <num_blocks, threads_per_block >> > /**/ \
				(responsibility_matrix, availibility_matrix, damping_factor); //
			SYNC_AND_CHECK_FOR_ERRORS(AP_update_availibility_matrix);		  //
		}

		void AP_update_critereon_matrix() {
			conf_2d_kernel(critereon_matrix.size());						   //
			AP_calculate_critereon_matrix << <num_blocks, threads_per_block >> >  /**/ \
				(availibility_matrix, responsiblity_matrix, critereon_matrix); //
			SYNC_AND_CHECK_FOR_ERRORS(AP_update_critereon_matrix);			   //

		}

		void AP_extract_and_examine_exemplars() {
			d_Mat exemplars(cv::Size(num_superpixels, 1), CV_32SC1);			 //
			conf_1d_kernel(exemplars.size());									 //
			AP_extract_and_examine_exemplars << <num_blocks, threads_per_block >> > /**/ \
				(critereon_matrix, exemplars, d_difference_in_exemplars);		 //
			SYNC_AND_CHECK_FOR_ERRORS(AP_extract_and_examine_exemplars);		 //
		}

		void segment_image_using_exemplars() {
			h_Mat exemplars;
			h_Mat labels;
			h_Mat average_colors;

			d_exemplars.download(exemplars);
			d_labels.download(labels);
			d_average_colors.download(average_colors);
			h_Mat result(labels.size(), CV_8UC3);

			for (int label = 0; label < exemplars.cols; label++) {

			}

			for (int row = 0; row < result.rows; row++) {
				for (int col = 0; col < result.cols; col++) {
					int label = labels.at<int>(row, col);
					cv::Vec3b color = average_colors.at<cv::Vec3b>(0, label);
					result.at<cv::Vec3b>(row, col) = color;
				}
			}
		}

	}

	void initialize(string source) {
		load_image(source);
		SLIC::initialize();
		AP::initialize();
	}

}

void SLIC() { using namespace SLICAP;

	//SLIC is an algorithm which oversegments an image into regularly spaced clusters of pixels with similar characteristics, known as superpixels. 
	//It is quite similar to the popular K-means algorithm, the main difference being that it explicitly takes into account spatial proximity, 
	//and seeds regularly spaced centers along a virtual grid, so as to ultimately produce a field of superpixels. 

	SLIC::sample_centers(); 
	//first, we sample a number of points from the source image to serve as centers, at regular intervals S

	REPEAT_UNTIL_CONVERGENCE((SLIC::displacement <= SLIC::displacement_threshold),

		SLIC::assign_pixels_to_centers(); 
		//for each pixel, given the 9 centers closest to it in space, we determine which of these centers is the nearest to it according to the distance formula:								  
		//[distance formula]
		//a given pixel will be labeled with the ID of the center which minimizes this equation.

		SLIC::update_centers(); 
		//for each center, given all pixels which are labeled with its ID, we calculate the geometric mean of these pixels and shift the center to this poisition.

	);

	SLIC::enforce_connectivity(); 
	//after producing the superpixels, we wish to know the exact number of superpixels produced by the algorithm, to ensure that 
	//regions with the same label are connected, and that the values of these labels run sequentially from 0 - K with no missed values.
}
	
void AP() { using namespace SLICAP;

	//Affinity Propagation is an algorithm which associates data points by means of message passing.
	
	AP::calculate_average_color_vectors();

	AP::generate_similarity_matrix();

	REPEAT_UNTIL_CONVERGENCE((AP::constant_cycles >= AP::num_constant_cycles_for_convergence),
		
		AP::update_responsibility_matrix();

		AP::update_availibility_matrix();

		AP::update_critereon_matrix();

		AP::extract_and_examine_exemplars();

	);
}

int main() {
	SLICAP::initialize("example.png");

	SLIC(); 
	//Simple Linear Iterative Clustering: an algorithm to oversegment the image into a field of reguarly sized clusters, fittingly called superpixels.

	AP(); 
	//Affinity Propagation: a message passing algorithm which groups data points under their 'exemplars': data points which are 
	//suitably representative of a large number of other data points.

	SLICAP::display_result(); 
	//we will use Affinity Propagation to associate the superpixels produced by SLIC into larger regions based on color distance, 
	//producing a segmentation of the original image. 

	return 0;
}