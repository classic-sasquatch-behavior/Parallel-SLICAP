#include"macros.h"


#pragma region parameters

const std::string image_filename = "example.png"; //the image to be segmented

#pragma endregion

#pragma region declarations

//these parameters are defined globally in order to cooperate with the kernel configuration functions
uint block_size_x; //threads per block in the x dimension
uint block_size_y; //threads per block in the y dimension
uint grid_size_x;  //blocks within grid in the x dimension
uint grid_size_y;  //blocks within grid in the y dimension
dim3 threads_per_block; //the actual threads per block to be passed into the kernel 
dim3 num_blocks; //the actual number of blocks to be passed into the kernel

bool converged = false;

#pragma endregion

#pragma region structs
static struct SLIC {
public:
	SLIC() {
		//initialize values
		int dmod = density / S;					 //factor which will be used to temper the influence of space on the result of the distance function
		int src_rows = labels.rows;				 //the number of actual rows in the source image
		int src_cols = labels.cols;				 //the number of actual columns in the source image
		int num_pixels = src_rows * src_cols;	 //the total number of pixels in the source image
		int SP_rows = floor(src_rows / S);		 //the number of superpixels in each row of the resulting image
		int SP_cols = floor(src_cols / S);		 //the number of superpixels in each column of the resulting image
		int num_superpixels = SP_rows * SP_cols; //the predicted number of superpixels (subject to variance, which will be addressed when we enforce connectivity)
		int& K = num_superpixels;				 //alias for the number of superpixels which conforms to the language of the original paper

			//bring source channels into scope
		d_Mat L_src, A_src, B_src;
		UNPACK(src_planes, L_src, A_src, B_src);

		//initialize centers
		d_Mat d_center_rows(cv::Size(SP_cols, SP_rows), CV_32SC1);
		d_Mat d_center_cols(cv::Size(SP_cols, SP_rows), CV_32SC1);

		//initialize matrices to support update_centers
		d_Mat row_sums(cv::Size(num_pixels, 1), CV_32SC1);
		d_Mat col_sums(cv::Size(num_pixels, 1), CV_32SC1);
		d_Mat num_instances(cv::Size(num_pixels, 1), CV_32SC1);

		bool converged = false;
		DECLARE_HOST_AND_DEVICE_POINTERS(float, displacement);
		CUDA_FUNCTION_CALL(cudaMalloc((void**)&d_displacement, sizeof(uint)));

		result = segment_image_using_exemplars(exemplars, labels, average_superpixel_color_vectors);
	}

	static void initialize_centers() {

	}

	static void assign_pixels_to_centers() {

	}

	static void update_centers() {

	}

	static void enforce_connectivity() {

	}

	static int displacement;
	static float convergence_threshold;

private:
	//parameters
	bool initialized = false;

	const uint SP_size_factor = 10; 
	const uint& S = SP_size_factor; 
	const uint density = 10;		
	//const float convergence_threshold = 1.0f;
	const uint enforce_connectivity_size_threshold = (S * S) / 2;

	int dmod;
	int src_rows, src_cols, num_pixels;
	int sector_rows, sector_cols, num_superpixels;
	d_Mat L_src, A_src, B_src;
	d_Mat center_rows, center_cols;
	d_Mat row_sums, col_sums, num_instances;
};

static struct AP {
public:
	AP() {

	}

	static void calculate_average_color_vectors() {

	}

	static void generate_similarity_matrix() {

	}

	static void update_responsibility_matrix() {

	}

	static void update_availibility_matrix() {

	}

	static void update_critereon_matrix() {

	}

	static void extract_and_examine_exemplars() {

	}

	static int constant_cycles;
	static int num_constant_cycles_for_convergence;

private:
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



};

static struct META {
public:

private:
};
#pragma endregion

#pragma region general kernels

__global__ void GEN_exclusive_scan_upsweep(int N, int step, int_ptr source, int_ptr buffer, int* sum) {
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

__global__ void GEN_exclusive_scan_downsweep(int N, int step, int_ptr source, int_ptr buffer) {
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

	__global__ void AP_calculate_average_color_vectors( int_ptr color_vectors , int_ptr pixels_per_superpixel) {
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

	#pragma endregion----------[5.1]

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

	__global__ void AP_set_preference_values( float preference_value, int_ptr similarity_matrix) {
		GET_DIMS(zero, id);
		CHECK_BOUNDS(similarity_matrix); //technically works to check bounds with a square mat, because the kernel is one dimensional

		similarity_matrix(id, id) = preference_value;

	}
	#pragma endregion----------------[5.2]

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

	#pragma endregion------------[5.3]

	#pragma region availibility matrix

	__global__ void AP_update_availibility_matrix(int_ptr responsibility_matrix, int_ptr availibility_matrix, int damping_factor) {
		GET_DIMS(data, exemplar);
		CHECK_BOUNDS(availibility_matrix);

		float sum = 0;
		for (int i_data = 0; i_data < availibility_matrix.rows; i_data++) {
			if ((i_data == data)||(i_data == exemplar)) { continue; }
			sum += fmaxf(0, responsibility_matrix(i_data, exemplar));
		}

		float result = responsibility_matrix(data, exemplar) + sum;
		result = (damping_factor * (availibility_matrix(data, exemplar))) + ((1.0 - damping_factor) * (result));
		result = fminf(0, result);
		availibility_matrix(data, exemplar) = result;

	}

	#pragma endregion--------------[5.4]

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

	#pragma endregion-----------------[5.5]

#pragma endregion

#pragma region SLIC kernels

	#pragma region initialize centers

	__global__ void SLIC_initialize_centers(int_ptr K_rows, int_ptr K_cols, int_ptr src) {
		GET_DIMS(col, row);
		CHECK_BOUNDS(K_rows);
		K_rows(row, col) = CAST_UP(row, K_rows.rows, src.rows);
		K_cols(row, col) = CAST_UP(col, K_rows.cols, src.cols);
	}

	__global__ void SLIC_gradient_descent() {
		GET_DIMS(row, col);
		CHECK_BOUNDS();

	}

	#pragma endregion-------------[4.1]

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

	#pragma endregion-------[4.2]

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

	#pragma endregion-----------------[4.3]

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

	#pragma endregion-----------------[4.4]

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

	#pragma endregion-------------[4.5]

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

	#pragma endregion---------[4.6]

#pragma endregion

#pragma region general support functions

	void initialize_sources(h_Mat BGR_src, std::vector<d_Mat*>& CIELAB_planes) {
		h_Mat CIELAB_src;
		BGR_src = cv::imread(image_filename, cv::IMREAD_COLOR);
		cv::cvtColor(BGR_src, CIELAB_src, cv::COLOR_BGR2Lab);
		CIELAB_planes = split_into_channels(CIELAB_src); //defined below
	}

	std::vector<d_Mat*> split_into_channels(h_Mat input) {
		std::vector<h_Mat> split;
		cv::split(input, split);
		d_Mat d_L, d_A, d_B;
		d_L.upload(split[0]);
		d_A.upload(split[1]);
		d_B.upload(split[2]);
		return { &d_L, &d_A, &d_B };
	}

	void conf_1d_kernel(cv::Size shape) {
		uint src_x = shape.width;
		uint src_y = shape.height;
		block_size_x = 1024;
		block_size_y = 1;
		grid_size_x = ((src_x - (src_x % block_size_x)) / block_size_x) + 1;
		grid_size_y = ((src_y - (src_y % block_size_y)) / block_size_y) + 1;
		threads_per_block = { block_size_x, block_size_y, 1 };
		num_blocks = { grid_size_x, grid_size_y, 1 };
	}

	void conf_2d_kernel(cv::Size shape) {
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

#pragma endregion

#pragma region AP support functions

	h_Mat segment_image_using_exemplars(d_Mat d_exemplars, d_Mat d_labels, d_Mat d_average_colors) {
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

#pragma endregion

#pragma region SLIC support functions

	void initialize_centers(d_Mat source, d_Mat& rows_out, d_Mat& cols_out) {

		conf_2d_kernel(rows_out.size());							
		SLIC_initialize_centers <<<num_blocks, threads_per_block>>> \
			(source, rows_out, cols_out);							
		SYNC_AND_CHECK_FOR_ERRORS(SLIC_initialize_centers);			

		conf_2d_kernel(rows_out.size());						 
		SLIC_gradient_descent <<<num_blocks, threads_per_block>>> \
			();													  
		SYNC_AND_CHECK_FOR_ERRORS(SLIC_gradient_descent);		  

	}

	void enforce_connectivity(int size_threshold, int num_superpixels, d_Mat labels) {

	launch_SLIC_separate_blobs();

	launch_SLIC_absorb_small_blobs();

	launch_SLIC_produce_ordered_labels();

	}

#pragma endregion

#pragma region AP launch functions
	void launch_AP_calculate_average_color_vectors() {
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

	void launch_AP_generate_similarity_matrix() {
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

	void launch_AP_update_responsibility_matrix() {
		conf_2d_kernel(responsibility_matrix.size());										 //
		AP_update_responsibility_matrix << <num_blocks, threads_per_block >> >					 /**/ \
			(similarity_matrix, availibility_matrix, damping_factor, responsibility_matrix); //
		SYNC_AND_CHECK_FOR_ERRORS(AP_update_responsibility_matrix);
	}

	void launch_AP_update_availibility_matrix() {
		conf_2d_kernel(availibility_matrix.size());						  //
		AP_update_availibility_matrix << <num_blocks, threads_per_block >> > /**/ \
			(responsibility_matrix, availibility_matrix, damping_factor); //
		SYNC_AND_CHECK_FOR_ERRORS(AP_update_availibility_matrix);		  //
	}

	void launch_AP_update_critereon_matrix() {
		conf_2d_kernel(critereon_matrix.size());						   //
		AP_calculate_critereon_matrix << <num_blocks, threads_per_block >> >  /**/ \
			(availibility_matrix, responsiblity_matrix, critereon_matrix); //
		SYNC_AND_CHECK_FOR_ERRORS(AP_update_critereon_matrix);			   //

	}

	void launch_AP_extract_and_examine_exemplars() {
		d_Mat exemplars(cv::Size(num_superpixels, 1), CV_32SC1);			 //
		conf_1d_kernel(exemplars.size());									 //
		AP_extract_and_examine_exemplars << <num_blocks, threads_per_block >> > /**/ \
			(critereon_matrix, exemplars, d_difference_in_exemplars);		 //
		SYNC_AND_CHECK_FOR_ERRORS(AP_extract_and_examine_exemplars);		 //
	}
	

#pragma endregion

#pragma region SLIC launch functions

	void launch_SLIC_assign_pixels_to_centers() {
		conf_2d_kernel(labels.size());
		SLIC_assign_pixels_to_centers << <num_blocks, threads_per_block >> > \
			(L_src, A_src, B_src, center_rows, center_cols, labels);
		SYNC_AND_CHECK_FOR_ERRORS(SLIC_assign_pixels_to_centers);
	}

	void launch_SLIC_update_centers() {
		conf_2d_kernel(labels.size());
		SLIC_condense_labels << <num_blocks, threads_per_block >> > \
			(labels, row_sums, col_sums, num_instances);
		SYNC_AND_CHECK_FOR_ERRORS(SLIC_condense_labels);

		conf_1d_kernel(num_instances.size());
		SLIC_update_centers << <num_blocks, threads_per_block >> > \
			(row_sums, col_sums, num_instances, center_rows, center_cols);
		SYNC_AND_CHECK_FOR_ERRORS(SLIC_update_centers);

		CUDA_FUNCTION_CALL(cudaMemcpy(h_displacement, d_displacement, sizeof(uint), cudaMemcpyDeviceToHost));
	}

	void launch_SLIC_separate_blobs() {
		d_Mat working_labels = labels;

		//separate_blobs
		DECLARE_HOST_AND_DEVICE_POINTERS(int, flag);
		flag = 0;
		CUDA_FUNCTION_CALL(cudaMalloc((void**)&d_flag, sizeof(int)));

		bool converged = false;
		while (!converged) {

			flag = 0;
			CUDA_FUNCTION_CALL(cudaMemcpy(d_flag, h_flag, sizeof(int), cudaMemcpyHostToDevice));

			conf_2d_kernel(labels.size());
			SLIC_separate_blobs << <num_threads, threads_per_block >> > \
				(labels, working_labels, d_flag);
			SYNC_AND_CHECK_FOR_ERRORS(SLIC_separate_blobs);

			CUDA_FUNCTION_CALL(cudaMemcpy(h_flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost));

			if (flag == 0) {
				converged = true;
			}
		}
	}

	void launch_SLIC_absorb_small_blobs() {		//find sizes
		d_Mat cluster_sizes(cv::Size(num_superpixels, 1), CV_32SC1);
		conf_2d_kernel(labels.size());
		SLIC_find_sizes << <num_threads, threads_per_block >> > \
			(labels, cluster_sizes);
		SYNC_AND_CHECK_FOR_ERRORS(SLIC_find_sizes);

		//find weak labels
		d_Mat cluster_strengths(cv::Size(num_superpixels, 1), CV_32SC1);
		conf_1d_kernel(cluster_sizes.size());
		SLIC_find_weak_labels << <num_threads, threads_per_block >> > \
			(cluster_sizes, cluster_strengths, size_threshold);
		SYNC_AND_CHECK_FOR_ERRORS(SLIC_find_weak_labels);

		//absorb small blobs
		flag = 0;
		CUDA_FUNCTION_CALL(cudaMalloc((void**)&d_flag, sizeof(int)));
		CUDA_FUNCTION_CALL(cudaMemcpy(d_flag, h_flag, sizeof(int), cudaMemcpyHostToDevice));
		converged = false;

		while (!converged) {
			conf_2d_kernel(labels.size());
			SLIC_absorb_small_blobs << <num_threads, threads_per_block >> > \
				(labels, cluster_strengths, cluster_sizes, working_labels, flag);
			SYNC_AND_CHECK_FOR_ERRORS(SLIC_absorb_small_blobs);
			CUDA_FUNCTION_CALL(cudaMemcpy(h_flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost));

			if (flag == 0) {
				converged = true;
				CUDA_FUNCTION_CALL(cudaFree(d_flag));
			}
		}
	}

	void launch_SLIC_produce_ordered_labels() {	//produce ordered labels
		//raise flags
		d_Mat flags(cv::Size(num_superpixels, 1), CV_32SC1, cv::Scalar{ 0 });
		conf_2d_kernel(labels.size());
		SLIC_raise_flags << <num_threads, threads_per_block >> > \
			(labels, flags);
		SYNC_AND_CHECK_FOR_ERRORS(SLIC_raise_flags);

		//exclusive sum
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
			conf_1d_kernel(cv::Size(N, 1));
			GEN_exclusive_scan_upsweep << <num_blocks, threads_per_block >> > \
				(N, step_up, results, buffer, d_true_K);
			SYNC_AND_CHECK_FOR_ERRORS(GEN_exclusive_scan_upsweep);
			results = buffer;
		}

		N = 1;
		int step_down = closest_greater_power;
		for (int i = log2(closest_greater_power); i > 0; i--) {
			buffer = results;
			N *= 2;
			step_down /= 2;
			conf_1d_kernel(cv::Size(N, 1));
			GEN_exclusive_scan_downsweep << <num_blocks, threads_per_block >> > \
				(N, step_down, results, buffer);
			SYNC_AND_CHECK_FOR_ERRORS(GEN_exclusive_scan_downsweep);
			results = buffer;
		}

		//initialize map
		d_Mat condensed_map(cv::Size(true_K, 1), CV_32SC1);
		conf_1d_kernel(condensed_map.size());
		SLIC_init_map << <num_blocks, threads_per_block >> > \
			(flags, results, condensed_map);
		SYNC_AND_CHECK_FOR_ERRORS(SLIC_init_map);

		//invert map
		d_Mat useful_map = flags;
		conf_1d_kernel(useful_map.size());
		SLIC_invert_map << <num_blocks, threads_per_block >> > \
			(condensed_map, useful_map);
		SYNC_AND_CHECK_FOR_ERRORS(SLIC_invert_map);

		//assign new labels
		conf_1d_kernel(labels.size());
		SLIC_assign_new_labels << <num_threads, threads_per_block >> > \
			(labels, useful_map);
		SYNC_AND_CHECK_FOR_ERRORS(SLIC_assign_new_labels);
	}

#pragma endregion

#pragma region entry points

void SLIC() {

	SLIC::initialize_centers();

	RUN_UNTIL_CONVERGENCE((SLIC::displacement <= SLIC::convergence_threshold),

		SLIC::assign_pixels_to_centers();

		SLIC::update_centers();

	);

	SLIC::enforce_connectivity(); 
}
	
void AP() {
	//Affinity Propagation is an algorithm which associates data points by means of message passing.
	
	AP::calculate_average_color_vectors();

	AP::generate_similarity_matrix();

	RUN_UNTIL_CONVERGENCE((AP::constant_cycles >= AP::num_constant_cycles_for_convergence),
		
		AP::update_responsibility_matrix();

		AP::update_availibility_matrix();

		AP::update_critereon_matrix();

		AP::extract_and_examine_exemplars();

	);
}

#pragma endregion

void display_result() {
	cv::imshow("source image", META::BGR_src);
	cv::imshow("segmented image", META::segmented_image);
	cv::waitKey(0);
}

void load_image(string image_filename) {
	h_Mat CIELAB_src;
	BGR_src = cv::imread(image_filename, cv::IMREAD_COLOR);
	cv::cvtColor(BGR_src, CIELAB_src, cv::COLOR_BGR2Lab);
	CIELAB_planes = split_into_channels(CIELAB_src); //defined below
}


void initialize(string source) {
	SLIC::initialize();
	AP::initialize();

}

int main() {
	initialize("example.png");

	SLIC(); //Simple Linear Iterative Clustering: an algorithm to oversegment the image into reguarly sized clusters, fittingly called superpixels.
	AP(); //Affinity Propagation: a message passing algorithm which groups data points under their 'exemplars': data points which exemplify a large number of data points.

	display_result(META::segmented_image); //we will use Affinity Propagation to associate the superpixels produced by SLIC into larger regions, based on spatial and color distance. 

	return 0;
}