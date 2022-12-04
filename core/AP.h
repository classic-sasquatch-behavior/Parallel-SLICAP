#pragma once




struct AP {

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
		Launch::kernel_2d(labels.cols, labels.rows);
		AP_condense_color_vectors<<<LAUNCH>>>(L_src, A_src, B_src, labels, num_superpixels, average_superpixel_color_vectors, pixels_per_superpixel);
		SYNC_KERNEL("AP_condense_color_vectors");
		//first we count how many pixels there are within each different superpixel, and have those pixels throw the values of their CIELAB vector into a big heap through atomic add. 

		Launch::kernel_1d(pixels_per_superpixel.cols);
		AP_calculate_average_color_vectors<<<LAUNCH>>>(average_superpixel_color_vectors, pixels_per_superpixel);
		SYNC_KERNEL("AP_calculate_average_color_vectors");
		//then we will divide these summed CIELAB vectors by the number of pixels that was assigned to each superpixel, thereby extracting the average CIELAB color vector of them each.
	}

	void generate_similarity_matrix() {

		Launch::kernel_2d(similarity_matrix.cols, similarity_matrix.rows);
		AP_generate_similarity_matrix<<<LAUNCH>>>(average_superpixel_color_vectors, pixels_per_superpixel, similarity_matrix);
		SYNC_KERNEL("AP_generate_similarity_matrix");
		//first we calculate the similarity matrix. Here we are using the negative euclidean distance between each 2 data points. therefore, all intial values will be negative, 
		//and lower values indicate dissimilarity between the two points, whereas higher values (i.e. those closer to 0) indicate similarity.

		d_Mat lowest_values(cv::Size(num_superpixels, 1), CV_32SC1);		  
		DECLARE_HOST_AND_DEVICE_POINTERS(float, preference);				  
		CUDA_SAFE_CALL(cudaMalloc((void**)&d_preference, sizeof(float))); 

		Launch::kernel_2d(lowest_values.cols, lowest_values.rows);
		AP_scan_for_lowest_value<<<LAUNCH>>>(similarity_matrix, lowest_values, d_preference);
		SYNC_KERNEL("AP_scan_for_lowest_value");
		//next we will obtain the lowest value of the similarity matrix (i.e. the negative number with the greatest magnitude) to serve as our preference value. The preference
		//value influences how many clusters are generated by the algorithm. higher values create more clusters (since data points will be biased towards thinking themselves exemplars).

		Launch::kernel_1d(num_superpixels);
		AP_set_preference_values<<<LAUNCH>>>(d_preference, similarity_matrix);
		SYNC_KERNEL("AP_set_preference_values");
		//now we will set the value along the diagonal of the similarity matrix. The preference value doesnt have to be the same across all data points. In fact, differences
		//in preference value can be used to introduce bias towards certain data points becoming exemplars. Here we have no need for bias, so we are using the same value for each.
	}

	void update_responsibility_matrix() {
		Launch::kernel_2d(responsibility_matrix.cols, responsibility_matrix.rows);
		AP_update_responsibility_matrix<<<LAUNCH>>>(similarity_matrix, availibility_matrix, damping_factor, responsibility_matrix);
		SYNC_KERNEL("AP_update_responsibility_matrix");
		//the responsibility at a given point is set as follows: r(data, exemplar) = s(data, exemplar) - max{a(data, i_exemplar) + s(data, i_exemplar)} s.t. i_exemplar != exemplar
		//conceptually, the responsibility is the table of messages sent from the data points to the exemplar candidates.
	}

	void update_availibility_matrix() {

		Launch::kernel_2d(availibility_matrix.cols, availibility_matrix.rows);
		AP_update_availibility_matrix<<<LAUNCH>>>(responsibility_matrix, availibility_matrix, damping_factor);
		SYNC_KERNEL("AP_update_availibility_matrix");
		//the availibility at a given point is set as follows: a(data, exemplar) = min{0, r(exemplar exemplar) + Σ{max(0, r(i_data, exemplar))}} s.t. i_data != data or exemplar
		//conceptually, the availibility is the table of messages sent from the exemplars candidates to the data points, in a sense the inverse of the responsibility messages. 
	}

	void update_critereon_matrix() {
		Launch::kernel_2d(critereon_matrix.cols, critereon_matrix.rows);
		AP_calculate_critereon_matrix<<<LAUNCH>>>(availibility_matrix, responsibility_matrix, critereon_matrix);
		SYNC_KERNEL("AP_calculate_critereon_matrix");
		//the critereon matrix is the sum of the responsibility and the availibility matrix. for each data point, the potential exemplar with the highest critereon value is selected
		//to be the exemplar of that data point. If a point is selected as an exemplar by another, this will necessarily be reflected by that point choosing itself as its own exemplar.
	}

	void extract_and_examine_exemplars() {

		Launch::kernel_1d(exemplars.cols);
		AP_extract_and_examine_exemplars<<<LAUNCH>>>(critereon_matrix, exemplars, d_difference_in_exemplars);
		SYNC_KERNEL("AP_extract_and_examine_exemplars");
		//we extract the exemplars for each data point, and check this list against the one generated in the previous cycle. We take note of the difference between
		//the two lists and compare it in the outer loop against a threshold to determine convergence. 
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