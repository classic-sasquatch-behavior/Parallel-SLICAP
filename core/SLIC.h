#pragma once

struct SLIC {

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

	SLIC(d_Mat SLIC_source) : source_rows(SLIC_source.rows), source_cols(SLIC_source.cols), num_pixels(source_rows * source_cols),
		space_between_centers(sqrt(num_pixels) / superpixel_size_factor), density_modifier(density / space_between_centers),
		SP_rows(floor(source_rows/space_between_centers)), SP_cols(floor(source_cols/space_between_centers)), num_superpixels(SP_rows * SP_cols),
		labels(cv::Size(source_cols, source_rows), CV_32SC1),
		center_rows(cv::Size(SP_cols, SP_rows), CV_32SC1), center_cols(cv::Size(SP_cols, SP_rows), CV_32SC1), center_grid(cv::Size(SP_cols, SP_rows), CV_32SC1),
		row_sums(d_Mat row_sums(cv::Size(num_pixels, 1), CV_32SC1)), col_sums(d_Mat row_sums(cv::Size(num_pixels, 1), CV_32SC1)), num_instances(d_Mat row_sums(cv::Size(num_pixels, 1), CV_32SC1))
	{

	}

	// void initialize() {

	// 	//initialize values
	// 	source_rows = source.rows;
	// 	source_cols = source.cols;
	// 	num_pixels = source_rows * source_cols; 

	// 	space_between_centers = sqrt(num_pixels) / superpixel_size_factor;
	// 	density_modifier = density / space_between_centers;

	// 	SP_rows = floor(source_rows/space_between_centers);
	// 	SP_cols = floor(source_cols/space_between_centers);
	// 	num_superpixels = SP_rows * SP_cols;

	// 	d_Mat labels(cv::Size(source_cols, source_rows), CV_32SC1);

	// 	d_Mat center_rows(cv::Size(SP_cols, SP_rows), CV_32SC1);
	// 	d_Mat center_cols(cv::Size(SP_cols, SP_rows), CV_32SC1);
	// 	d_Mat center_grid(cv::Size(SP_cols, SP_rows), CV_32SC1);

	// 	d_Mat row_sums(cv::Size(num_pixels, 1), CV_32SC1);
	// 	d_Mat col_sums(cv::Size(num_pixels, 1), CV_32SC1);
	// 	d_Mat num_instances(cv::Size(num_pixels, 1), CV_32SC1);

	// }

	void sample_centers() {

		Launch::kernel_2d(center_grid.cols, center_grid.rows);
		SLIC_initialize_centers<<<LAUNCH>>>(L_src, center_rows, center_cols);
		SYNC_KERNEL("SLIC_initialize_centers");

	}

	void assign_pixels_to_centers() {

		Launch::kernel_2d(labels.cols, labels.rows);
		SLIC_assign_pixels_to_centers<<<LAUNCH>>>(L_src, A_src, B_src, density_modifier, center_rows, center_cols, labels);
		SYNC_KERNEL("SLIC_assign_pixels_to_centers");

	}

	void reset_displacement() {
		displacement = 0;
		CUDA_SAFE_CALL(cudaMalloc((void**)&d_displacement, sizeof(int)));
		CUDA_SAFE_CALL(cudaMemcpy(d_displacement, h_displacement, sizeof(int), cudaMemcpyHostToDevice));
	}

	void read_displacement() {
		CUDA_SAFE_CALL(cudaMemcpy(d_displacement, h_displacement, sizeof(int), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaFree(d_displacement));
	}

	void update_centers() {
		Launch::kernel_2d(labels.cols, labels.rows);
		SLIC_condense_labels<<<LAUNCH>>>(labels, row_sums, col_sums, num_instances);
		SYNC_KERNEL("SLIC_condense_labels");
		//first we sum the rows, columns of each set of pixels which shares a label, and we record how many pixels there were of each label.

		reset_displacement();

		Launch::kernel_1d(num_instances.cols);
		SLIC_update_centers<<<LAUNCH>>>(row_sums, col_sums, num_instances, center_rows, center_cols);
		SYNC_KERNEL("SLIC_update_centers");
		//then we derive the average row and column for each label, and move the center correspoding to that label to that space. additionally, 
		//we take note of how far it moved to monitor convergence.
		read_displacement();
	}

	void separate_blobs() {
		d_Mat working_labels = labels;

		while (flag != 0) {
			reset_flag();

			Launch::kernel_2d(labels.cols, labels.rows);
			SLIC_separate_blobs<<<LAUNCH>>>(labels, working_labels, d_flag);
			SYNC_KERNEL("SLIC_separate_blobs");

			//here we are using a cellular automaton. first we assign each pixel its linear ID from 0 - num_pixels. then each pixel looks at the neighbors with which it
			//originally shared a label, and adopts the numerically highest label that it sees. This repeats until the image no longer changes.

			read_flag();
		}

	}

	void absorb_small_blobs() {		
		d_Mat cluster_sizes(cv::Size(num_superpixels, 1), CV_32SC1);
		d_Mat cluster_strengths(cv::Size(num_superpixels, 1), CV_32SC1);
		d_Mat working_labels = labels;
		
		Launch::kernel_2d(labels.cols, labels.rows);
		SLIC_find_sizes<<<LAUNCH>>>(labels, cluster_sizes);
		SYNC_KERNEL("SLIC_find_sizes");
		//first we simply record the size of each blob in pixels

		Launch::kernel_1d(cluster_sizes.cols);
		SLIC_find_weak_labels<<<LAUNCH>>>(cluster_sizes, cluster_strengths, size_threshold);
		SYNC_KERNEL("SLIC_find_weak_labels");
		//then, we compare each blob size to a prescribed threshold, and assign a binary flag indicating whether that blob surpasses or falls short of the threshold.


		while (flag != 0) {
			reset_flag();

			Launch::kernel_2d(labels.cols, labels.rows);
			SLIC_absorb_small_blobs<<<LAUNCH>>>(labels, cluster_strengths, cluster_sizes, working_labels, d_flag);
			SYNC_KERNEL("SLIC_absorb_small_blobs");
			//we are using a celluar automaton in almost the exact same way as we did in separate_blobs, except this time we are looking at the sizes of the blobs,
			//and whether they fall below the prescribed size threshold.

			read_flag();
		}

	}

	void produce_ordered_labels() {

		std::cout << "raising flags" << std::endl;
		d_Mat flags(cv::Size(num_pixels, 1), CV_32SC1, cv::Scalar{ 0 });

		Launch::kernel_2d(labels.cols, labels.rows);
		SLIC_raise_flags<<<LAUNCH>>>(labels, flags);
		SYNC_KERNEL("SLIC_raise_flags");
		//first, note that all our possible labels lie within 0 - num_pixels. To count them, we will first have each pixel raise a binary flag at the index of a 0 - num_pixels list
		//correlating to the value of its label. Many threads will write to the same flag, but this is ok because we're only raising them.

		std::cout << "running exclusive scan" << std::endl;
		int true_K;
		d_Mat exclusive_scan_flags(cv::Size(num_pixels, 1), CV_32SC1, cv::Scalar{0});
		exclusive_scan(true_K, flags, exclusive_scan_flags);
		num_superpixels = true_K;
		//since the number of ones in the array correlates to the position and number of labels, running it through an exclusive scan will yield both the true number of labels, 
		//and an array which contains the information concerning their positions within the array of 0 - num_pixels.

		std::cout << "initializing map" << std::endl;
		d_Mat condensed_map(cv::Size(true_K, 1), CV_32SC1);

		Launch::kernel_1d(flags.cols);
		SLIC_init_map<<<LAUNCH>>>(flags, exclusive_scan_flags, condensed_map);
		SYNC_KERNEL("SLIC_init_map");
		//this is where it gets a bit tricky. since every element of the exclusive scan array contains the sum of all elements before it, and since our original array of flags 
		//contained only ones, we can effectively "pop" all of the unused labels. this leaves us with a 0 - true_K map, with the new labels as indices and the old labels as values.

		std::cout << "inverting map" << std::endl;
		d_Mat useful_map = flags;

		Launch::kernel_1d(useful_map.cols);
		SLIC_invert_map<<<LAUNCH>>>(condensed_map, useful_map);
		SYNC_KERNEL("SLIC_invert_map");
		//we take the condensed map, and we invert it so that the indices now correlate to the old labels, and the values correlate to their new 0 - true_K labels. We will use this
		//map to assign each old label (which fell sparsely between 0 - num_pixels) to their new labels, which fit perfectly between 0 - true_K.

		std::cout << "assigning new labels" << std::endl;
		Launch::kernel_2d(labels.cols, labels.rows);
		SLIC_assign_new_labels<<<LAUNCH>>>(labels, useful_map);
		SYNC_KERNEL("SLIC_assign_new_labels");
		//now we simply apply these new labels according to the old labels and the map that we produced.
	}

	void enforce_connectivity() {
		//after the main body of the algorithm has converged, we are left with an image of the superpixels written in labels 0-K. Or at least, we would be if we were perfectly
		//confident that all superpixels intialized at the beginning of the algorithm survived, contiuous and unatrophied. In reality, we cannot assume that, so we must assure it.

		std::cout << "separating blobs" << std::endl;
		separate_blobs();
		//the first step is to make sure that all pixels which share a label fall in the same spatially connected region. This is accomplished by relabeling all superpixels to a 
		//sparse set of values between 0 - num_pixels, such that each actually contiguous region has a unique label. We will call these uniquely-labeled continuous regions 'blobs'.

		std::cout << "absorbing small blobs" << std::endl;
		absorb_small_blobs();
		//next, we must account for any newly created blobs which are too small to be useful as superpixels. We will do this by simply allowing sufficiently large blobs to absorb their
		//smaller neighbors. As this part is currently structured, if a small blob has several large neighbors they compete for all or nothing. It may be better to divide the territory evenly.

		std::cout << "producing ordered labels" << std::endl;
		produce_ordered_labels();
		//now that the actual blobs are in their final form, we must count and relabel them. This process is tricky, so it is detailed further in produce_ordered_labels.

	}

}




void SLIC() { using namespace SLICAP;
	//SLIC is an algorithm which oversegments an image into regularly spaced clusters of pixels with similar characteristics, known as superpixels. It is quite similar to the popular K-means algorithm, 
	//the main difference being that it explicitly takes into account spatial proximity, and seeds regularly spaced centers along a virtual grid, so as to ultimately produce a field of superpixels. 

	std::cout << "sampling centers" << std::endl;
	SLIC::sample_centers(); 
	//first, we sample a number of points from the source image to serve as centers, at regular intervals S.

	do {
		std::cout << "assigning pixels to centers" << std::endl;
		SLIC::assign_pixels_to_centers();
		//for each pixel, given the 9 centers closest to it in space, we determine which of these centers is the nearest to it according to the distance formula given by the paper.
		//this formula consists of the euclidean distance between the color vectors of the pixel and the center, plus the distance between their coordinates tempered by the 'density' parameter. 

		std::cout << "updating centers" << std::endl;
		SLIC::update_centers();
		//for each center, given all pixels which are labeled with its ID, we calculate the geometric mean of these pixels and shift the center to this new poisition.
		//we then record the average distance that the centers were displaced, for the purpose of checking for convergence.


	} while (SLIC::displacement > SLIC::displacement_threshold);
	//once the average displacement of the centers falls below the threshold, the algorithm has converged.

	std::cout << "enforcing connectivity" << std::endl;
	SLIC::enforce_connectivity(); 
	//after producing the superpixels, we need to ensure that regions with the same label are connected, and that the values of these labels run sequentially from 0 - K with no missed values.
	//the exact process is somewhat complex, and detailed further within the enforce_connectivity function.
}
	