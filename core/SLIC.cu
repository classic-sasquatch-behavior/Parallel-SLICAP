#include"SLIC.h"

SLIC::SLIC(h_Mat SLIC_source, SLICAP* _parent): 
        parent(_parent),
		source_rows(SLIC_source.rows), 
		source_cols(SLIC_source.cols), 
		num_pixels(source_rows * source_cols),

		space_between_centers(sqrt(num_pixels) / superpixel_size_factor), 
		density_modifier(density / space_between_centers),

		SP_rows(floor(source_rows/space_between_centers)), 
		SP_cols(floor(source_cols/space_between_centers)), 
		num_superpixels(SP_rows * SP_cols),

		center_rows(cv::Size(SP_cols, SP_rows), CV_32SC1), 
		center_cols(cv::Size(SP_cols, SP_rows), CV_32SC1), 
		center_grid(cv::Size(SP_cols, SP_rows), CV_32SC1),

		row_sums(cv::Size(num_pixels, 1), CV_32SC1), 
		col_sums(cv::Size(num_pixels, 1), CV_32SC1), 
		num_instances(cv::Size(num_pixels, 1), CV_32SC1)
	{}

void SLIC::sample_centers(){
		Launch::kernel_2d(center_grid.cols, center_grid.rows);
		SLIC_initialize_centers<<<LAUNCH>>>(parent->L_src(), center_rows, center_cols);
		SYNC_KERNEL("SLIC_initialize_centers");
}

void SLIC::assign_pixels_to_centers(){
    	Launch::kernel_2d(parent->d_labels.cols, parent->d_labels.rows);
		SLIC_assign_pixels_to_centers<<<LAUNCH>>>(parent->L_src(), parent->A_src(), parent->B_src(), density_modifier, center_rows, center_cols, parent->d_labels);
		SYNC_KERNEL("SLIC_assign_pixels_to_centers");
}

void SLIC::reset_displacement(){
    displacement = 0;
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_displacement, sizeof(int)));
    CUDA_SAFE_CALL(cudaMemcpy(d_displacement, h_displacement, sizeof(int), cudaMemcpyHostToDevice));
}

void SLIC::read_displacement(){
    CUDA_SAFE_CALL(cudaMemcpy(d_displacement, h_displacement, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaFree(d_displacement));
}

void SLIC::update_centers(){
    Launch::kernel_2d(parent->d_labels.cols, parent->d_labels.rows);
    SLIC_condense_labels<<<LAUNCH>>>(parent->d_labels, row_sums, col_sums, num_instances);
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

void SLIC::separate_blobs(){
    d_Mat working_labels = parent->d_labels;

    while (parent->flag != 0) {
        parent->reset_flag();

        Launch::kernel_2d(parent->d_labels.cols, parent->d_labels.rows);
        SLIC_separate_blobs<<<LAUNCH>>>(parent->d_labels, working_labels, parent->d_flag);
        SYNC_KERNEL("SLIC_separate_blobs");

        //here we are using a cellular automaton. first we assign each pixel its linear ID from 0 - num_pixels. then each pixel looks at the neighbors with which it
        //originally shared a label, and adopts the numerically highest label that it sees. This repeats until the image no longer changes.

        parent->read_flag();
    }
    parent->reset_flag();

}

void SLIC::absorb_small_blobs(){		
    d_Mat cluster_sizes(cv::Size(num_superpixels, 1), CV_32SC1);
    d_Mat cluster_strengths(cv::Size(num_superpixels, 1), CV_32SC1);
    d_Mat working_labels = parent->d_labels;
    
    Launch::kernel_2d(parent->d_labels.cols, parent->d_labels.rows);
    SLIC_find_sizes<<<LAUNCH>>>(parent->d_labels, cluster_sizes);
    SYNC_KERNEL("SLIC_find_sizes");
    //first we simply record the size of each blob in pixels

    Launch::kernel_1d(cluster_sizes.cols);
    SLIC_find_weak_labels<<<LAUNCH>>>(cluster_sizes, cluster_strengths, size_threshold);
    SYNC_KERNEL("SLIC_find_weak_labels");
    //then, we compare each blob size to a prescribed threshold, and assign a binary flag indicating whether that blob surpasses or falls short of the threshold.


    while (parent->flag != 0) {
        parent->reset_flag();

        Launch::kernel_2d(parent->d_labels.cols, parent->d_labels.rows);
        SLIC_absorb_small_blobs<<<LAUNCH>>>(parent->d_labels, cluster_strengths, cluster_sizes, working_labels, parent->d_flag);
        SYNC_KERNEL("SLIC_absorb_small_blobs");
        //we are using a celluar automaton in almost the exact same way as we did in separate_blobs, except this time we are looking at the sizes of the blobs,
        //and whether they fall below the prescribed size threshold.

        parent->read_flag();
    }
    parent->reset_flag();

}

void SLIC::produce_ordered_labels(){

    std::cout << "raising flags" << std::endl;
    d_Mat flags(cv::Size(num_pixels, 1), CV_32SC1, cv::Scalar{ 0 });

    Launch::kernel_2d(parent->d_labels.cols, parent->d_labels.rows);
    SLIC_raise_flags<<<LAUNCH>>>(parent->d_labels, flags);
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
    Launch::kernel_2d(parent->d_labels.cols, parent->d_labels.rows);
    SLIC_assign_new_labels<<<LAUNCH>>>(parent->d_labels, useful_map);
    SYNC_KERNEL("SLIC_assign_new_labels");
    //now we simply apply these new labels according to the old labels and the map that we produced.
}

void SLIC::enforce_connectivity(){
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