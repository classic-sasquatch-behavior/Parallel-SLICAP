#pragma once

#include"SLIC_kernels.cuh"
#include"exclusive_scan.cuh"
#include"SLICAP.h"


struct SLIC {
	
	SLICAP* parent;

	int source_rows, source_cols, num_pixels;
	int SP_rows, SP_cols, num_superpixels;
	int space_between_centers;
	int density_modifier;

	d_Mat center_rows, center_cols, center_grid;
	d_Mat row_sums, col_sums, num_instances;

	int displacement = 0;
	int* h_displacement = &displacement;
	int* d_displacement;

	SLIC(int _source_cols, int _source_rows, int _num_pixels, int _space_between_centers, 
    int _density_modifier, int _SP_cols, int _SP_rows, int _num_superpixels, SLICAP* _parent);

	void sample_centers();

	void assign_pixels_to_centers();

	void reset_displacement();
	
	void read_displacement();

	void update_centers();

	void separate_blobs();

	void absorb_small_blobs();

	void produce_ordered_labels();

	void enforce_connectivity();

};



