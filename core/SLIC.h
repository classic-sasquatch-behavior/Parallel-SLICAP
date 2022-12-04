#pragma once

#include"SLIC_kernels.cuh"
#include"exclusive_scan.cuh"
#include"SLICAP.h"

struct SLIC {

	const int displacement_threshold = 1;
	const float density = 0.5;
	const int superpixel_size_factor = 10;
	const int size_threshold = (superpixel_size_factor * superpixel_size_factor) / 2;
	
	SLICAP* parent;

	int source_rows, source_cols, num_pixels;
	int SP_rows, SP_cols, num_superpixels;
	int space_between_centers;
	int density_modifier;

	d_Mat center_rows, center_cols, center_grid;
	d_Mat row_sums, col_sums, num_instances;

	int displacement;
	int* h_displacement = &displacement;
	int* d_displacement;

	SLIC(h_Mat SLIC_source, SLICAP* _parent);

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



