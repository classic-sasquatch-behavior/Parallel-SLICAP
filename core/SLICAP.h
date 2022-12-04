#pragma once
#include"SLIC.h"
#include"AP.h"

struct SLICAP {

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

	int closest_greater_power_of_two(int input) {
		return powf(2, ceil(log2(input)));
	}

	void display_result() {
		cv::imshow("source image", source);
		cv::imshow("segmented image", result);
		cv::waitKey(0);
	}

	void reset_flag() {
		flag = 0;
		CUDA_SAFE_CALL(cudaMalloc((void**)&d_flag, sizeof(uint)));
		CUDA_SAFE_CALL(cudaMemcpy(d_flag, h_flag, sizeof(uint), cudaMemcpyHostToDevice));
	}

	void read_flag() {
		CUDA_SAFE_CALL(cudaMemcpy(d_flag, h_flag, sizeof(uint), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaFree(d_flag));
	}

	void initialize(string source) {
		load_image(source);
		SLIC::initialize();
		AP::initialize();
	}

	

}

