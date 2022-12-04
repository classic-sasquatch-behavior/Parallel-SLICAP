#pragma once
#include"external_includes.h"

struct SLIC;
struct AP;

struct SLICAP {

	SLIC* SLIC_eng;
	AP* AP_eng;

	h_Mat h_source;
	h_Mat h_labels;

	h_Mat SLIC_result;
	h_Mat AP_result;

	d_Mat d_source;
	d_Mat d_labels;

	vector<d_Mat> CIELAB_planes;

	inline d_Mat L_src() {return CIELAB_planes[0];}
	inline d_Mat A_src() {return CIELAB_planes[1];}
	inline d_Mat B_src() {return CIELAB_planes[2];}

	uint flag;
	uint* h_flag = &flag;
	uint* d_flag;

	int num_superpixels;

	SLICAP(std::string source_path);
	~SLICAP(){
		delete SLIC_eng;
		delete AP_eng;
	}

	void run_SLIC();
	void run_AP();

	void display_SLIC_result();
	void display_AP_result();

	vector<d_Mat> split_into_channels(h_Mat input);

	inline int closest_greater_power_of_two(int input) {return powf(2, ceil(log2(input)));}

	void reset_flag();

	void read_flag();

};

