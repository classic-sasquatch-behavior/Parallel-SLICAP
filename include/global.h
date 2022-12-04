#pragma once

typedef unsigned int uint;





namespace Param{

    //SLIC
	const int displacement_threshold = 1;
	const float density = 0.5;
	const int superpixel_size_factor = 10;
	const int size_threshold = (superpixel_size_factor * superpixel_size_factor) / 2;

    //AP
    const float damping_factor = 0.5f;
	const uint difference_threshold = 10;
	const uint num_constant_cycles_for_convergence = 3;


}