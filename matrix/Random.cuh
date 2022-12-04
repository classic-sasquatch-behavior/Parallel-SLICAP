#pragma once
#include"external_includes.h"
#include"Matrix.cuh"



__global__ void seed_curand_xor( int size, int seed, curandState* states) {
	GET_DIMS(id, zero);
	CHECK_BOUNDS(size, 1);

	curand_init(seed, id, 0, &states[id]);
}


struct Random {

    curandState* engine;
    int size = -1;
    int seed = 0;
    bool initialized = false;

    Random(){}

    Random(int _size, int _seed = 0){
        size = _size;
        seed = _seed;
        allocate();
        seed_engine();

        initialized = true;
    }

    ~Random(){
        if(initialized){
            cudaFree(engine);
        }
    }

    Random(const Random& input){
        size = input.size;
        seed = input.seed;
        allocate();
        copy(input.engine);

        initialized = true;
    }

    void operator=(const Random& input) {
        if(initialized){
            cudaFree(engine);
        }

        size = input.size;
        seed = input.seed;
        allocate();
        copy(input.engine);

        initialized = true;
    }

    void allocate(){
        cudaMalloc((void**)&engine, size * sizeof(curandState));
    }

    void seed_engine(){
        Launch::kernel_1d(size);
        seed_curand_xor<<<LAUNCH>>>(size, seed, engine);
        SYNC_KERNEL("seed_curand_xor");
    }

    void copy(curandState* input){
        cudaMemcpy(engine, input, size * sizeof(curandState), cudaMemcpyDeviceToDevice);
    }

    __device__ int randint (int id, int min, int max){
        curandState localstate = engine[id];
        return (curand(&localstate) % (max - min + 1)) + min;
    } 

};











