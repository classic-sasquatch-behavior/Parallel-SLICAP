#pragma once
#include"matrix/Matrix.cuh"


template<typename Type>
__global__ static void block_upsweep(Device_Ptr<Type> input, Device_Ptr<Type> sum){
    int id = threadIdx.x;

    __shared__ extern Type temp[];

    const uint input_size = input.x();

    const uint unit = 31 - __clz(input_size) - 10;

    const uint m = id * powf(2,unit) + powf(2,unit) - 1;
    const uint n = id * powf(2,unit) + powf(2,unit -1) -1;

    temp[id] = input[m] + input[n]; 

    __syncthreads();

    for(int step = 0; step < 10; step++){

        if(id < powf(2,9-step) ){

            uint step_m = id * powf(2, step + 1) + powf(2, step+1) - 1;
            uint step_n = id * powf(2, step + 1) + powf(2, step) - 1;

            temp[step_m] = temp[step_m] + temp[step_n];
        }


        __syncthreads();
    }

    if(id == 1023){sum[0] = temp[id]; temp[id] = 0;}

    input[m] = temp[id];
}

template<typename Type>
__global__ static void grid_upsweep(Device_Ptr<Type> buffer_1, Device_Ptr<Type> buffer_2, Device_Ptr<Type> sum, int step, int iterations, int num_threads){
    GET_DIMS(id, zero);
    CHECK_BOUNDS(num_threads,1);

    int n = (id * powf(2,step + 1)) + powf(2,step + 1) - 1;
    int m = (id * powf(2,step + 1)) + powf(2, step) - 1;

    buffer_2[n] = buffer_1[m] + buffer_1[n];
}

template<typename Type>
__global__ static void block_downsweep(Device_Ptr<Type> input){
    int id = threadIdx.x;

    __shared__ extern Type temp[];

    const uint unit = 31 - __clz(input.size()) - 10;

    int m = id * powf(2,unit) + powf(2,unit) - 1;

    temp[id] = input[m];

    __syncthreads();

    for(int step = 9; step >= 0; step--){

        if(id < powf(2,9 - step)){

            int k = (id * powf(2,step + 1));

            int step_m = k + powf(2,step + 1) - 1;
            int step_n = k + powf(2, step) - 1;

            Type t = temp[step_m];
            temp[step_m] += temp[step_n];
            temp[step_n] = t;
        }

        __syncthreads();
    }

    input[m] = temp[id];

}

template<typename Type>
__global__ static void grid_downsweep(Device_Ptr<Type> buffer_1, Device_Ptr<Type> buffer_2, int step, int iterations, int num_threads){
    GET_DIMS(id, zero);
    CHECK_BOUNDS(num_threads, 1);

    int inverse_step = iterations - step;

    int k = id * powf(2, step);

    int m = k + powf(2, step) - 1;
    int n = k + powf(2,step -1) -1;

    buffer_2[m] = buffer_1[n] + buffer_1[m];
    buffer_2[n] = buffer_1[m];
}

template<typename Type>
static void exclusive_scan(Type& sum, d_Mat mat_input, d_Mat& mat_output){
    cv::Mat temp_input;
    cv::Mat temp_output;
    mat_input.download(temp_input);
    mat_output.download(temp_output);

    std::cout << "copying mat to matrix" << std::endl;
    Matrix<Type> input = temp_input;
    Matrix<Type> output = temp_output;


    int input_size = input.x();
    int closest_power = 1 << (int)ceil(log2(input_size));
    int iterations = log2(closest_power);

    Matrix<Type> buffer_1({closest_power}, 0);
    Matrix<Type> buffer_2({closest_power}, 0);
    Matrix<Type> final_sum({1},0);

    std::cout << "loading buffers" << std::endl;
    buffer_1.load(input.data(device), input_size * sizeof(Type), Direction::device );
    buffer_2 = buffer_1;


    std::cout << "performing grid upsweep" << std::endl;
    //upsweep
    for(int step = 0; step < iterations - 11; step++){

        int num_threads = powf(2, iterations - step - 1);
        std::string step_string = std::to_string(step);

        Launch::kernel_1d(num_threads);
        grid_upsweep<Type><<<LAUNCH>>>(buffer_1, buffer_2, final_sum, step, iterations, num_threads);
        SYNC_KERNEL("grid_upsweep " + step_string);

        buffer_1 = buffer_2;
    }

    std::cout << "performing block upsweep" << std::endl;
    Launch::kernel_1d(1024);
    Launch::allocate_shmem(1024*sizeof(Type));
    block_upsweep<Type><<<LAUNCH_SHMEM>>>(buffer_1, final_sum);
    SYNC_KERNEL("block_upsweep");

    final_sum.unload(&sum, 1 * sizeof(Type), Direction::host);

    std::cout << "performing block downsweep" <<std::endl;
    block_downsweep<Type><<<LAUNCH_SHMEM>>>(buffer_1);
    SYNC_KERNEL("block_downsweep");

    buffer_2 = buffer_1;

    std::cout << "performing grid downsweep" <<std::endl;
    //downsweep
    for(int step = iterations - 10; step > 0; step--){
        int num_threads = powf(2, iterations - step);
        std::string step_string = std::to_string(step);

        Launch::kernel_1d(num_threads);
        grid_downsweep<Type><<<LAUNCH>>>(buffer_1, buffer_2, step, iterations, num_threads);
        SYNC_KERNEL("grid downsweep " + step_string);

        buffer_1 = buffer_2;
    }

    std::cout << "loading output" << std::endl;
    output.load(buffer_1.data(device), input_size * sizeof(Type),Direction::device);

    std::cout << "returning output" << std::endl;
    temp_output = output;

    std::cout << "uploading output" << std::endl;
    mat_output.upload(temp_output);

    std::cout << "finished exclusive scan" << std::endl;

}