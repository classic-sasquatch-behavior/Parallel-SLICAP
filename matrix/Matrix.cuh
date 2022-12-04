#pragma once

#include"external_includes.h"
#include"launch.h"


enum Direction{
    host = 0,
    device = 1,
};

struct Position{
    int x;
    int y;
    Position(int _x, int _y):x(_x), y(_y){}
};

template<typename Type>
struct Device_Ptr{
    int dims[4] = {1,1,1,1};
    int num_dims = 0;

    __device__ inline int x() const {return dims[0];}
    __device__ inline int y() const {return dims[1];}
    __device__ inline int z() const {return dims[2];}
    __device__ inline int w() const {return dims[3];}

    Type* data;

    __device__ int size(){return x() * y() * z() * w();}

    Device_Ptr(std::vector<int> _dims, int _num_dims, Type* _data){
        num_dims = _num_dims;
        for(int i = 0; i < _num_dims; i++){
            dims[i] = _dims[i];
        }
        data = _data;
    }

    __device__ Type& operator[](int input){
        return data[input];
    }

    __device__ Type& operator()(int x, int y = 0, int z = 0, int w = 0){
        return data[(((x * dims[1] + y) * dims[2] + z) * dims[3] + w)];
    }
};

template<typename Type>
__global__ static void get_submatrix_kernel(Device_Ptr<Type> input, Device_Ptr<Type> output, int start_x, int start_y){
    GET_DIMS(x,y);
    CHECK_BOUNDS(output.dims[0], output.dims[1]);

    output(x,y) = input(x + start_x, y + start_y);
}

template<typename Type>
struct Matrix {

    private:
    Type* host_data = nullptr;
    Type* device_data = nullptr;

    public:
    int dims[4] = {1,1,1,1};
    int num_dims = 0;
    int size;
    int bytesize;

    inline int x() const {return dims[0];}
    inline int y() const {return dims[1];}
    inline int z() const {return dims[2];}
    inline int w() const {return dims[3];}

    Direction recent = host;
    bool synced = true;

    Type* data(Direction direction){
        access(direction);
        switch(direction){
            case host: return host_data;
            case device: return device_data;
        }
    }

    Matrix(){}

    Matrix(const std::vector<int> _dims, Type _constant = 0){
        num_dims = _dims.size();
        for(int i = 0; i< num_dims; i++){
            dims[i] = _dims[i];
        }
        size = dims[0] * dims[1] * dims[2] * dims[3];
        bytesize = size * sizeof(Type);
        allocate();
        fill(_constant);
    }

    ~Matrix(){
        delete host_data;
        cudaFree(device_data);
    }

    Matrix(const Matrix<Type>& input){
        num_dims = input.num_dims;
        for(int i  = 0; i < num_dims; i++){
            dims[i] = input.dims[i];
        }
        size = input.size;
        bytesize = input.bytesize;
        allocate();

        //detect the input matrix's sync state
        //load according to that sync state
        Direction input_state = input.recent;
        switch(input_state){
            case host: host_load(input.host_data);
            case device: device_load(input.device_data);
        }
        desync(input_state);
    }

    void operator= (const Matrix<Type>& input){
        delete host_data;
        cudaFree(device_data);

        num_dims = input.num_dims;
        for(int i  = 0; i < num_dims; i++){
            dims[i] = input.dims[i];
        }
        size = input.size;
        bytesize = input.bytesize;

        allocate();

        Direction input_state = input.recent;
        switch(input_state){
            case host: host_load(input.host_data);
            case device: device_load(input.device_data);
        }
        desync(input_state);
    }

    operator Device_Ptr<Type>(){
        return Device_Ptr<Type>({dims[0], dims[1], dims[2], dims[3]}, num_dims, data(device));
    }

    Matrix<Type> get_submatrix(std::vector<int> size, std::vector<int> start){

        //assumes 2d only, for now
        Matrix<Type> output(size, 0);

        Launch::kernel_2d(size[0], size[1]);
        get_submatrix_kernel<Type><<<LAUNCH>>>(*this, output, start[0], start[1]);
        SYNC_KERNEL("get_submatrix_kernel");

        return output;
    }

    inline Type& operator()(int x, int y = 0, int z = 0, int w = 0){
        access(host);
        return host_data[((x * dims[1] + y) * dims[2] + z) * dims[3] + w];
    }

    void allocate(){
        host_data = new Type[size];
        cudaMalloc((void**)&device_data, bytesize);
    }

    void sync(){
        switch(recent){
            case host: upload(); break;
            case device: download(); break;
        } synced = true;
    }

    void desync(Direction changing){
        synced = false;
        recent = changing;
    }

    void access(Direction changing){
        switch(synced){
            case true:  desync(changing); break;
            case false: if(recent != changing){sync(); desync(changing);}
        }
    }

    void load(Type* input, int input_size, Direction direction){
        access(direction);
        switch(direction){
            case host: cudaMemcpy(host_data, input, input_size, cudaMemcpyHostToHost); break;
            case device: cudaMemcpy(device_data, input, input_size, cudaMemcpyDeviceToDevice); break;
        }
    }

    void unload(Type* output, int output_size, Direction direction){
        access(direction);
        switch(direction){
            case host: cudaMemcpy(output, host_data, output_size, cudaMemcpyHostToHost); break;
            case device: cudaMemcpy(output, device_data, output_size, cudaMemcpyDeviceToDevice); break;
        }

    }

    void host_load(Type* input){
        for(int i = 0; i < size; i++){
            host_data[i] = input[i];
        }
    }

    void device_load(Type* input){
        cudaMemcpy(device_data, input, bytesize, cudaMemcpyDeviceToDevice);
    }

    void fill(int constant){
        cudaMemset(device_data, constant, bytesize);
        download();
    }

    void fill_device_memory(int constant){
        cudaMemset(device_data, constant, bytesize);
    }

    void upload(){
        cudaMemcpy(device_data, host_data, bytesize, cudaMemcpyHostToDevice);
    }

    void download(){
        cudaMemcpy(host_data, device_data, bytesize, cudaMemcpyDeviceToHost);
    }


    Matrix(cv::Mat input){
        num_dims = input.dims;
        bool has_channels = (input.channels() > 1);
        num_dims += has_channels;

        dims[0] = input.cols;
        dims[1] = input.rows;
        dims[2] = input.channels();

        size = dims[0] * dims[1] * dims[2] * dims[3];
        bytesize = size * sizeof(Type);
        allocate();

        //!PROBABLY BROKEN
        load((Type*)input.data, bytesize, host);

    }

    void operator=(cv::Mat input) {
        num_dims = input.dims;
        bool has_channels = (input.channels() > 1);
        num_dims += has_channels;

        dims[0] = input.cols;
        dims[1] = input.rows;
        dims[2] = input.channels();

        size = dims[0] * dims[1] * dims[2] * dims[3];
        bytesize = size * sizeof(Type);
        allocate();

        //!PROBABLY BROKEN
        load((Type*)input.data, bytesize, host);
    }


    operator cv::Mat() {
        access(host);
        return cv::Mat(dims[0], dims[1], cv::DataType<Type>::type, host_data);
    }


    void operator=(cv::cuda::GpuMat input){
        cv::Mat temp;
        input.download(temp);
        *this = temp;
    }


    operator cv::cuda::GpuMat(){
        cv::Mat temp = *this;
        cv::cuda::GpuMat gpu_temp;
        gpu_temp.upload(temp);
        return gpu_temp;
    }











};



