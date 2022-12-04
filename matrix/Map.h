#pragma once



#include<unordered_map>
#include"Matrix.cuh"

template<typename Type>
struct Map{

    Matrix<Type> background;
    Matrix<Type> objects;
    int dims[2] = {0,0};

    Map(std::vector<int> _dims){
        dims[0] = _dims[0];
        dims[1] = _dims[1];

        Matrix<Type> background_({dims[0], dims[1]}, 0);
        Matrix<Type> objects_ ({dims[0], dims[1]}, 0);
        background = background_;
        objects = objects_;
    }
};





