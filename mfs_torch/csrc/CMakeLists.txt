add_library(mfs_torch SHARED
    mfs_torch_ops.cu
    mfs_torch_ops.cpp
)
target_compile_features(mfs_torch PRIVATE cxx_std_11)
target_link_libraries(mfs_torch PRIVATE ${TORCH_LIBRARIES} Python::Python Eigen3::Eigen 
cuda_kl_chol CUDA::cudart CUDA::cusparse CUDA::cusolver CUDA::cublas)