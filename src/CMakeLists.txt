add_library(cuda_kl_chol SHARED kl_chol.cu)
set_target_properties(cuda_kl_chol PROPERTIES CUDA_ARCHITECTURES 86)
target_link_libraries(cuda_kl_chol
  CUDA::cudart
  CUDA::cusparse
  CUDA::cusolver
  CUDA::cublas
  )