
cuda_add_executable(create_and_destroy.e management/cusparse_create_and_destroy.cu OPTIONS --cudart=shared -lcusparse)
#set_target_properties(create_and_destroy.e PROPERTIES CUDA_RUNTIME_LIBRARY shared EXTRA_NVCCFLAGS --cudart=shared)
install(TARGETS create_and_destroy.e RUNTIME DESTINATION ${GVIRTUS_HOME}/demo/cusparse)
message("create_and_destroy added")

cuda_add_executable(get_error_string.e management/cusparse_get_error_string.cu OPTIONS --cudart=shared -lcusparse)
#set_target_properties(get_error_string.e PROPERTIES CUDA_RUNTIME_LIBRARY shared EXTRA_NVCCFLAGS --cudart=shared)
install(TARGETS get_error_string.e RUNTIME DESTINATION ${GVIRTUS_HOME}/demo/cusparse)
message("get_error_string added")
