const char * getErrorString(cusparseStatus_t error)
{
    switch (error)
    {
        case CUSPARSE_STATUS_SUCCESS:
            return "The operation completed successfully.";
        case CUSPARSE_STATUS_NOT_INITIALIZED:
            return "The cuSPARSE library was not initialized. This is usually caused by the lack of a prior call, an error in the CUDA Runtime API called by the cuSPARSE routine, or an error in the hardware setup.\n" \
				"To correct: call cusparseCreate() prior to the function call; and check that the hardware, an appropriate version of the driver, and the cuSPARSE library are correctly installed.";

        case CUSPARSE_STATUS_ALLOC_FAILED:
            return "Resource allocation failed inside the cuSPARSE library. This is usually caused by a cudaMalloc() failure.\n"\
					"To correct: prior to the function call, deallocate previously allocated memory as much as possible.";

        case CUSPARSE_STATUS_INVALID_VALUE:
            return "An unsupported value or parameter was passed to the function (a negative vector size, for example).\n"\
				"To correct: ensure that all the parameters being passed have valid values.";

        case CUSPARSE_STATUS_ARCH_MISMATCH:
            return "The function requires a feature absent from the device architecture; usually caused by the lack of support for atomic operations or double precision.\n"\
				"To correct: compile and run the application on a device with appropriate compute capability, which is 1.1 for 32-bit atomic operations and 1.3 for double precision.";

        case CUSPARSE_STATUS_MAPPING_ERROR:
            return "An access to GPU memory space failed, which is usually caused by a failure to bind a texture.\n"\
				"To correct: prior to the function call, unbind any previously bound textures.";

        case CUSPARSE_STATUS_EXECUTION_FAILED:
            return "The GPU program failed to execute. This is often caused by a launch failure of the kernel on the GPU, which can be caused by multiple reasons.\n"\
					"To correct: check that the hardware, an appropriate version of the driver, and the cuSPARSE library are correctly installed.";

        case CUSPARSE_STATUS_INTERNAL_ERROR:
            return "An internal cuSPARSE operation failed. This error is usually caused by a cudaMemcpyAsync() failure.\n"\
					"To correct: check that the hardware, an appropriate version of the driver, and the cuSPARSE library are correctly installed. Also, check that the memory passed as a parameter to the routine is not being deallocated prior to the routine’s completion.";

        case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
            return "The matrix type is not supported by this function. This is usually caused by passing an invalid matrix descriptor to the function.\n"\
					"To correct: check that the fields in cusparseMatDescr_t descrA were set correctly.";
    }

    return "<unknown>";
}

void CHECK_CUSPARSE(cusparseStatus_t status)
{
    if (status != CUSPARSE_STATUS_SUCCESS) {
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n", __LINE__, getErrorString(status), status);
        exit(EXIT_FAILURE);
    }
}

void CHECK_CUDA(cudaError_t status)
{
    if (status != cudaSuccess) {
        printf("CUDA API failed at line %d with error: %s (%d)\n", __LINE__, cudaGetErrorString(status), status);
        exit(EXIT_FAILURE);
    }
}

void initializeArrayToZero(float *array, int n)
{
    int i;
    for(i=0;i<n;i++)
        array[i] = 0;
}

void initializeArrayToZeroDouble(double *array, int n)
{
    int i;
    for(i=0;i<n;i++)
        array[i] = 0;
}

void initializeArrayToZerocuComplex(cuComplex *array, int n)
{
    int i;
    for(i=0;i<n;i++)
        array[i] = make_cuComplex(0, 0);
}

void initializeArrayToZerocuDoubleComplex(cuDoubleComplex *array, int n)
{
    int i;
    for(i=0;i<n;i++)
        array[i] = make_cuDoubleComplex(0, 0);
}

void initializeArrayRandomSparse(float *array, int n, int nnz)
{
    int i;
    float random_number;
    initializeArrayToZero(array, n);
    for(i=0;i<nnz;)
    {
        int index = (int) (n * ((float) rand() / (RAND_MAX)));
        if (array[index]) {
            continue;
        }
        random_number = (float) rand() / ( (float) RAND_MAX / 100 ) + 1;
        array[index] = random_number;
        ++i;
    }
}

void initializeArrayRandomSparseDouble(double *array, int n, int nnz)
{
    int i;
    double random_number;
    initializeArrayToZeroDouble(array, n);
    for(i=0;i<nnz;)
    {
        int index = (int) (n * ((double) rand() / (RAND_MAX)));
        if (array[index]) {
            continue;
        }
        random_number = (double) rand() / ( (double) RAND_MAX / 100 ) + 1;
        array[index] = random_number;
        ++i;
    }
}

void initializeArrayRandomSparsecuComplex(cuComplex *array, int n, int nnz)
{
    int i;
    float random_number;
    initializeArrayToZerocuComplex(array, n);
    for(i=0;i<nnz;)
    {
        int index = (int) (n * ((float) rand() / (RAND_MAX)));
        if (array[index].x) {
            continue;
        }
        random_number = (float) rand() / ( (float) RAND_MAX / 100 ) + 1;
        array[index] = make_cuComplex(random_number, 1);
        ++i;
    }
}

void initializeArrayRandomSparsecuDoubleComplex(cuDoubleComplex *array, int n, int nnz)
{
    int i;
    double random_number;
    initializeArrayToZerocuDoubleComplex(array, n);
    for(i=0;i<nnz;)
    {
        int index = (int) (n * ((double) rand() / (RAND_MAX)));
        if (array[index].x) {
            continue;
        }
        random_number = (double) rand() / ( (double) RAND_MAX / 100 ) + 1;
        array[index] = make_cuDoubleComplex(random_number, 1);
        ++i;
    }
}

void stampaArray(int* array, int n)
{
    int i;
    for(i=0;i<n;i++)
        printf("%d ", array[i]);
    printf("\n");
}

void stampaArrayF(float* array, int n)
{
    int i;
    for(i=0;i<n;i++)
        printf("%f ", array[i]);
    printf("\n");
}

void stampaArrayD(double* array, int n)
{
    int i;
    for(i=0;i<n;i++)
        printf("%f ", array[i]);
    printf("\n");
}

void stampaArrayC(cuComplex* array, int n)
{
    int i;
    for(i=0;i<n;i++)
        printf("[%f %f] ", array[i].x, array[i].y);
    printf("\n");
}

void stampaArrayZ(cuDoubleComplex* array, int n)
{
    int i;
    for(i=0;i<n;i++)
        printf("[%f %f] ", array[i].x, array[i].y);
    printf("\n");
}