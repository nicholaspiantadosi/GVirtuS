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

void initializeMatrixZeroF(float *matrix, int M, int N)
{
    int i, j, k=0;
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < M; j++)
        {
            matrix[k++]=0;
        }
    }
}

void initializeMatrixZeroD(double *matrix, int M, int N)
{
    int i, j, k=0;
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < M; j++)
        {
            matrix[k++]=0;
        }
    }
}

void initializeMatrixZeroC(cuComplex *matrix, int M, int N)
{
    int i, j, k=0;
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < M; j++)
        {
            matrix[k++]=make_cuComplex(0, 0);
        }
    }
}

void initializeMatrixZeroZ(cuDoubleComplex *matrix, int M, int N)
{
    int i, j, k=0;
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < M; j++)
        {
            matrix[k++]=make_cuDoubleComplex(0, 0);
        }
    }
}

void initializeMatrixRandomSparseF(float *matrix, int M, int N, int nnz)
{
    initializeMatrixZeroF(matrix, M, N);
    int i=0;
    float random_number;
    for (i = 0; i < nnz;) {
        int index = (int) (M * N * ((float) rand() / (RAND_MAX + 1.0)));
        if (matrix[index]) {
            continue;
        }
        random_number = (float) rand() / ( (float) RAND_MAX / 100 ) + 1;
        matrix[index] = random_number;
        ++i;
    }
}

void initializeMatrixRandomSparseD(double *matrix, int M, int N, int nnz)
{
    initializeMatrixZeroD(matrix, M, N);
    int i=0;
    double random_number;
    for (i = 0; i < nnz;) {
        int index = (int) (M * N * ((double) rand() / (RAND_MAX + 1.0)));
        if (matrix[index]) {
            continue;
        }
        random_number = (double) rand() / ( (double) RAND_MAX / 100 ) + 1;
        matrix[index] = random_number;
        ++i;
    }
}

void initializeMatrixRandomSparseC(cuComplex *matrix, int M, int N, int nnz)
{
    initializeMatrixZeroC(matrix, M, N);
    int i=0;
    float random_number;
    for (i = 0; i < nnz;) {
        int index = (int) (M * N * ((float) rand() / (RAND_MAX + 1.0)));
        if (matrix[index].x) {
            continue;
        }
        random_number = (float) rand() / ( (float) RAND_MAX / 100 ) + 1;
        matrix[index] = make_cuComplex(random_number, 1);
        ++i;
    }
}

void initializeMatrixRandomSparseZ(cuDoubleComplex *matrix, int M, int N, int nnz)
{
    initializeMatrixZeroZ(matrix, M, N);
    int i=0;
    double random_number;
    for (i = 0; i < nnz;) {
        int index = (int) (M * N * ((double) rand() / (RAND_MAX + 1.0)));
        if (matrix[index].x) {
            continue;
        }
        random_number = (double) rand() / ( (double) RAND_MAX / 100 ) + 1;
        matrix[index] = make_cuDoubleComplex(random_number, 1);
        ++i;
    }
}

void initializeMatrixLowerTriangularSparseRandomF(float *matrix, int M, int nnz)
{
    initializeMatrixZeroF(matrix, M, M);
    int i=0;
    double random_number;
    for (i = 0; i < M*M; i++) {
        random_number = (float) rand() / ( (float) RAND_MAX / 100 ) + 1;
        //matrix[i] = random_number;
        matrix[i] = i+1;
    }
    int j=0;
    int size=0;
    for (i = 0; i < M; i++) {
        for (j = 0; j < M; j++) {
            if (i > j) {
                matrix[i*M+j] = 0;
            } else {
                ++size;
            }
        }
    }
    for (i = 0; i < size-nnz;) {
        int index = (int) (M * M * ((float) rand() / (RAND_MAX + 1.0)));
        if (matrix[index] == 0) {
            continue;
        }
        matrix[index] = 0;
        ++i;
    }
}

void initializeMatrixLowerTriangularSparseRandomD(double *matrix, int M, int nnz)
{
    initializeMatrixZeroD(matrix, M, M);
    int i=0;
    double random_number;
    for (i = 0; i < M*M; i++) {
        random_number = (double) rand() / ( (double) RAND_MAX / 100 ) + 1;
        //matrix[i] = random_number;
        matrix[i] = i+1;
    }
    int j=0;
    int size=0;
    for (i = 0; i < M; i++) {
        for (j = 0; j < M; j++) {
            if (i > j) {
                matrix[i*M+j] = 0;
            } else {
                ++size;
            }
        }
    }
    for (i = 0; i < size-nnz;) {
        int index = (int) (M * M * ((double) rand() / (RAND_MAX + 1.0)));
        if (matrix[index] == 0) {
            continue;
        }
        matrix[index] = 0;
        ++i;
    }
}

void initializeMatrixLowerTriangularSparseRandomC(cuComplex *matrix, int M, int nnz)
{
    initializeMatrixZeroC(matrix, M, M);
    int i=0;
    double random_number;
    for (i = 0; i < M*M; i++) {
        random_number = (float) rand() / ( (float) RAND_MAX / 100 ) + 1;
        //matrix[i] = make_cuComplex(random_number, 1);
        matrix[i] = make_cuComplex(i+1, 1);
    }
    int j=0;
    int size=0;
    for (i = 0; i < M; i++) {
        for (j = 0; j < M; j++) {
            if (i > j) {
                matrix[i*M+j] = make_cuComplex(0, 0);
            } else {
                ++size;
            }
        }
    }
    for (i = 0; i < size-nnz;) {
        int index = (int) (M * M * ((float) rand() / (RAND_MAX + 1.0)));
        if (matrix[index].x == 0) {
            continue;
        }
        matrix[index] = make_cuComplex(0, 0);
        ++i;
    }
}

void initializeMatrixLowerTriangularSparseRandomZ(cuDoubleComplex *matrix, int M, int nnz)
{
    initializeMatrixZeroZ(matrix, M, M);
    int i=0;
    double random_number;
    for (i = 0; i < M*M; i++) {
        random_number = (double) rand() / ( (double) RAND_MAX / 100 ) + 1;
        //matrix[i] = make_cuDoubleComplex(random_number, 1);
        matrix[i] = make_cuDoubleComplex(i+1, 1);
    }
    int j=0;
    int size=0;
    for (i = 0; i < M; i++) {
        for (j = 0; j < M; j++) {
            if (i > j) {
                matrix[i*M+j] = make_cuDoubleComplex(0, 0);
            } else {
                ++size;
            }
        }
    }
    for (i = 0; i < size-nnz;) {
        int index = (int) (M * M * ((double) rand() / (RAND_MAX + 1.0)));
        if (matrix[index].x == 0) {
            continue;
        }
        matrix[index] = make_cuDoubleComplex(0, 0);
        ++i;
    }
}

void initializeArrayTo2(int *array, int n)
{
    int i;
    for(i=0;i<n;i++)
    {
        array[i] = 2;
    }
}

void initializeArrayRandom(int *array, int n)
{
    int i;
    int random_number;
    for(i=0;i<n;i++)
    {
        random_number=(int)rand()/((int)RAND_MAX/(100)) + 1;
        array[i] = random_number;
    }
}

void initializeArrayRandomF(float *array, int n)
{
    int i;
    float random_number;
    for(i=0;i<n;i++)
    {
        random_number=(float)rand()/((float)RAND_MAX/(100)) + 1;
        array[i] = random_number;
    }
}

void initializeArrayRandomD(double *array, int n)
{
    int i;
    double random_number;
    for(i=0;i<n;i++)
    {
        random_number=(double)rand()/((double)RAND_MAX/(100)) + 1;
        array[i] = random_number;
    }
}

void initializeArrayRandomC(cuComplex *array, int n)
{
    int i;
    float random_number;
    for(i=0;i<n;i++)
    {
        random_number=(float)rand()/((float)RAND_MAX/(100)) + 1;
        array[i] = make_cuComplex(random_number, 1);
    }
}

void initializeArrayRandomZ(cuDoubleComplex *array, int n)
{
    int i;
    double random_number;
    for(i=0;i<n;i++)
    {
        random_number=(double)rand()/((double)RAND_MAX/(100)) + 1;
        array[i] = make_cuDoubleComplex(random_number, 1);
    }
}

void initializeArrayToZeroF(float *array, int n)
{
    int i;
    for(i=0;i<n;i++)
        array[i] = 0;
}

void initializeArrayToZeroD(double *array, int n)
{
    int i;
    for(i=0;i<n;i++)
        array[i] = 0;
}

void initializeArrayToZeroC(cuComplex *array, int n)
{
    int i;
    for(i=0;i<n;i++)
        array[i] = make_cuComplex(0, 0);
}

void initializeArrayToZeroZ(cuDoubleComplex *array, int n)
{
    int i;
    for(i=0;i<n;i++)
        array[i] = make_cuDoubleComplex(0, 0);
}

void copyArrayF(float * source, float * destination, int n) {
    int i;
    for(i=0;i<n;i++)
        destination[i] = source[i];
}

void copyArrayD(double * source, double * destination, int n) {
    int i;
    for(i=0;i<n;i++)
        destination[i] = source[i];
}

void copyArrayC(cuComplex * source, cuComplex * destination, int n) {
    int i;
    for(i=0;i<n;i++)
        destination[i] = source[i];
}

void copyArrayZ(cuDoubleComplex * source, cuDoubleComplex * destination, int n) {
    int i;
    for(i=0;i<n;i++)
        destination[i] = source[i];
}

void swapMatrixF(float * matrix, int m, int n, float * matrix_out)
{
    int i,j;
    for (i = 0; i < m; i++)
        for (j = 0; j < n; j++)
            matrix_out[i*n+j]= matrix[j*m+i];
}

void swapMatrixD(double * matrix, int m, int n, double * matrix_out)
{
    int i,j;
    for (i = 0; i < m; i++)
        for (j = 0; j < n; j++)
            matrix_out[i*n+j]= matrix[j*m+i];
}

void swapMatrixC(cuComplex * matrix, int m, int n, cuComplex * matrix_out)
{
    int i,j;
    for (i = 0; i < m; i++)
        for (j = 0; j < n; j++)
            matrix_out[i*n+j]= matrix[j*m+i];
}

void swapMatrixZ(cuDoubleComplex * matrix, int m, int n, cuDoubleComplex * matrix_out)
{
    int i,j;
    for (i = 0; i < m; i++)
        for (j = 0; j < n; j++)
            matrix_out[i*n+j]= matrix[j*m+i];
}

void stampaMatrixF(float* matrix, int M, int N)
{
    int i,j;
    for(i=0;i<M;i++)
    {
        for(j=0;j<N;j++)
            printf("%f ", matrix[i+j*M]);
        printf("\n");
    }
}

void stampaMatrixD(double* matrix, int M, int N)
{
    int i,j;
    for(i=0;i<M;i++)
    {
        for(j=0;j<N;j++)
            printf("%f ", matrix[i+j*M]);
        printf("\n");
    }
}

void stampaMatrixC(cuComplex* matrix, int M, int N)
{
    int i,j;
    for(i=0;i<M;i++)
    {
        for(j=0;j<N;j++)
            printf("[%f, %f] ", matrix[i+j*M].x, matrix[i+j*M].y);
        printf("\n");
    }
}

void stampaMatrixZ(cuDoubleComplex* matrix, int M, int N)
{
    int i,j;
    for(i=0;i<M;i++)
    {
        for(j=0;j<N;j++)
            printf("[%f, %f] ", matrix[i+j*M].x, matrix[i+j*M].y);
        printf("\n");
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
        printf("[%f, %f] ", array[i].x, array[i].y);
    printf("\n");
}

void stampaArrayZ(cuDoubleComplex* array, int n)
{
    int i;
    for(i=0;i<n;i++)
        printf("[%f, %f] ", array[i].x, array[i].y);
    printf("\n");
}

void stampaMatrixF1D(float* matrix, int M, int N)
{
    int i,j;
    for(i=0;i<M;i++)
    {
        for(j=0;j<N;j++)
            printf("%f ", matrix[i*N+j]);
        printf("\n");
    }
}

void stampaMatrixD1D(double* matrix, int M, int N)
{
    int i,j;
    for(i=0;i<M;i++)
    {
        for(j=0;j<N;j++)
            printf("%f ", matrix[i*N+j]);
        printf("\n");
    }
}

void stampaMatrixC1D(cuComplex* matrix, int M, int N)
{
    int i,j;
    for(i=0;i<M;i++)
    {
        for(j=0;j<N;j++)
            printf("[%f, %f] ", matrix[i*N+j].x, matrix[i*N+j].y);
        printf("\n");
    }
}

void stampaMatrixZ1D(cuDoubleComplex* matrix, int M, int N)
{
    int i,j;
    for(i=0;i<M;i++)
    {
        for(j=0;j<N;j++)
            printf("[%f, %f] ", matrix[i*N+j].x, matrix[i*N+j].y);
        printf("\n");
    }
}