#include<stdio.h>
#include<cusparse.h>

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

int main(void)
{
    printf("cusparseCreateMatDescr\n");
    cusparseMatDescr_t descrA;
    CHECK_CUSPARSE(cusparseCreateMatDescr(&descrA));
    printf("descrA: %d\n\n", descrA);

    printf("cusparseGetMatDiagType\n");
    cusparseDiagType_t diagType = cusparseGetMatDiagType(descrA);
    printf("diagType: %d\n", diagType);
    printf("cusparseSetMatDiagType to 1\n");
    CHECK_CUSPARSE(cusparseSetMatDiagType(descrA, CUSPARSE_DIAG_TYPE_UNIT));
    printf("cusparseGetMatDiagType\n");
    printf("diagType: %d\n", cusparseGetMatDiagType(descrA));
    printf("cusparseSetMatDiagType to 0\n");
    CHECK_CUSPARSE(cusparseSetMatDiagType(descrA, CUSPARSE_DIAG_TYPE_NON_UNIT));
    printf("cusparseGetMatDiagType\n");
    printf("diagType: %d\n\n", cusparseGetMatDiagType(descrA));

    printf("cusparseGetMatFillMode\n");
    cusparseFillMode_t fillMode = cusparseGetMatFillMode(descrA);
    printf("fillMode: %d\n", fillMode);
    printf("cusparseSetMatFillMode to 1\n");
    CHECK_CUSPARSE(cusparseSetMatFillMode(descrA, CUSPARSE_FILL_MODE_UPPER));
    printf("cusparseGetMatFillMode\n");
    printf("fillMode: %d\n", cusparseGetMatFillMode(descrA));
    printf("cusparseSetMatFillMode to 0\n");
    CHECK_CUSPARSE(cusparseSetMatFillMode(descrA, CUSPARSE_FILL_MODE_LOWER));
    printf("cusparseGetMatFillMode\n");
    printf("fillMode: %d\n\n", cusparseGetMatFillMode(descrA));

    printf("cusparseGetMatIndexBase\n");
    cusparseIndexBase_t indexBase = cusparseGetMatIndexBase(descrA);
    printf("indexBase: %d\n", indexBase);
    printf("cusparseSetMatIndexBase to 1\n");
    CHECK_CUSPARSE(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ONE));
    printf("cusparseGetMatIndexBase\n");
    printf("indexBase: %d\n", cusparseGetMatIndexBase(descrA));
    printf("cusparseSetMatIndexBase to 0\n");
    CHECK_CUSPARSE(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));
    printf("cusparseGetMatIndexBase\n");
    printf("indexBase: %d\n\n", cusparseGetMatIndexBase(descrA));

    printf("cusparseGetMatType\n");
    cusparseMatrixType_t matrixType = cusparseGetMatType(descrA);
    printf("matrixType: %d\n", matrixType);
    printf("cusparseSetMatType to 1\n");
    CHECK_CUSPARSE(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_SYMMETRIC));
    printf("cusparseGetMatType\n");
    printf("matrixType: %d\n", cusparseGetMatType(descrA));
    printf("cusparseSetMatType to 2\n");
    CHECK_CUSPARSE(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_HERMITIAN));
    printf("cusparseGetMatType\n");
    printf("matrixType: %d\n", cusparseGetMatType(descrA));
    printf("cusparseSetMatType to 3\n");
    CHECK_CUSPARSE(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_TRIANGULAR));
    printf("cusparseGetMatType\n");
    printf("matrixType: %d\n", cusparseGetMatType(descrA));
    printf("cusparseSetMatType to 0\n");
    CHECK_CUSPARSE(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
    printf("cusparseGetMatType\n");
    printf("matrixType: %d\n", cusparseGetMatType(descrA));

    printf("cusparseDestroyMatDescr\n");
    CHECK_CUSPARSE(cusparseDestroyMatDescr(descrA));
    printf("descrA destroyed\n");

    return 0;
}
