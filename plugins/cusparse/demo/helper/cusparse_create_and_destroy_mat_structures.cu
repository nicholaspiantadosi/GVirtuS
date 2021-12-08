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
    printf("cusparseCreateCsrsv2Info\n");
    csrsv2Info_t csrsv2Info;
    CHECK_CUSPARSE(cusparseCreateCsrsv2Info(&csrsv2Info));
    printf("csrsv2Info: %d\n", csrsv2Info);

    printf("cusparseDestroyCsrsv2Info\n");
    CHECK_CUSPARSE(cusparseDestroyCsrsv2Info(csrsv2Info));
    printf("csrsv2Info destroyed\n\n");

    printf("cusparseCreateCsrsm2Info\n");
    csrsm2Info_t csrsm2Info;
    CHECK_CUSPARSE(cusparseCreateCsrsm2Info(&csrsm2Info));
    printf("csrsm2Info: %d\n", csrsm2Info);

    printf("cusparseDestroyCsrsm2Info\n");
    CHECK_CUSPARSE(cusparseDestroyCsrsm2Info(csrsm2Info));
    printf("csrsm2Info destroyed\n\n");

    printf("cusparseCreateCsric02Info\n");
    csric02Info_t csric02Info;
    CHECK_CUSPARSE(cusparseCreateCsric02Info(&csric02Info));
    printf("csric02Info: %d\n", csric02Info);

    printf("cusparseDestroyCsric02Info\n");
    CHECK_CUSPARSE(cusparseDestroyCsric02Info(csric02Info));
    printf("csric02Info destroyed\n\n");

    printf("cusparseCreateCsrilu02Info\n");
    csrilu02Info_t csrilu02Info;
    CHECK_CUSPARSE(cusparseCreateCsrilu02Info(&csrilu02Info));
    printf("csrilu02Info: %d\n", csrilu02Info);

    printf("cusparseDestroyCsrilu02Info\n");
    CHECK_CUSPARSE(cusparseDestroyCsrilu02Info(csrilu02Info));
    printf("csrilu02Info destroyed\n\n");

    printf("cusparseCreateBsrsv2Info\n");
    bsrsv2Info_t bsrsv2Info;
    CHECK_CUSPARSE(cusparseCreateBsrsv2Info(&bsrsv2Info));
    printf("bsrsv2Info: %d\n", bsrsv2Info);

    printf("cusparseDestroyBsrsv2Info\n");
    CHECK_CUSPARSE(cusparseDestroyBsrsv2Info(bsrsv2Info));
    printf("bsrsv2Info destroyed\n\n");

    printf("cusparseCreateBsrsm2Info\n");
    bsrsm2Info_t bsrsm2Info;
    CHECK_CUSPARSE(cusparseCreateBsrsm2Info(&bsrsm2Info));
    printf("bsrsm2Info: %d\n", bsrsm2Info);

    printf("cusparseDestroyBsrsm2Info\n");
    CHECK_CUSPARSE(cusparseDestroyBsrsm2Info(bsrsm2Info));
    printf("bsrsm2Info destroyed\n\n");

    printf("cusparseCreateBsric02Info\n");
    bsric02Info_t bsric02Info;
    CHECK_CUSPARSE(cusparseCreateBsric02Info(&bsric02Info));
    printf("bsric02Info: %d\n", bsric02Info);

    printf("cusparseDestroyBsric02Info\n");
    CHECK_CUSPARSE(cusparseDestroyBsric02Info(bsric02Info));
    printf("bsric02Info destroyed\n\n");

    printf("cusparseCreateBsrilu02Info\n");
    bsrilu02Info_t bsrilu02Info;
    CHECK_CUSPARSE(cusparseCreateBsrilu02Info(&bsrilu02Info));
    printf("bsrilu02Info: %d\n", bsrilu02Info);

    printf("cusparseDestroyBsrilu02Info\n");
    CHECK_CUSPARSE(cusparseDestroyBsrilu02Info(bsrilu02Info));
    printf("bsrilu02Info destroyed\n\n");

    printf("cusparseCreateCsrgemm2Info\n");
    csrgemm2Info_t csrgemm2Info;
    CHECK_CUSPARSE(cusparseCreateCsrgemm2Info(&csrgemm2Info));
    printf("csrgemm2Info: %d\n", csrgemm2Info);

    printf("cusparseDestroyCsrgemm2Info\n");
    CHECK_CUSPARSE(cusparseDestroyCsrgemm2Info(csrgemm2Info));
    printf("csrgemm2Info destroyed\n\n");

    printf("cusparseCreatePruneInfo\n");
    pruneInfo_t pruneInfo;
    CHECK_CUSPARSE(cusparseCreatePruneInfo(&pruneInfo));
    printf("pruneInfo: %d\n", pruneInfo);

    printf("cusparseDestroyPruneInfo\n");
    CHECK_CUSPARSE(cusparseDestroyPruneInfo(pruneInfo));
    printf("pruneInfo destroyed\n\n");

    return 0;
}