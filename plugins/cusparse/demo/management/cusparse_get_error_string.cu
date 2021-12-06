#include<stdio.h>
#include<cusparse.h>

int main(void)
{
    printf("CUSPARSE_STATUS_ALLOC_FAILED: %s\n", cusparseGetErrorString(CUSPARSE_STATUS_ALLOC_FAILED));
    return 0;
}
