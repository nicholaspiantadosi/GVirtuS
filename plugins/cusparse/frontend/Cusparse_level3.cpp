/*
 * gVirtuS -- A GPGPU transparent virtualization component.
 *
 * Copyright (C) 2009-2010  The University of Napoli Parthenope at Naples.
 *
 * This file is part of gVirtuS.
 *
 * gVirtuS is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * gVirtuS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with gVirtuS; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 *
 * Written by: Nicholas Piantadosi <nicholas.piantadosi@studenti.uniparthenope.it>,
 * Department of Science and Technologies
 */

#include "Cusparse.h"

using namespace std;

extern "C" cusparseStatus_t CUSPARSEAPI cusparseSbsrmm(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, cusparseOperation_t transB, int mb, int n, int kb, int nnzb, const float* alpha, const cusparseMatDescr_t descrA, const float* bsrValA, const int* bsrRowPtrA, const int* bsrColIndA, int blockDim, const float* B, int ldb, const float* beta, float* C, int ldc) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<cusparseDirection_t>(dirA);
    CusparseFrontend::AddVariableForArguments<cusparseOperation_t>(transA);
    CusparseFrontend::AddVariableForArguments<cusparseOperation_t>(transB);
    CusparseFrontend::AddVariableForArguments<int>(mb);
    CusparseFrontend::AddVariableForArguments<int>(n);
    CusparseFrontend::AddVariableForArguments<int>(kb);
    CusparseFrontend::AddVariableForArguments<int>(nnzb);
    CusparseFrontend::AddHostPointerForArguments(const_cast<float *>(alpha));
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrValA);
    CusparseFrontend::AddDevicePointerForArguments(bsrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrColIndA);
    CusparseFrontend::AddVariableForArguments<int>(blockDim);
    CusparseFrontend::AddDevicePointerForArguments(B);
    CusparseFrontend::AddVariableForArguments<int>(ldb);
    CusparseFrontend::AddHostPointerForArguments(const_cast<float *>(beta));
    CusparseFrontend::AddDevicePointerForArguments(C);
    CusparseFrontend::AddVariableForArguments<int>(ldc);
    CusparseFrontend::Execute("cusparseSbsrmm");
    if (CusparseFrontend::Success()) {
        C = (float *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseDbsrmm(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, cusparseOperation_t transB, int mb, int n, int kb, int nnzb, const double* alpha, const cusparseMatDescr_t descrA, const double* bsrValA, const int* bsrRowPtrA, const int* bsrColIndA, int blockDim, const double* B, int ldb, const double* beta, double* C, int ldc) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<cusparseDirection_t>(dirA);
    CusparseFrontend::AddVariableForArguments<cusparseOperation_t>(transA);
    CusparseFrontend::AddVariableForArguments<cusparseOperation_t>(transB);
    CusparseFrontend::AddVariableForArguments<int>(mb);
    CusparseFrontend::AddVariableForArguments<int>(n);
    CusparseFrontend::AddVariableForArguments<int>(kb);
    CusparseFrontend::AddVariableForArguments<int>(nnzb);
    CusparseFrontend::AddHostPointerForArguments(const_cast<double *>(alpha));
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrValA);
    CusparseFrontend::AddDevicePointerForArguments(bsrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrColIndA);
    CusparseFrontend::AddVariableForArguments<int>(blockDim);
    CusparseFrontend::AddDevicePointerForArguments(B);
    CusparseFrontend::AddVariableForArguments<int>(ldb);
    CusparseFrontend::AddHostPointerForArguments(const_cast<double *>(beta));
    CusparseFrontend::AddDevicePointerForArguments(C);
    CusparseFrontend::AddVariableForArguments<int>(ldc);
    CusparseFrontend::Execute("cusparseDbsrmm");
    if (CusparseFrontend::Success()) {
        C = (double *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseCbsrmm(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, cusparseOperation_t transB, int mb, int n, int kb, int nnzb, const cuComplex* alpha, const cusparseMatDescr_t descrA, const cuComplex* bsrValA, const int* bsrRowPtrA, const int* bsrColIndA, int blockDim, const cuComplex* B, int ldb, const cuComplex* beta, cuComplex* C, int ldc) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<cusparseDirection_t>(dirA);
    CusparseFrontend::AddVariableForArguments<cusparseOperation_t>(transA);
    CusparseFrontend::AddVariableForArguments<cusparseOperation_t>(transB);
    CusparseFrontend::AddVariableForArguments<int>(mb);
    CusparseFrontend::AddVariableForArguments<int>(n);
    CusparseFrontend::AddVariableForArguments<int>(kb);
    CusparseFrontend::AddVariableForArguments<int>(nnzb);
    CusparseFrontend::AddHostPointerForArguments(const_cast<cuComplex *>(alpha));
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrValA);
    CusparseFrontend::AddDevicePointerForArguments(bsrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrColIndA);
    CusparseFrontend::AddVariableForArguments<int>(blockDim);
    CusparseFrontend::AddDevicePointerForArguments(B);
    CusparseFrontend::AddVariableForArguments<int>(ldb);
    CusparseFrontend::AddHostPointerForArguments(const_cast<cuComplex *>(beta));
    CusparseFrontend::AddDevicePointerForArguments(C);
    CusparseFrontend::AddVariableForArguments<int>(ldc);
    CusparseFrontend::Execute("cusparseCbsrmm");
    if (CusparseFrontend::Success()) {
        C = (cuComplex *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseZbsrmm(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, cusparseOperation_t transB, int mb, int n, int kb, int nnzb, const cuDoubleComplex* alpha, const cusparseMatDescr_t descrA, const cuDoubleComplex* bsrValA, const int* bsrRowPtrA, const int* bsrColIndA, int blockDim, const cuDoubleComplex* B, int ldb, const cuDoubleComplex* beta, cuDoubleComplex* C, int ldc) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<cusparseDirection_t>(dirA);
    CusparseFrontend::AddVariableForArguments<cusparseOperation_t>(transA);
    CusparseFrontend::AddVariableForArguments<cusparseOperation_t>(transB);
    CusparseFrontend::AddVariableForArguments<int>(mb);
    CusparseFrontend::AddVariableForArguments<int>(n);
    CusparseFrontend::AddVariableForArguments<int>(kb);
    CusparseFrontend::AddVariableForArguments<int>(nnzb);
    CusparseFrontend::AddHostPointerForArguments(const_cast<cuDoubleComplex *>(alpha));
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrValA);
    CusparseFrontend::AddDevicePointerForArguments(bsrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrColIndA);
    CusparseFrontend::AddVariableForArguments<int>(blockDim);
    CusparseFrontend::AddDevicePointerForArguments(B);
    CusparseFrontend::AddVariableForArguments<int>(ldb);
    CusparseFrontend::AddHostPointerForArguments(const_cast<cuDoubleComplex *>(beta));
    CusparseFrontend::AddDevicePointerForArguments(C);
    CusparseFrontend::AddVariableForArguments<int>(ldc);
    CusparseFrontend::Execute("cusparseZbsrmm");
    if (CusparseFrontend::Success()) {
        C = (cuDoubleComplex *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseSbsrsm2_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, cusparseOperation_t transX, int mb, int n, int nnzb, const cusparseMatDescr_t descrA, float* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim, bsrsm2Info_t info, int* pBufferSizeInBytes) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<cusparseDirection_t>(dirA);
    CusparseFrontend::AddVariableForArguments<cusparseOperation_t>(transA);
    CusparseFrontend::AddVariableForArguments<cusparseOperation_t>(transX);
    CusparseFrontend::AddVariableForArguments<int>(mb);
    CusparseFrontend::AddVariableForArguments<int>(n);
    CusparseFrontend::AddVariableForArguments<int>(nnzb);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrSortedValA);
    CusparseFrontend::AddDevicePointerForArguments(bsrSortedRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrSortedColIndA);
    CusparseFrontend::AddVariableForArguments<int>(blockDim);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) info);
    CusparseFrontend::Execute("cusparseSbsrsm2_bufferSize");
    if (CusparseFrontend::Success()) {
        info = CusparseFrontend::GetOutputVariable<bsrsm2Info_t>();
        pBufferSizeInBytes = (int *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseDbsrsm2_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, cusparseOperation_t transX, int mb, int n, int nnzb, const cusparseMatDescr_t descrA, double* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim, bsrsm2Info_t info, int* pBufferSizeInBytes) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<cusparseDirection_t>(dirA);
    CusparseFrontend::AddVariableForArguments<cusparseOperation_t>(transA);
    CusparseFrontend::AddVariableForArguments<cusparseOperation_t>(transX);
    CusparseFrontend::AddVariableForArguments<int>(mb);
    CusparseFrontend::AddVariableForArguments<int>(n);
    CusparseFrontend::AddVariableForArguments<int>(nnzb);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrSortedValA);
    CusparseFrontend::AddDevicePointerForArguments(bsrSortedRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrSortedColIndA);
    CusparseFrontend::AddVariableForArguments<int>(blockDim);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) info);
    CusparseFrontend::Execute("cusparseDbsrsm2_bufferSize");
    if (CusparseFrontend::Success()) {
        info = CusparseFrontend::GetOutputVariable<bsrsm2Info_t>();
        pBufferSizeInBytes = (int *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseCbsrsm2_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, cusparseOperation_t transX, int mb, int n, int nnzb, const cusparseMatDescr_t descrA, cuComplex* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim, bsrsm2Info_t info, int* pBufferSizeInBytes) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<cusparseDirection_t>(dirA);
    CusparseFrontend::AddVariableForArguments<cusparseOperation_t>(transA);
    CusparseFrontend::AddVariableForArguments<cusparseOperation_t>(transX);
    CusparseFrontend::AddVariableForArguments<int>(mb);
    CusparseFrontend::AddVariableForArguments<int>(n);
    CusparseFrontend::AddVariableForArguments<int>(nnzb);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrSortedValA);
    CusparseFrontend::AddDevicePointerForArguments(bsrSortedRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrSortedColIndA);
    CusparseFrontend::AddVariableForArguments<int>(blockDim);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) info);
    CusparseFrontend::Execute("cusparseCbsrsm2_bufferSize");
    if (CusparseFrontend::Success()) {
        info = CusparseFrontend::GetOutputVariable<bsrsm2Info_t>();
        pBufferSizeInBytes = (int *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseZbsrsm2_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, cusparseOperation_t transX, int mb, int n, int nnzb, const cusparseMatDescr_t descrA, cuDoubleComplex* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim, bsrsm2Info_t info, int* pBufferSizeInBytes) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<cusparseDirection_t>(dirA);
    CusparseFrontend::AddVariableForArguments<cusparseOperation_t>(transA);
    CusparseFrontend::AddVariableForArguments<cusparseOperation_t>(transX);
    CusparseFrontend::AddVariableForArguments<int>(mb);
    CusparseFrontend::AddVariableForArguments<int>(n);
    CusparseFrontend::AddVariableForArguments<int>(nnzb);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrSortedValA);
    CusparseFrontend::AddDevicePointerForArguments(bsrSortedRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrSortedColIndA);
    CusparseFrontend::AddVariableForArguments<int>(blockDim);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) info);
    CusparseFrontend::Execute("cusparseZbsrsm2_bufferSize");
    if (CusparseFrontend::Success()) {
        info = CusparseFrontend::GetOutputVariable<bsrsm2Info_t>();
        pBufferSizeInBytes = (int *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseSbsrsm2_analysis(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, cusparseOperation_t transX, int mb, int n, int nnzb, const cusparseMatDescr_t descrA, const float* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim, bsrsm2Info_t info, cusparseSolvePolicy_t policy, void* pBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<cusparseDirection_t>(dirA);
    CusparseFrontend::AddVariableForArguments<cusparseOperation_t>(transA);
    CusparseFrontend::AddVariableForArguments<cusparseOperation_t>(transX);
    CusparseFrontend::AddVariableForArguments<int>(mb);
    CusparseFrontend::AddVariableForArguments<int>(n);
    CusparseFrontend::AddVariableForArguments<int>(nnzb);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrSortedVal);
    CusparseFrontend::AddDevicePointerForArguments(bsrSortedRowPtr);
    CusparseFrontend::AddDevicePointerForArguments(bsrSortedColInd);
    CusparseFrontend::AddVariableForArguments<int>(blockDim);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) info);
    CusparseFrontend::AddVariableForArguments<cusparseSolvePolicy_t>(policy);
    CusparseFrontend::AddDevicePointerForArguments(pBuffer);
    CusparseFrontend::Execute("cusparseSbsrsm2_analysis");
    if (CusparseFrontend::Success()) {
        info = CusparseFrontend::GetOutputVariable<bsrsm2Info_t>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseDbsrsm2_analysis(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, cusparseOperation_t transX, int mb, int n, int nnzb, const cusparseMatDescr_t descrA, const double* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim, bsrsm2Info_t info, cusparseSolvePolicy_t policy, void* pBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<cusparseDirection_t>(dirA);
    CusparseFrontend::AddVariableForArguments<cusparseOperation_t>(transA);
    CusparseFrontend::AddVariableForArguments<cusparseOperation_t>(transX);
    CusparseFrontend::AddVariableForArguments<int>(mb);
    CusparseFrontend::AddVariableForArguments<int>(n);
    CusparseFrontend::AddVariableForArguments<int>(nnzb);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrSortedVal);
    CusparseFrontend::AddDevicePointerForArguments(bsrSortedRowPtr);
    CusparseFrontend::AddDevicePointerForArguments(bsrSortedColInd);
    CusparseFrontend::AddVariableForArguments<int>(blockDim);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) info);
    CusparseFrontend::AddVariableForArguments<cusparseSolvePolicy_t>(policy);
    CusparseFrontend::AddDevicePointerForArguments(pBuffer);
    CusparseFrontend::Execute("cusparseDbsrsm2_analysis");
    if (CusparseFrontend::Success()) {
        info = CusparseFrontend::GetOutputVariable<bsrsm2Info_t>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseCbsrsm2_analysis(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, cusparseOperation_t transX, int mb, int n, int nnzb, const cusparseMatDescr_t descrA, const cuComplex* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim, bsrsm2Info_t info, cusparseSolvePolicy_t policy, void* pBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<cusparseDirection_t>(dirA);
    CusparseFrontend::AddVariableForArguments<cusparseOperation_t>(transA);
    CusparseFrontend::AddVariableForArguments<cusparseOperation_t>(transX);
    CusparseFrontend::AddVariableForArguments<int>(mb);
    CusparseFrontend::AddVariableForArguments<int>(n);
    CusparseFrontend::AddVariableForArguments<int>(nnzb);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrSortedVal);
    CusparseFrontend::AddDevicePointerForArguments(bsrSortedRowPtr);
    CusparseFrontend::AddDevicePointerForArguments(bsrSortedColInd);
    CusparseFrontend::AddVariableForArguments<int>(blockDim);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) info);
    CusparseFrontend::AddVariableForArguments<cusparseSolvePolicy_t>(policy);
    CusparseFrontend::AddDevicePointerForArguments(pBuffer);
    CusparseFrontend::Execute("cusparseCbsrsm2_analysis");
    if (CusparseFrontend::Success()) {
        info = CusparseFrontend::GetOutputVariable<bsrsm2Info_t>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseZbsrsm2_analysis(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, cusparseOperation_t transX, int mb, int n, int nnzb, const cusparseMatDescr_t descrA, const cuDoubleComplex* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim, bsrsm2Info_t info, cusparseSolvePolicy_t policy, void* pBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<cusparseDirection_t>(dirA);
    CusparseFrontend::AddVariableForArguments<cusparseOperation_t>(transA);
    CusparseFrontend::AddVariableForArguments<cusparseOperation_t>(transX);
    CusparseFrontend::AddVariableForArguments<int>(mb);
    CusparseFrontend::AddVariableForArguments<int>(n);
    CusparseFrontend::AddVariableForArguments<int>(nnzb);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrSortedVal);
    CusparseFrontend::AddDevicePointerForArguments(bsrSortedRowPtr);
    CusparseFrontend::AddDevicePointerForArguments(bsrSortedColInd);
    CusparseFrontend::AddVariableForArguments<int>(blockDim);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) info);
    CusparseFrontend::AddVariableForArguments<cusparseSolvePolicy_t>(policy);
    CusparseFrontend::AddDevicePointerForArguments(pBuffer);
    CusparseFrontend::Execute("cusparseZbsrsm2_analysis");
    if (CusparseFrontend::Success()) {
        info = CusparseFrontend::GetOutputVariable<bsrsm2Info_t>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseSbsrsm2_solve(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, cusparseOperation_t transX, int mb, int n, int nnzb, const float* alpha, const cusparseMatDescr_t descrA, const float* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim, bsrsm2Info_t info, const float* B, int ldb, float* X, int ldx, cusparseSolvePolicy_t policy, void* pBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<cusparseDirection_t>(dirA);
    CusparseFrontend::AddVariableForArguments<cusparseOperation_t>(transA);
    CusparseFrontend::AddVariableForArguments<cusparseOperation_t>(transX);
    CusparseFrontend::AddVariableForArguments<int>(mb);
    CusparseFrontend::AddVariableForArguments<int>(n);
    CusparseFrontend::AddVariableForArguments<int>(nnzb);
    CusparseFrontend::AddHostPointerForArguments(const_cast<float *>(alpha));
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrSortedVal);
    CusparseFrontend::AddDevicePointerForArguments(bsrSortedRowPtr);
    CusparseFrontend::AddDevicePointerForArguments(bsrSortedColInd);
    CusparseFrontend::AddVariableForArguments<int>(blockDim);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) info);
    CusparseFrontend::AddDevicePointerForArguments(B);
    CusparseFrontend::AddVariableForArguments<int>(ldb);
    CusparseFrontend::AddDevicePointerForArguments(X);
    CusparseFrontend::AddVariableForArguments<int>(ldx);
    CusparseFrontend::AddVariableForArguments<cusparseSolvePolicy_t>(policy);
    CusparseFrontend::AddDevicePointerForArguments(pBuffer);
    CusparseFrontend::Execute("cusparseSbsrsm2_solve");
    if (CusparseFrontend::Success()) {
        X = (float *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseDbsrsm2_solve(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, cusparseOperation_t transX, int mb, int n, int nnzb, const double* alpha, const cusparseMatDescr_t descrA, const double* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim, bsrsm2Info_t info, const double* B, int ldb, double* X, int ldx, cusparseSolvePolicy_t policy, void* pBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<cusparseDirection_t>(dirA);
    CusparseFrontend::AddVariableForArguments<cusparseOperation_t>(transA);
    CusparseFrontend::AddVariableForArguments<cusparseOperation_t>(transX);
    CusparseFrontend::AddVariableForArguments<int>(mb);
    CusparseFrontend::AddVariableForArguments<int>(n);
    CusparseFrontend::AddVariableForArguments<int>(nnzb);
    CusparseFrontend::AddHostPointerForArguments(const_cast<double *>(alpha));
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrSortedVal);
    CusparseFrontend::AddDevicePointerForArguments(bsrSortedRowPtr);
    CusparseFrontend::AddDevicePointerForArguments(bsrSortedColInd);
    CusparseFrontend::AddVariableForArguments<int>(blockDim);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) info);
    CusparseFrontend::AddDevicePointerForArguments(B);
    CusparseFrontend::AddVariableForArguments<int>(ldb);
    CusparseFrontend::AddDevicePointerForArguments(X);
    CusparseFrontend::AddVariableForArguments<int>(ldx);
    CusparseFrontend::AddVariableForArguments<cusparseSolvePolicy_t>(policy);
    CusparseFrontend::AddDevicePointerForArguments(pBuffer);
    CusparseFrontend::Execute("cusparseDbsrsm2_solve");
    if (CusparseFrontend::Success()) {
        X = (double *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseCbsrsm2_solve(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, cusparseOperation_t transX, int mb, int n, int nnzb, const cuComplex* alpha, const cusparseMatDescr_t descrA, const cuComplex* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim, bsrsm2Info_t info, const cuComplex* B, int ldb, cuComplex* X, int ldx, cusparseSolvePolicy_t policy, void* pBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<cusparseDirection_t>(dirA);
    CusparseFrontend::AddVariableForArguments<cusparseOperation_t>(transA);
    CusparseFrontend::AddVariableForArguments<cusparseOperation_t>(transX);
    CusparseFrontend::AddVariableForArguments<int>(mb);
    CusparseFrontend::AddVariableForArguments<int>(n);
    CusparseFrontend::AddVariableForArguments<int>(nnzb);
    CusparseFrontend::AddHostPointerForArguments(const_cast<cuComplex *>(alpha));
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrSortedVal);
    CusparseFrontend::AddDevicePointerForArguments(bsrSortedRowPtr);
    CusparseFrontend::AddDevicePointerForArguments(bsrSortedColInd);
    CusparseFrontend::AddVariableForArguments<int>(blockDim);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) info);
    CusparseFrontend::AddDevicePointerForArguments(B);
    CusparseFrontend::AddVariableForArguments<int>(ldb);
    CusparseFrontend::AddDevicePointerForArguments(X);
    CusparseFrontend::AddVariableForArguments<int>(ldx);
    CusparseFrontend::AddVariableForArguments<cusparseSolvePolicy_t>(policy);
    CusparseFrontend::AddDevicePointerForArguments(pBuffer);
    CusparseFrontend::Execute("cusparseCbsrsm2_solve");
    if (CusparseFrontend::Success()) {
        X = (cuComplex *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseZbsrsm2_solve(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, cusparseOperation_t transX, int mb, int n, int nnzb, const cuDoubleComplex* alpha, const cusparseMatDescr_t descrA, const cuDoubleComplex* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim, bsrsm2Info_t info, const cuDoubleComplex* B, int ldb, cuDoubleComplex* X, int ldx, cusparseSolvePolicy_t policy, void* pBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<cusparseDirection_t>(dirA);
    CusparseFrontend::AddVariableForArguments<cusparseOperation_t>(transA);
    CusparseFrontend::AddVariableForArguments<cusparseOperation_t>(transX);
    CusparseFrontend::AddVariableForArguments<int>(mb);
    CusparseFrontend::AddVariableForArguments<int>(n);
    CusparseFrontend::AddVariableForArguments<int>(nnzb);
    CusparseFrontend::AddHostPointerForArguments(const_cast<cuDoubleComplex *>(alpha));
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrSortedVal);
    CusparseFrontend::AddDevicePointerForArguments(bsrSortedRowPtr);
    CusparseFrontend::AddDevicePointerForArguments(bsrSortedColInd);
    CusparseFrontend::AddVariableForArguments<int>(blockDim);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) info);
    CusparseFrontend::AddDevicePointerForArguments(B);
    CusparseFrontend::AddVariableForArguments<int>(ldb);
    CusparseFrontend::AddDevicePointerForArguments(X);
    CusparseFrontend::AddVariableForArguments<int>(ldx);
    CusparseFrontend::AddVariableForArguments<cusparseSolvePolicy_t>(policy);
    CusparseFrontend::AddDevicePointerForArguments(pBuffer);
    CusparseFrontend::Execute("cusparseZbsrsm2_solve");
    if (CusparseFrontend::Success()) {
        X = (cuDoubleComplex *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseXbsrsm2_zeroPivot(cusparseHandle_t handle, bsrsm2Info_t info, int* position) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) info);
    CusparseFrontend::Execute("cusparseXbsrsm2_zeroPivot");
    if (CusparseFrontend::Success()) {
        position = (int *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}