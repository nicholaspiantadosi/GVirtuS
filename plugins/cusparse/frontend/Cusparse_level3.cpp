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