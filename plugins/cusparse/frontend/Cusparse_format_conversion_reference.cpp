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
 *             Department of Science and Technologies
 */

#include "Cusparse.h"

using namespace std;

extern "C" cusparseStatus_t CUSPARSEAPI cusparseSbsr2csr(cusparseHandle_t handle, cusparseDirection_t dir, int mb, int nb, const cusparseMatDescr_t descrA, const float* bsrValA, const int* bsrRowPtrA, const int* bsrColIndA, int blockDim, const cusparseMatDescr_t descrC, float* csrValC, int* csrRowPtrC, int* csrColIndC) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CusparseFrontend::AddVariableForArguments<cusparseDirection_t>(dir);
    CusparseFrontend::AddVariableForArguments<int>(mb);
    CusparseFrontend::AddVariableForArguments<int>(nb);
    CusparseFrontend::AddVariableForArguments<long long int>((long long int)descrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrValA);
    CusparseFrontend::AddDevicePointerForArguments(bsrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrColIndA);
    CusparseFrontend::AddVariableForArguments<int>(blockDim);
    CusparseFrontend::AddVariableForArguments<long long int>((long long int)descrC);
    CusparseFrontend::AddDevicePointerForArguments(csrValC);
    CusparseFrontend::AddDevicePointerForArguments(csrRowPtrC);
    CusparseFrontend::AddDevicePointerForArguments(csrColIndC);
    CusparseFrontend::Execute("cusparseSbsr2csr");
    if (CusparseFrontend::Success()) {
        csrValC = (float *)CusparseFrontend::GetOutputDevicePointer();
        csrRowPtrC = (int *)CusparseFrontend::GetOutputDevicePointer();
        csrColIndC = (int *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseDbsr2csr(cusparseHandle_t handle, cusparseDirection_t dir, int mb, int nb, const cusparseMatDescr_t descrA, const double* bsrValA, const int* bsrRowPtrA, const int* bsrColIndA, int blockDim, const cusparseMatDescr_t descrC, double* csrValC, int* csrRowPtrC, int* csrColIndC) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CusparseFrontend::AddVariableForArguments<cusparseDirection_t>(dir);
    CusparseFrontend::AddVariableForArguments<int>(mb);
    CusparseFrontend::AddVariableForArguments<int>(nb);
    CusparseFrontend::AddVariableForArguments<long long int>((long long int)descrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrValA);
    CusparseFrontend::AddDevicePointerForArguments(bsrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrColIndA);
    CusparseFrontend::AddVariableForArguments<int>(blockDim);
    CusparseFrontend::AddVariableForArguments<long long int>((long long int)descrC);
    CusparseFrontend::AddDevicePointerForArguments(csrValC);
    CusparseFrontend::AddDevicePointerForArguments(csrRowPtrC);
    CusparseFrontend::AddDevicePointerForArguments(csrColIndC);
    CusparseFrontend::Execute("cusparseDbsr2csr");
    if (CusparseFrontend::Success()) {
        csrValC = (double *)CusparseFrontend::GetOutputDevicePointer();
        csrRowPtrC = (int *)CusparseFrontend::GetOutputDevicePointer();
        csrColIndC = (int *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseCbsr2csr(cusparseHandle_t handle, cusparseDirection_t dir, int mb, int nb, const cusparseMatDescr_t descrA, const cuComplex* bsrValA, const int* bsrRowPtrA, const int* bsrColIndA, int blockDim, const cusparseMatDescr_t descrC, cuComplex* csrValC, int* csrRowPtrC, int* csrColIndC) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CusparseFrontend::AddVariableForArguments<cusparseDirection_t>(dir);
    CusparseFrontend::AddVariableForArguments<int>(mb);
    CusparseFrontend::AddVariableForArguments<int>(nb);
    CusparseFrontend::AddVariableForArguments<long long int>((long long int)descrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrValA);
    CusparseFrontend::AddDevicePointerForArguments(bsrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrColIndA);
    CusparseFrontend::AddVariableForArguments<int>(blockDim);
    CusparseFrontend::AddVariableForArguments<long long int>((long long int)descrC);
    CusparseFrontend::AddDevicePointerForArguments(csrValC);
    CusparseFrontend::AddDevicePointerForArguments(csrRowPtrC);
    CusparseFrontend::AddDevicePointerForArguments(csrColIndC);
    CusparseFrontend::Execute("cusparseCbsr2csr");
    if (CusparseFrontend::Success()) {
        csrValC = (cuComplex *)CusparseFrontend::GetOutputDevicePointer();
        csrRowPtrC = (int *)CusparseFrontend::GetOutputDevicePointer();
        csrColIndC = (int *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseZbsr2csr(cusparseHandle_t handle, cusparseDirection_t dir, int mb, int nb, const cusparseMatDescr_t descrA, const cuDoubleComplex* bsrValA, const int* bsrRowPtrA, const int* bsrColIndA, int blockDim, const cusparseMatDescr_t descrC, cuDoubleComplex* csrValC, int* csrRowPtrC, int* csrColIndC) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CusparseFrontend::AddVariableForArguments<cusparseDirection_t>(dir);
    CusparseFrontend::AddVariableForArguments<int>(mb);
    CusparseFrontend::AddVariableForArguments<int>(nb);
    CusparseFrontend::AddVariableForArguments<long long int>((long long int)descrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrValA);
    CusparseFrontend::AddDevicePointerForArguments(bsrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrColIndA);
    CusparseFrontend::AddVariableForArguments<int>(blockDim);
    CusparseFrontend::AddVariableForArguments<long long int>((long long int)descrC);
    CusparseFrontend::AddDevicePointerForArguments(csrValC);
    CusparseFrontend::AddDevicePointerForArguments(csrRowPtrC);
    CusparseFrontend::AddDevicePointerForArguments(csrColIndC);
    CusparseFrontend::Execute("cusparseZbsr2csr");
    if (CusparseFrontend::Success()) {
        csrValC = (cuDoubleComplex *)CusparseFrontend::GetOutputDevicePointer();
        csrRowPtrC = (int *)CusparseFrontend::GetOutputDevicePointer();
        csrColIndC = (int *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseXcsr2bsrNnz(cusparseHandle_t handle, cusparseDirection_t dir, int m, int n, const cusparseMatDescr_t descrA, const int* csrRowPtrA, const int* csrColIndA, int blockDim, const cusparseMatDescr_t descrC, int* bsrRowPtrC, int* nnzTotalDevHostPtr) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CusparseFrontend::AddVariableForArguments<cusparseDirection_t>(dir);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddVariableForArguments<int>(n);
    CusparseFrontend::AddVariableForArguments<long long int>((long long int)descrA);
    CusparseFrontend::AddDevicePointerForArguments(csrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(csrColIndA);
    CusparseFrontend::AddVariableForArguments<int>(blockDim);
    CusparseFrontend::AddVariableForArguments<long long int>((long long int)descrC);
    CusparseFrontend::AddDevicePointerForArguments(bsrRowPtrC);
    CusparseFrontend::AddHostPointerForArguments<int>(nnzTotalDevHostPtr);
    CusparseFrontend::Execute("cusparseXcsr2bsrNnz");
    if (CusparseFrontend::Success()) {
        bsrRowPtrC = (int *)CusparseFrontend::GetOutputDevicePointer();
        *nnzTotalDevHostPtr = (int) CusparseFrontend::GetOutputVariable<int>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseScsr2bsr(cusparseHandle_t handle, cusparseDirection_t dir, int m, int n, const cusparseMatDescr_t descrA, const float* csrValA, const int* csrRowPtrA, const int* csrColIndA, int blockDim, const cusparseMatDescr_t descrC, float* bsrValC, int* bsrRowPtrC, int* bsrColIndC) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CusparseFrontend::AddVariableForArguments<cusparseDirection_t>(dir);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddVariableForArguments<int>(n);
    CusparseFrontend::AddVariableForArguments<long long int>((long long int)descrA);
    CusparseFrontend::AddDevicePointerForArguments(csrValA);
    CusparseFrontend::AddDevicePointerForArguments(csrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(csrColIndA);
    CusparseFrontend::AddVariableForArguments<int>(blockDim);
    CusparseFrontend::AddVariableForArguments<long long int>((long long int)descrC);
    CusparseFrontend::AddDevicePointerForArguments(bsrValC);
    CusparseFrontend::AddDevicePointerForArguments(bsrRowPtrC);
    CusparseFrontend::AddDevicePointerForArguments(bsrColIndC);
    CusparseFrontend::Execute("cusparseScsr2bsr");
    if (CusparseFrontend::Success()) {
        bsrValC = (float *)CusparseFrontend::GetOutputDevicePointer();
        bsrRowPtrC = (int *)CusparseFrontend::GetOutputDevicePointer();
        bsrColIndC = (int *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseDcsr2bsr(cusparseHandle_t handle, cusparseDirection_t dir, int m, int n, const cusparseMatDescr_t descrA, const double* csrValA, const int* csrRowPtrA, const int* csrColIndA, int blockDim, const cusparseMatDescr_t descrC, double* bsrValC, int* bsrRowPtrC, int* bsrColIndC) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CusparseFrontend::AddVariableForArguments<cusparseDirection_t>(dir);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddVariableForArguments<int>(n);
    CusparseFrontend::AddVariableForArguments<long long int>((long long int)descrA);
    CusparseFrontend::AddDevicePointerForArguments(csrValA);
    CusparseFrontend::AddDevicePointerForArguments(csrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(csrColIndA);
    CusparseFrontend::AddVariableForArguments<int>(blockDim);
    CusparseFrontend::AddVariableForArguments<long long int>((long long int)descrC);
    CusparseFrontend::AddDevicePointerForArguments(bsrValC);
    CusparseFrontend::AddDevicePointerForArguments(bsrRowPtrC);
    CusparseFrontend::AddDevicePointerForArguments(bsrColIndC);
    CusparseFrontend::Execute("cusparseDcsr2bsr");
    if (CusparseFrontend::Success()) {
        bsrValC = (double *)CusparseFrontend::GetOutputDevicePointer();
        bsrRowPtrC = (int *)CusparseFrontend::GetOutputDevicePointer();
        bsrColIndC = (int *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseCcsr2bsr(cusparseHandle_t handle, cusparseDirection_t dir, int m, int n, const cusparseMatDescr_t descrA, const cuComplex* csrValA, const int* csrRowPtrA, const int* csrColIndA, int blockDim, const cusparseMatDescr_t descrC, cuComplex* bsrValC, int* bsrRowPtrC, int* bsrColIndC) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CusparseFrontend::AddVariableForArguments<cusparseDirection_t>(dir);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddVariableForArguments<int>(n);
    CusparseFrontend::AddVariableForArguments<long long int>((long long int)descrA);
    CusparseFrontend::AddDevicePointerForArguments(csrValA);
    CusparseFrontend::AddDevicePointerForArguments(csrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(csrColIndA);
    CusparseFrontend::AddVariableForArguments<int>(blockDim);
    CusparseFrontend::AddVariableForArguments<long long int>((long long int)descrC);
    CusparseFrontend::AddDevicePointerForArguments(bsrValC);
    CusparseFrontend::AddDevicePointerForArguments(bsrRowPtrC);
    CusparseFrontend::AddDevicePointerForArguments(bsrColIndC);
    CusparseFrontend::Execute("cusparseCcsr2bsr");
    if (CusparseFrontend::Success()) {
        bsrValC = (cuComplex *)CusparseFrontend::GetOutputDevicePointer();
        bsrRowPtrC = (int *)CusparseFrontend::GetOutputDevicePointer();
        bsrColIndC = (int *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseZcsr2bsr(cusparseHandle_t handle, cusparseDirection_t dir, int m, int n, const cusparseMatDescr_t descrA, const cuDoubleComplex* csrValA, const int* csrRowPtrA, const int* csrColIndA, int blockDim, const cusparseMatDescr_t descrC, cuDoubleComplex* bsrValC, int* bsrRowPtrC, int* bsrColIndC) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CusparseFrontend::AddVariableForArguments<cusparseDirection_t>(dir);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddVariableForArguments<int>(n);
    CusparseFrontend::AddVariableForArguments<long long int>((long long int)descrA);
    CusparseFrontend::AddDevicePointerForArguments(csrValA);
    CusparseFrontend::AddDevicePointerForArguments(csrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(csrColIndA);
    CusparseFrontend::AddVariableForArguments<int>(blockDim);
    CusparseFrontend::AddVariableForArguments<long long int>((long long int)descrC);
    CusparseFrontend::AddDevicePointerForArguments(bsrValC);
    CusparseFrontend::AddDevicePointerForArguments(bsrRowPtrC);
    CusparseFrontend::AddDevicePointerForArguments(bsrColIndC);
    CusparseFrontend::Execute("cusparseZcsr2bsr");
    if (CusparseFrontend::Success()) {
        bsrValC = (cuDoubleComplex *)CusparseFrontend::GetOutputDevicePointer();
        bsrRowPtrC = (int *)CusparseFrontend::GetOutputDevicePointer();
        bsrColIndC = (int *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseSdense2csr(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA, const float* A, int lda, const int* nnzPerRow, float* csrValA, int* csrRowPtrA, int* csrColIndA) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddVariableForArguments<int>(n);
    CusparseFrontend::AddVariableForArguments<long long int>((long long int)descrA);
    CusparseFrontend::AddDevicePointerForArguments(A);
    CusparseFrontend::AddVariableForArguments<int>(lda);
    CusparseFrontend::AddDevicePointerForArguments(nnzPerRow);
    CusparseFrontend::AddDevicePointerForArguments(csrValA);
    CusparseFrontend::AddDevicePointerForArguments(csrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(csrColIndA);
    CusparseFrontend::Execute("cusparseSdense2csr");
    if (CusparseFrontend::Success()) {
        csrValA = (float *)CusparseFrontend::GetOutputDevicePointer();
        csrRowPtrA = (int *)CusparseFrontend::GetOutputDevicePointer();
        csrColIndA = (int *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseDdense2csr(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA, const double* A, int lda, const int* nnzPerRow, double* csrValA, int* csrRowPtrA, int* csrColIndA) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddVariableForArguments<int>(n);
    CusparseFrontend::AddVariableForArguments<long long int>((long long int)descrA);
    CusparseFrontend::AddDevicePointerForArguments(A);
    CusparseFrontend::AddVariableForArguments<int>(lda);
    CusparseFrontend::AddDevicePointerForArguments(nnzPerRow);
    CusparseFrontend::AddDevicePointerForArguments(csrValA);
    CusparseFrontend::AddDevicePointerForArguments(csrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(csrColIndA);
    CusparseFrontend::Execute("cusparseDdense2csr");
    if (CusparseFrontend::Success()) {
        csrValA = (double *)CusparseFrontend::GetOutputDevicePointer();
        csrRowPtrA = (int *)CusparseFrontend::GetOutputDevicePointer();
        csrColIndA = (int *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseCdense2csr(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA, const cuComplex* A, int lda, const int* nnzPerRow, cuComplex* csrValA, int* csrRowPtrA, int* csrColIndA) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddVariableForArguments<int>(n);
    CusparseFrontend::AddVariableForArguments<long long int>((long long int)descrA);
    CusparseFrontend::AddDevicePointerForArguments(A);
    CusparseFrontend::AddVariableForArguments<int>(lda);
    CusparseFrontend::AddDevicePointerForArguments(nnzPerRow);
    CusparseFrontend::AddDevicePointerForArguments(csrValA);
    CusparseFrontend::AddDevicePointerForArguments(csrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(csrColIndA);
    CusparseFrontend::Execute("cusparseCdense2csr");
    if (CusparseFrontend::Success()) {
        csrValA = (cuComplex *)CusparseFrontend::GetOutputDevicePointer();
        csrRowPtrA = (int *)CusparseFrontend::GetOutputDevicePointer();
        csrColIndA = (int *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseZdense2csr(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA, const cuDoubleComplex* A, int lda, const int* nnzPerRow, cuDoubleComplex* csrValA, int* csrRowPtrA, int* csrColIndA) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddVariableForArguments<int>(n);
    CusparseFrontend::AddVariableForArguments<long long int>((long long int)descrA);
    CusparseFrontend::AddDevicePointerForArguments(A);
    CusparseFrontend::AddVariableForArguments<int>(lda);
    CusparseFrontend::AddDevicePointerForArguments(nnzPerRow);
    CusparseFrontend::AddDevicePointerForArguments(csrValA);
    CusparseFrontend::AddDevicePointerForArguments(csrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(csrColIndA);
    CusparseFrontend::Execute("cusparseZdense2csr");
    if (CusparseFrontend::Success()) {
        csrValA = (cuDoubleComplex *)CusparseFrontend::GetOutputDevicePointer();
        csrRowPtrA = (int *)CusparseFrontend::GetOutputDevicePointer();
        csrColIndA = (int *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseSnnz(cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n, const cusparseMatDescr_t descrA, const float* A, int lda, int* nnzPerRowColumn, int* nnzTotalDevHostPtr) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CusparseFrontend::AddVariableForArguments<cusparseDirection_t>(dirA);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddVariableForArguments<int>(n);
    CusparseFrontend::AddVariableForArguments<long long int>((long long int)descrA);
    CusparseFrontend::AddDevicePointerForArguments(A);
    CusparseFrontend::AddVariableForArguments<int>(lda);
    CusparseFrontend::AddDevicePointerForArguments(nnzPerRowColumn);
    CusparseFrontend::AddHostPointerForArguments<int>(nnzTotalDevHostPtr);
    CusparseFrontend::Execute("cusparseSnnz");
    if (CusparseFrontend::Success()) {
        nnzPerRowColumn = (int *)CusparseFrontend::GetOutputDevicePointer();
        *nnzTotalDevHostPtr = (int) CusparseFrontend::GetOutputVariable<int>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseDnnz(cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n, const cusparseMatDescr_t descrA, const double* A, int lda, int* nnzPerRowColumn, int* nnzTotalDevHostPtr) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CusparseFrontend::AddVariableForArguments<cusparseDirection_t>(dirA);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddVariableForArguments<int>(n);
    CusparseFrontend::AddVariableForArguments<long long int>((long long int)descrA);
    CusparseFrontend::AddDevicePointerForArguments(A);
    CusparseFrontend::AddVariableForArguments<int>(lda);
    CusparseFrontend::AddDevicePointerForArguments(nnzPerRowColumn);
    CusparseFrontend::AddHostPointerForArguments<int>(nnzTotalDevHostPtr);
    CusparseFrontend::Execute("cusparseDnnz");
    if (CusparseFrontend::Success()) {
        nnzPerRowColumn = (int *)CusparseFrontend::GetOutputDevicePointer();
        *nnzTotalDevHostPtr = (int) CusparseFrontend::GetOutputVariable<int>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseCnnz(cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n, const cusparseMatDescr_t descrA, const cuComplex* A, int lda, int* nnzPerRowColumn, int* nnzTotalDevHostPtr) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CusparseFrontend::AddVariableForArguments<cusparseDirection_t>(dirA);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddVariableForArguments<int>(n);
    CusparseFrontend::AddVariableForArguments<long long int>((long long int)descrA);
    CusparseFrontend::AddDevicePointerForArguments(A);
    CusparseFrontend::AddVariableForArguments<int>(lda);
    CusparseFrontend::AddDevicePointerForArguments(nnzPerRowColumn);
    CusparseFrontend::AddHostPointerForArguments<int>(nnzTotalDevHostPtr);
    CusparseFrontend::Execute("cusparseCnnz");
    if (CusparseFrontend::Success()) {
        nnzPerRowColumn = (int *)CusparseFrontend::GetOutputDevicePointer();
        *nnzTotalDevHostPtr = (int) CusparseFrontend::GetOutputVariable<int>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseZnnz(cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n, const cusparseMatDescr_t descrA, const cuDoubleComplex* A, int lda, int* nnzPerRowColumn, int* nnzTotalDevHostPtr) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CusparseFrontend::AddVariableForArguments<cusparseDirection_t>(dirA);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddVariableForArguments<int>(n);
    CusparseFrontend::AddVariableForArguments<long long int>((long long int)descrA);
    CusparseFrontend::AddDevicePointerForArguments(A);
    CusparseFrontend::AddVariableForArguments<int>(lda);
    CusparseFrontend::AddDevicePointerForArguments(nnzPerRowColumn);
    CusparseFrontend::AddHostPointerForArguments<int>(nnzTotalDevHostPtr);
    CusparseFrontend::Execute("cusparseZnnz");
    if (CusparseFrontend::Success()) {
        nnzPerRowColumn = (int *)CusparseFrontend::GetOutputDevicePointer();
        *nnzTotalDevHostPtr = (int) CusparseFrontend::GetOutputVariable<int>();
    }
    return CusparseFrontend::GetExitCode();
}