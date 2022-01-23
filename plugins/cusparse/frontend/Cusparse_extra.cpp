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

extern "C" cusparseStatus_t CUSPARSEAPI cusparseScsrgeam2_bufferSizeExt(cusparseHandle_t handle, int m, int n, const float* alpha, const cusparseMatDescr_t descrA, int nnzA, const float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const float* beta, const cusparseMatDescr_t descrB, int nnzB, const float* csrSortedValB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const cusparseMatDescr_t descrC, const float* csrSortedValC, const int* csrSortedRowPtrC, const int* csrSortedColIndC, size_t* pBufferSizeInBytes) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddVariableForArguments<int>(n);
    CusparseFrontend::AddHostPointerForArguments(const_cast<float *>(alpha));
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusparseFrontend::AddVariableForArguments<int>(nnzA);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedValA);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedColIndA);
    CusparseFrontend::AddHostPointerForArguments(const_cast<float *>(beta));
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrB);
    CusparseFrontend::AddVariableForArguments<int>(nnzB);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedValB);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedRowPtrB);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedColIndB);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrC);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedValC);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedRowPtrC);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedColIndC);
    CusparseFrontend::AddHostPointerForArguments<size_t>(pBufferSizeInBytes);
    CusparseFrontend::Execute("cusparseScsrgeam2_bufferSizeExt");
    if (CusparseFrontend::Success()) {
        *pBufferSizeInBytes = *CusparseFrontend::GetOutputHostPointer<size_t>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseDcsrgeam2_bufferSizeExt(cusparseHandle_t handle, int m, int n, const double* alpha, const cusparseMatDescr_t descrA, int nnzA, const double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const double* beta, const cusparseMatDescr_t descrB, int nnzB, const double* csrSortedValB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const cusparseMatDescr_t descrC, const double* csrSortedValC, const int* csrSortedRowPtrC, const int* csrSortedColIndC, size_t* pBufferSizeInBytes) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddVariableForArguments<int>(n);
    CusparseFrontend::AddHostPointerForArguments(const_cast<double *>(alpha));
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusparseFrontend::AddVariableForArguments<int>(nnzA);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedValA);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedColIndA);
    CusparseFrontend::AddHostPointerForArguments(const_cast<double *>(beta));
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrB);
    CusparseFrontend::AddVariableForArguments<int>(nnzB);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedValB);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedRowPtrB);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedColIndB);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrC);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedValC);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedRowPtrC);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedColIndC);
    CusparseFrontend::AddHostPointerForArguments<size_t>(pBufferSizeInBytes);
    CusparseFrontend::Execute("cusparseDcsrgeam2_bufferSizeExt");
    if (CusparseFrontend::Success()) {
        *pBufferSizeInBytes = *CusparseFrontend::GetOutputHostPointer<size_t>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseCcsrgeam2_bufferSizeExt(cusparseHandle_t handle, int m, int n, const cuComplex* alpha, const cusparseMatDescr_t descrA, int nnzA, const cuComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cuComplex* beta, const cusparseMatDescr_t descrB, int nnzB, const cuComplex* csrSortedValB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const cusparseMatDescr_t descrC, const cuComplex* csrSortedValC, const int* csrSortedRowPtrC, const int* csrSortedColIndC, size_t* pBufferSizeInBytes) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddVariableForArguments<int>(n);
    CusparseFrontend::AddHostPointerForArguments(const_cast<cuComplex *>(alpha));
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusparseFrontend::AddVariableForArguments<int>(nnzA);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedValA);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedColIndA);
    CusparseFrontend::AddHostPointerForArguments(const_cast<cuComplex *>(beta));
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrB);
    CusparseFrontend::AddVariableForArguments<int>(nnzB);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedValB);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedRowPtrB);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedColIndB);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrC);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedValC);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedRowPtrC);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedColIndC);
    CusparseFrontend::AddHostPointerForArguments<size_t>(pBufferSizeInBytes);
    CusparseFrontend::Execute("cusparseCcsrgeam2_bufferSizeExt");
    if (CusparseFrontend::Success()) {
        *pBufferSizeInBytes = *CusparseFrontend::GetOutputHostPointer<size_t>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseZcsrgeam2_bufferSizeExt(cusparseHandle_t handle, int m, int n, const cuDoubleComplex* alpha, const cusparseMatDescr_t descrA, int nnzA, const cuDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cuDoubleComplex* beta, const cusparseMatDescr_t descrB, int nnzB, const cuDoubleComplex* csrSortedValB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const cusparseMatDescr_t descrC, const cuDoubleComplex* csrSortedValC, const int* csrSortedRowPtrC, const int* csrSortedColIndC, size_t* pBufferSizeInBytes) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddVariableForArguments<int>(n);
    CusparseFrontend::AddHostPointerForArguments(const_cast<cuDoubleComplex *>(alpha));
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusparseFrontend::AddVariableForArguments<int>(nnzA);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedValA);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedColIndA);
    CusparseFrontend::AddHostPointerForArguments(const_cast<cuDoubleComplex *>(beta));
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrB);
    CusparseFrontend::AddVariableForArguments<int>(nnzB);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedValB);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedRowPtrB);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedColIndB);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrC);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedValC);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedRowPtrC);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedColIndC);
    CusparseFrontend::AddHostPointerForArguments<size_t>(pBufferSizeInBytes);
    CusparseFrontend::Execute("cusparseZcsrgeam2_bufferSizeExt");
    if (CusparseFrontend::Success()) {
        *pBufferSizeInBytes = *CusparseFrontend::GetOutputHostPointer<size_t>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseXcsrgeam2Nnz(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA, int nnzA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cusparseMatDescr_t descrB, int nnzB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const cusparseMatDescr_t descrC, int* csrSortedRowPtrC, int* nnzTotalDevHostPtr, void* workspace) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddVariableForArguments<int>(n);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusparseFrontend::AddVariableForArguments<int>(nnzA);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedColIndA);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrB);
    CusparseFrontend::AddVariableForArguments<int>(nnzB);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedRowPtrB);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedColIndB);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrC);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedRowPtrC);
    CusparseFrontend::AddDevicePointerForArguments(workspace);
    CusparseFrontend::AddHostPointerForArguments<int>(nnzTotalDevHostPtr);
    CusparseFrontend::Execute("cusparseXcsrgeam2Nnz");
    if (CusparseFrontend::Success()) {
        csrSortedRowPtrC = (int *)CusparseFrontend::GetOutputDevicePointer();
        *nnzTotalDevHostPtr = *CusparseFrontend::GetOutputHostPointer<int>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseScsrgeam2(cusparseHandle_t handle, int m, int n, const float* alpha, const cusparseMatDescr_t descrA, int nnzA, const float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const float* beta, const cusparseMatDescr_t descrB, int nnzB, const float* csrSortedValB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const cusparseMatDescr_t descrC, float* csrSortedValC, int* csrSortedRowPtrC, int* csrSortedColIndC, void* pBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddVariableForArguments<int>(n);
    CusparseFrontend::AddHostPointerForArguments(const_cast<float *>(alpha));
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusparseFrontend::AddVariableForArguments<int>(nnzA);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedValA);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedColIndA);
    CusparseFrontend::AddHostPointerForArguments(const_cast<float *>(beta));
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrB);
    CusparseFrontend::AddVariableForArguments<int>(nnzB);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedValB);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedRowPtrB);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedColIndB);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrC);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedValC);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedRowPtrC);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedColIndC);
    CusparseFrontend::AddDevicePointerForArguments(pBuffer);
    CusparseFrontend::Execute("cusparseScsrgeam2");
    if (CusparseFrontend::Success()) {
        csrSortedValC = (float *)CusparseFrontend::GetOutputDevicePointer();
        csrSortedRowPtrC = (int *)CusparseFrontend::GetOutputDevicePointer();
        csrSortedColIndC = (int *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseDcsrgeam2(cusparseHandle_t handle, int m, int n, const double* alpha, const cusparseMatDescr_t descrA, int nnzA, const double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const double* beta, const cusparseMatDescr_t descrB, int nnzB, const double* csrSortedValB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const cusparseMatDescr_t descrC, double* csrSortedValC, int* csrSortedRowPtrC, int* csrSortedColIndC, void* pBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddVariableForArguments<int>(n);
    CusparseFrontend::AddHostPointerForArguments(const_cast<double *>(alpha));
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusparseFrontend::AddVariableForArguments<int>(nnzA);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedValA);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedColIndA);
    CusparseFrontend::AddHostPointerForArguments(const_cast<double *>(beta));
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrB);
    CusparseFrontend::AddVariableForArguments<int>(nnzB);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedValB);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedRowPtrB);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedColIndB);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrC);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedValC);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedRowPtrC);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedColIndC);
    CusparseFrontend::AddDevicePointerForArguments(pBuffer);
    CusparseFrontend::Execute("cusparseDcsrgeam2");
    if (CusparseFrontend::Success()) {
        csrSortedValC = (double *)CusparseFrontend::GetOutputDevicePointer();
        csrSortedRowPtrC = (int *)CusparseFrontend::GetOutputDevicePointer();
        csrSortedColIndC = (int *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseCcsrgeam2(cusparseHandle_t handle, int m, int n, const cuComplex* alpha, const cusparseMatDescr_t descrA, int nnzA, const cuComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cuComplex* beta, const cusparseMatDescr_t descrB, int nnzB, const cuComplex* csrSortedValB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const cusparseMatDescr_t descrC, cuComplex* csrSortedValC, int* csrSortedRowPtrC, int* csrSortedColIndC, void* pBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddVariableForArguments<int>(n);
    CusparseFrontend::AddHostPointerForArguments(const_cast<cuComplex *>(alpha));
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusparseFrontend::AddVariableForArguments<int>(nnzA);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedValA);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedColIndA);
    CusparseFrontend::AddHostPointerForArguments(const_cast<cuComplex *>(beta));
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrB);
    CusparseFrontend::AddVariableForArguments<int>(nnzB);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedValB);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedRowPtrB);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedColIndB);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrC);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedValC);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedRowPtrC);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedColIndC);
    CusparseFrontend::AddDevicePointerForArguments(pBuffer);
    CusparseFrontend::Execute("cusparseCcsrgeam2");
    if (CusparseFrontend::Success()) {
        csrSortedValC = (cuComplex *)CusparseFrontend::GetOutputDevicePointer();
        csrSortedRowPtrC = (int *)CusparseFrontend::GetOutputDevicePointer();
        csrSortedColIndC = (int *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseZcsrgeam2(cusparseHandle_t handle, int m, int n, const cuDoubleComplex* alpha, const cusparseMatDescr_t descrA, int nnzA, const cuDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cuDoubleComplex* beta, const cusparseMatDescr_t descrB, int nnzB, const cuDoubleComplex* csrSortedValB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const cusparseMatDescr_t descrC, cuDoubleComplex* csrSortedValC, int* csrSortedRowPtrC, int* csrSortedColIndC, void* pBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddVariableForArguments<int>(n);
    CusparseFrontend::AddHostPointerForArguments(const_cast<cuDoubleComplex *>(alpha));
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusparseFrontend::AddVariableForArguments<int>(nnzA);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedValA);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedColIndA);
    CusparseFrontend::AddHostPointerForArguments(const_cast<cuDoubleComplex *>(beta));
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrB);
    CusparseFrontend::AddVariableForArguments<int>(nnzB);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedValB);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedRowPtrB);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedColIndB);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrC);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedValC);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedRowPtrC);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedColIndC);
    CusparseFrontend::AddDevicePointerForArguments(pBuffer);
    CusparseFrontend::Execute("cusparseZcsrgeam2");
    if (CusparseFrontend::Success()) {
        csrSortedValC = (cuDoubleComplex *)CusparseFrontend::GetOutputDevicePointer();
        csrSortedRowPtrC = (int *)CusparseFrontend::GetOutputDevicePointer();
        csrSortedColIndC = (int *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}