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

extern "C" cusparseStatus_t CUSPARSEAPI cusparseScsric02_bufferSize(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csric02Info_t info, int* pBufferSize) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddVariableForArguments<int>(nnz);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedValA);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedColIndA);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) info);
    CusparseFrontend::Execute("cusparseScsric02_bufferSize");
    if (CusparseFrontend::Success()) {
        info = CusparseFrontend::GetOutputVariable<csric02Info_t>();
        pBufferSize = (int *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseDcsric02_bufferSize(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csric02Info_t info, int* pBufferSize) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddVariableForArguments<int>(nnz);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedValA);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedColIndA);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) info);
    CusparseFrontend::Execute("cusparseDcsric02_bufferSize");
    if (CusparseFrontend::Success()) {
        info = CusparseFrontend::GetOutputVariable<csric02Info_t>();
        pBufferSize = (int *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseCcsric02_bufferSize(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, cuComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csric02Info_t info, int* pBufferSize) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddVariableForArguments<int>(nnz);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedValA);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedColIndA);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) info);
    CusparseFrontend::Execute("cusparseCcsric02_bufferSize");
    if (CusparseFrontend::Success()) {
        info = CusparseFrontend::GetOutputVariable<csric02Info_t>();
        pBufferSize = (int *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseZcsric02_bufferSize(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, cuDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csric02Info_t info, int* pBufferSize) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddVariableForArguments<int>(nnz);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedValA);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedColIndA);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) info);
    CusparseFrontend::Execute("cusparseZcsric02_bufferSize");
    if (CusparseFrontend::Success()) {
        info = CusparseFrontend::GetOutputVariable<csric02Info_t>();
        pBufferSize = (int *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseScsric02_analysis(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, const float* csrValA, const int* csrRowPtrA, const int* csrColIndA, csric02Info_t info, cusparseSolvePolicy_t policy, void* pBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddVariableForArguments<int>(nnz);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusparseFrontend::AddDevicePointerForArguments(csrValA);
    CusparseFrontend::AddDevicePointerForArguments(csrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(csrColIndA);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) info);
    CusparseFrontend::AddVariableForArguments<cusparseSolvePolicy_t>(policy);
    CusparseFrontend::AddDevicePointerForArguments(pBuffer);
    CusparseFrontend::Execute("cusparseScsric02_analysis");
    if (CusparseFrontend::Success()) {
        info = CusparseFrontend::GetOutputVariable<csric02Info_t>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseDcsric02_analysis(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, const double* csrValA, const int* csrRowPtrA, const int* csrColIndA, csric02Info_t info, cusparseSolvePolicy_t policy, void* pBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddVariableForArguments<int>(nnz);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusparseFrontend::AddDevicePointerForArguments(csrValA);
    CusparseFrontend::AddDevicePointerForArguments(csrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(csrColIndA);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) info);
    CusparseFrontend::AddVariableForArguments<cusparseSolvePolicy_t>(policy);
    CusparseFrontend::AddDevicePointerForArguments(pBuffer);
    CusparseFrontend::Execute("cusparseDcsric02_analysis");
    if (CusparseFrontend::Success()) {
        info = CusparseFrontend::GetOutputVariable<csric02Info_t>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseCcsric02_analysis(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, const cuComplex* csrValA, const int* csrRowPtrA, const int* csrColIndA, csric02Info_t info, cusparseSolvePolicy_t policy, void* pBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddVariableForArguments<int>(nnz);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusparseFrontend::AddDevicePointerForArguments(csrValA);
    CusparseFrontend::AddDevicePointerForArguments(csrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(csrColIndA);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) info);
    CusparseFrontend::AddVariableForArguments<cusparseSolvePolicy_t>(policy);
    CusparseFrontend::AddDevicePointerForArguments(pBuffer);
    CusparseFrontend::Execute("cusparseCcsric02_analysis");
    if (CusparseFrontend::Success()) {
        info = CusparseFrontend::GetOutputVariable<csric02Info_t>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseZcsric02_analysis(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, const cuDoubleComplex* csrValA, const int* csrRowPtrA, const int* csrColIndA, csric02Info_t info, cusparseSolvePolicy_t policy, void* pBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddVariableForArguments<int>(nnz);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusparseFrontend::AddDevicePointerForArguments(csrValA);
    CusparseFrontend::AddDevicePointerForArguments(csrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(csrColIndA);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) info);
    CusparseFrontend::AddVariableForArguments<cusparseSolvePolicy_t>(policy);
    CusparseFrontend::AddDevicePointerForArguments(pBuffer);
    CusparseFrontend::Execute("cusparseZcsric02_analysis");
    if (CusparseFrontend::Success()) {
        info = CusparseFrontend::GetOutputVariable<csric02Info_t>();
    }
    return CusparseFrontend::GetExitCode();
}


extern "C" cusparseStatus_t CUSPARSEAPI cusparseScsric02(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, float* csrValA, const int* csrRowPtrA, const int* csrColIndA, csric02Info_t info, cusparseSolvePolicy_t policy, void* pBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddVariableForArguments<int>(nnz);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusparseFrontend::AddDevicePointerForArguments(csrValA);
    CusparseFrontend::AddDevicePointerForArguments(csrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(csrColIndA);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) info);
    CusparseFrontend::AddVariableForArguments<cusparseSolvePolicy_t>(policy);
    CusparseFrontend::AddDevicePointerForArguments(pBuffer);
    CusparseFrontend::Execute("cusparseScsric02");
    if (CusparseFrontend::Success()) {
        csrValA = (float *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseDcsric02(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, double* csrValA, const int* csrRowPtrA, const int* csrColIndA, csric02Info_t info, cusparseSolvePolicy_t policy, void* pBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddVariableForArguments<int>(nnz);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusparseFrontend::AddDevicePointerForArguments(csrValA);
    CusparseFrontend::AddDevicePointerForArguments(csrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(csrColIndA);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) info);
    CusparseFrontend::AddVariableForArguments<cusparseSolvePolicy_t>(policy);
    CusparseFrontend::AddDevicePointerForArguments(pBuffer);
    CusparseFrontend::Execute("cusparseDcsric02");
    if (CusparseFrontend::Success()) {
        csrValA = (double *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseCcsric02(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, cuComplex* csrValA, const int* csrRowPtrA, const int* csrColIndA, csric02Info_t info, cusparseSolvePolicy_t policy, void* pBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddVariableForArguments<int>(nnz);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusparseFrontend::AddDevicePointerForArguments(csrValA);
    CusparseFrontend::AddDevicePointerForArguments(csrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(csrColIndA);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) info);
    CusparseFrontend::AddVariableForArguments<cusparseSolvePolicy_t>(policy);
    CusparseFrontend::AddDevicePointerForArguments(pBuffer);
    CusparseFrontend::Execute("cusparseCcsric02");
    if (CusparseFrontend::Success()) {
        csrValA = (cuComplex *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseZcsric02(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, cuDoubleComplex* csrValA, const int* csrRowPtrA, const int* csrColIndA, csric02Info_t info, cusparseSolvePolicy_t policy, void* pBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddVariableForArguments<int>(nnz);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusparseFrontend::AddDevicePointerForArguments(csrValA);
    CusparseFrontend::AddDevicePointerForArguments(csrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(csrColIndA);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) info);
    CusparseFrontend::AddVariableForArguments<cusparseSolvePolicy_t>(policy);
    CusparseFrontend::AddDevicePointerForArguments(pBuffer);
    CusparseFrontend::Execute("cusparseZcsric02");
    if (CusparseFrontend::Success()) {
        csrValA = (cuDoubleComplex *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseXcsric02_zeroPivot(cusparseHandle_t handle, csric02Info_t info, int* position) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) info);
    CusparseFrontend::Execute("cusparseXcsric02_zeroPivot");
    if (CusparseFrontend::Success()) {
        position = (int *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseSbsric02_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb, const cusparseMatDescr_t descrA, float* bsrValA, const int* bsrRowPtrA, const int* bsrColIndA, int blockDim, bsric02Info_t info, int* pBufferSize) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<cusparseDirection_t>(dirA);
    CusparseFrontend::AddVariableForArguments<int>(mb);
    CusparseFrontend::AddVariableForArguments<int>(nnzb);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrValA);
    CusparseFrontend::AddDevicePointerForArguments(bsrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrColIndA);
    CusparseFrontend::AddVariableForArguments<int>(blockDim);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) info);
    CusparseFrontend::Execute("cusparseSbsric02_bufferSize");
    if (CusparseFrontend::Success()) {
        info = CusparseFrontend::GetOutputVariable<bsric02Info_t>();
        pBufferSize = (int *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseDbsric02_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb, const cusparseMatDescr_t descrA, double* bsrValA, const int* bsrRowPtrA, const int* bsrColIndA, int blockDim, bsric02Info_t info, int* pBufferSize) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<cusparseDirection_t>(dirA);
    CusparseFrontend::AddVariableForArguments<int>(mb);
    CusparseFrontend::AddVariableForArguments<int>(nnzb);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrValA);
    CusparseFrontend::AddDevicePointerForArguments(bsrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrColIndA);
    CusparseFrontend::AddVariableForArguments<int>(blockDim);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) info);
    CusparseFrontend::Execute("cusparseDbsric02_bufferSize");
    if (CusparseFrontend::Success()) {
        info = CusparseFrontend::GetOutputVariable<bsric02Info_t>();
        pBufferSize = (int *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseCbsric02_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb, const cusparseMatDescr_t descrA, cuComplex* bsrValA, const int* bsrRowPtrA, const int* bsrColIndA, int blockDim, bsric02Info_t info, int* pBufferSize) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<cusparseDirection_t>(dirA);
    CusparseFrontend::AddVariableForArguments<int>(mb);
    CusparseFrontend::AddVariableForArguments<int>(nnzb);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrValA);
    CusparseFrontend::AddDevicePointerForArguments(bsrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrColIndA);
    CusparseFrontend::AddVariableForArguments<int>(blockDim);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) info);
    CusparseFrontend::Execute("cusparseCbsric02_bufferSize");
    if (CusparseFrontend::Success()) {
        info = CusparseFrontend::GetOutputVariable<bsric02Info_t>();
        pBufferSize = (int *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseZbsric02_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb, const cusparseMatDescr_t descrA, cuDoubleComplex* bsrValA, const int* bsrRowPtrA, const int* bsrColIndA, int blockDim, bsric02Info_t info, int* pBufferSize) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<cusparseDirection_t>(dirA);
    CusparseFrontend::AddVariableForArguments<int>(mb);
    CusparseFrontend::AddVariableForArguments<int>(nnzb);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrValA);
    CusparseFrontend::AddDevicePointerForArguments(bsrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrColIndA);
    CusparseFrontend::AddVariableForArguments<int>(blockDim);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) info);
    CusparseFrontend::Execute("cusparseZbsric02_bufferSize");
    if (CusparseFrontend::Success()) {
        info = CusparseFrontend::GetOutputVariable<bsric02Info_t>();
        pBufferSize = (int *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseSbsric02_analysis(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb, const cusparseMatDescr_t descrA, const float* bsrValA, const int* bsrRowPtrA, const int* bsrColIndA, int blockDim, bsric02Info_t info, cusparseSolvePolicy_t policy, void* pBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<cusparseDirection_t>(dirA);
    CusparseFrontend::AddVariableForArguments<int>(mb);
    CusparseFrontend::AddVariableForArguments<int>(nnzb);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrValA);
    CusparseFrontend::AddDevicePointerForArguments(bsrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrColIndA);
    CusparseFrontend::AddVariableForArguments<int>(blockDim);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) info);
    CusparseFrontend::AddVariableForArguments<cusparseSolvePolicy_t>(policy);
    CusparseFrontend::AddDevicePointerForArguments(pBuffer);
    CusparseFrontend::Execute("cusparseSbsric02_analysis");
    if (CusparseFrontend::Success()) {
        info = CusparseFrontend::GetOutputVariable<bsric02Info_t>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseDbsric02_analysis(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb, const cusparseMatDescr_t descrA, const double* bsrValA, const int* bsrRowPtrA, const int* bsrColIndA, int blockDim, bsric02Info_t info, cusparseSolvePolicy_t policy, void* pBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<cusparseDirection_t>(dirA);
    CusparseFrontend::AddVariableForArguments<int>(mb);
    CusparseFrontend::AddVariableForArguments<int>(nnzb);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrValA);
    CusparseFrontend::AddDevicePointerForArguments(bsrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrColIndA);
    CusparseFrontend::AddVariableForArguments<int>(blockDim);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) info);
    CusparseFrontend::AddVariableForArguments<cusparseSolvePolicy_t>(policy);
    CusparseFrontend::AddDevicePointerForArguments(pBuffer);
    CusparseFrontend::Execute("cusparseDbsric02_analysis");
    if (CusparseFrontend::Success()) {
        info = CusparseFrontend::GetOutputVariable<bsric02Info_t>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseCbsric02_analysis(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb, const cusparseMatDescr_t descrA, const cuComplex* bsrValA, const int* bsrRowPtrA, const int* bsrColIndA, int blockDim, bsric02Info_t info, cusparseSolvePolicy_t policy, void* pBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<cusparseDirection_t>(dirA);
    CusparseFrontend::AddVariableForArguments<int>(mb);
    CusparseFrontend::AddVariableForArguments<int>(nnzb);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrValA);
    CusparseFrontend::AddDevicePointerForArguments(bsrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrColIndA);
    CusparseFrontend::AddVariableForArguments<int>(blockDim);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) info);
    CusparseFrontend::AddVariableForArguments<cusparseSolvePolicy_t>(policy);
    CusparseFrontend::AddDevicePointerForArguments(pBuffer);
    CusparseFrontend::Execute("cusparseCbsric02_analysis");
    if (CusparseFrontend::Success()) {
        info = CusparseFrontend::GetOutputVariable<bsric02Info_t>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseZbsric02_analysis(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb, const cusparseMatDescr_t descrA, const cuDoubleComplex* bsrValA, const int* bsrRowPtrA, const int* bsrColIndA, int blockDim, bsric02Info_t info, cusparseSolvePolicy_t policy, void* pBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<cusparseDirection_t>(dirA);
    CusparseFrontend::AddVariableForArguments<int>(mb);
    CusparseFrontend::AddVariableForArguments<int>(nnzb);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrValA);
    CusparseFrontend::AddDevicePointerForArguments(bsrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrColIndA);
    CusparseFrontend::AddVariableForArguments<int>(blockDim);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) info);
    CusparseFrontend::AddVariableForArguments<cusparseSolvePolicy_t>(policy);
    CusparseFrontend::AddDevicePointerForArguments(pBuffer);
    CusparseFrontend::Execute("cusparseZbsric02_analysis");
    if (CusparseFrontend::Success()) {
        info = CusparseFrontend::GetOutputVariable<bsric02Info_t>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseSbsric02(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb, const cusparseMatDescr_t descrA, float* bsrValA, const int* bsrRowPtrA, const int* bsrColIndA, int blockDim, bsric02Info_t info, cusparseSolvePolicy_t policy, void* pBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<cusparseDirection_t>(dirA);
    CusparseFrontend::AddVariableForArguments<int>(mb);
    CusparseFrontend::AddVariableForArguments<int>(nnzb);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrValA);
    CusparseFrontend::AddDevicePointerForArguments(bsrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrColIndA);
    CusparseFrontend::AddVariableForArguments<int>(blockDim);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) info);
    CusparseFrontend::AddVariableForArguments<cusparseSolvePolicy_t>(policy);
    CusparseFrontend::AddDevicePointerForArguments(pBuffer);
    CusparseFrontend::Execute("cusparseSbsric02");
    if (CusparseFrontend::Success()) {
        bsrValA = (float *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseDbsric02(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb, const cusparseMatDescr_t descrA, double* bsrValA, const int* bsrRowPtrA, const int* bsrColIndA, int blockDim, bsric02Info_t info, cusparseSolvePolicy_t policy, void* pBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<cusparseDirection_t>(dirA);
    CusparseFrontend::AddVariableForArguments<int>(mb);
    CusparseFrontend::AddVariableForArguments<int>(nnzb);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrValA);
    CusparseFrontend::AddDevicePointerForArguments(bsrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrColIndA);
    CusparseFrontend::AddVariableForArguments<int>(blockDim);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) info);
    CusparseFrontend::AddVariableForArguments<cusparseSolvePolicy_t>(policy);
    CusparseFrontend::AddDevicePointerForArguments(pBuffer);
    CusparseFrontend::Execute("cusparseDbsric02");
    if (CusparseFrontend::Success()) {
        bsrValA = (double *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseCbsric02(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb, const cusparseMatDescr_t descrA, cuComplex* bsrValA, const int* bsrRowPtrA, const int* bsrColIndA, int blockDim, bsric02Info_t info, cusparseSolvePolicy_t policy, void* pBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<cusparseDirection_t>(dirA);
    CusparseFrontend::AddVariableForArguments<int>(mb);
    CusparseFrontend::AddVariableForArguments<int>(nnzb);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrValA);
    CusparseFrontend::AddDevicePointerForArguments(bsrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrColIndA);
    CusparseFrontend::AddVariableForArguments<int>(blockDim);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) info);
    CusparseFrontend::AddVariableForArguments<cusparseSolvePolicy_t>(policy);
    CusparseFrontend::AddDevicePointerForArguments(pBuffer);
    CusparseFrontend::Execute("cusparseCbsric02");
    if (CusparseFrontend::Success()) {
        bsrValA = (cuComplex *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseZbsric02(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb, const cusparseMatDescr_t descrA, cuDoubleComplex* bsrValA, const int* bsrRowPtrA, const int* bsrColIndA, int blockDim, bsric02Info_t info, cusparseSolvePolicy_t policy, void* pBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<cusparseDirection_t>(dirA);
    CusparseFrontend::AddVariableForArguments<int>(mb);
    CusparseFrontend::AddVariableForArguments<int>(nnzb);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrValA);
    CusparseFrontend::AddDevicePointerForArguments(bsrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrColIndA);
    CusparseFrontend::AddVariableForArguments<int>(blockDim);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) info);
    CusparseFrontend::AddVariableForArguments<cusparseSolvePolicy_t>(policy);
    CusparseFrontend::AddDevicePointerForArguments(pBuffer);
    CusparseFrontend::Execute("cusparseZbsric02");
    if (CusparseFrontend::Success()) {
        bsrValA = (cuDoubleComplex *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseXbsric02_zeroPivot(cusparseHandle_t handle, bsric02Info_t info, int* position) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) info);
    CusparseFrontend::Execute("cusparseXbsric02_zeroPivot");
    if (CusparseFrontend::Success()) {
        position = (int *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseScsrilu02_numericBoost(cusparseHandle_t handle, csrilu02Info_t info, int enable_boost, double* tol, float* boost_val) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) info);
    CusparseFrontend::AddVariableForArguments<int>(enable_boost);
    CusparseFrontend::AddDevicePointerForArguments(tol);
    CusparseFrontend::AddDevicePointerForArguments(boost_val);
    CusparseFrontend::Execute("cusparseScsrilu02_numericBoost");
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseDcsrilu02_numericBoost(cusparseHandle_t handle, csrilu02Info_t info, int enable_boost, double* tol, double* boost_val) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) info);
    CusparseFrontend::AddVariableForArguments<int>(enable_boost);
    CusparseFrontend::AddDevicePointerForArguments(tol);
    CusparseFrontend::AddDevicePointerForArguments(boost_val);
    CusparseFrontend::Execute("cusparseDcsrilu02_numericBoost");
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseCcsrilu02_numericBoost(cusparseHandle_t handle, csrilu02Info_t info, int enable_boost, double* tol, cuComplex* boost_val) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) info);
    CusparseFrontend::AddVariableForArguments<int>(enable_boost);
    CusparseFrontend::AddDevicePointerForArguments(tol);
    CusparseFrontend::AddDevicePointerForArguments(boost_val);
    CusparseFrontend::Execute("cusparseCcsrilu02_numericBoost");
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseZcsrilu02_numericBoost(cusparseHandle_t handle, csrilu02Info_t info, int enable_boost, double* tol, cuDoubleComplex* boost_val) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) info);
    CusparseFrontend::AddVariableForArguments<int>(enable_boost);
    CusparseFrontend::AddDevicePointerForArguments(tol);
    CusparseFrontend::AddDevicePointerForArguments(boost_val);
    CusparseFrontend::Execute("cusparseZcsrilu02_numericBoost");
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseScsrilu02_bufferSize(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrilu02Info_t info, int* pBufferSize) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddVariableForArguments<int>(nnz);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedValA);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedColIndA);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) info);
    CusparseFrontend::Execute("cusparseScsrilu02_bufferSize");
    if (CusparseFrontend::Success()) {
        info = CusparseFrontend::GetOutputVariable<csrilu02Info_t>();
        pBufferSize = (int *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseDcsrilu02_bufferSize(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrilu02Info_t info, int* pBufferSize) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddVariableForArguments<int>(nnz);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedValA);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedColIndA);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) info);
    CusparseFrontend::Execute("cusparseDcsrilu02_bufferSize");
    if (CusparseFrontend::Success()) {
        info = CusparseFrontend::GetOutputVariable<csrilu02Info_t>();
        pBufferSize = (int *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseCcsrilu02_bufferSize(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, cuComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrilu02Info_t info, int* pBufferSize) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddVariableForArguments<int>(nnz);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedValA);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedColIndA);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) info);
    CusparseFrontend::Execute("cusparseCcsrilu02_bufferSize");
    if (CusparseFrontend::Success()) {
        info = CusparseFrontend::GetOutputVariable<csrilu02Info_t>();
        pBufferSize = (int *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseZcsrilu02_bufferSize(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, cuDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrilu02Info_t info, int* pBufferSize) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddVariableForArguments<int>(nnz);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedValA);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedColIndA);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) info);
    CusparseFrontend::Execute("cusparseZcsrilu02_bufferSize");
    if (CusparseFrontend::Success()) {
        info = CusparseFrontend::GetOutputVariable<csrilu02Info_t>();
        pBufferSize = (int *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseScsrilu02_analysis(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, const float* csrValA, const int* csrRowPtrA, const int* csrColIndA, csrilu02Info_t info, cusparseSolvePolicy_t policy, void* pBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddVariableForArguments<int>(nnz);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusparseFrontend::AddDevicePointerForArguments(csrValA);
    CusparseFrontend::AddDevicePointerForArguments(csrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(csrColIndA);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) info);
    CusparseFrontend::AddVariableForArguments<cusparseSolvePolicy_t>(policy);
    CusparseFrontend::AddDevicePointerForArguments(pBuffer);
    CusparseFrontend::Execute("cusparseScsrilu02_analysis");
    if (CusparseFrontend::Success()) {
        info = CusparseFrontend::GetOutputVariable<csrilu02Info_t>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseDcsrilu02_analysis(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, const double* csrValA, const int* csrRowPtrA, const int* csrColIndA, csrilu02Info_t info, cusparseSolvePolicy_t policy, void* pBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddVariableForArguments<int>(nnz);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusparseFrontend::AddDevicePointerForArguments(csrValA);
    CusparseFrontend::AddDevicePointerForArguments(csrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(csrColIndA);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) info);
    CusparseFrontend::AddVariableForArguments<cusparseSolvePolicy_t>(policy);
    CusparseFrontend::AddDevicePointerForArguments(pBuffer);
    CusparseFrontend::Execute("cusparseDcsrilu02_analysis");
    if (CusparseFrontend::Success()) {
        info = CusparseFrontend::GetOutputVariable<csrilu02Info_t>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseCcsrilu02_analysis(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, const cuComplex* csrValA, const int* csrRowPtrA, const int* csrColIndA, csrilu02Info_t info, cusparseSolvePolicy_t policy, void* pBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddVariableForArguments<int>(nnz);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusparseFrontend::AddDevicePointerForArguments(csrValA);
    CusparseFrontend::AddDevicePointerForArguments(csrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(csrColIndA);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) info);
    CusparseFrontend::AddVariableForArguments<cusparseSolvePolicy_t>(policy);
    CusparseFrontend::AddDevicePointerForArguments(pBuffer);
    CusparseFrontend::Execute("cusparseCcsrilu02_analysis");
    if (CusparseFrontend::Success()) {
        info = CusparseFrontend::GetOutputVariable<csrilu02Info_t>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseZcsrilu02_analysis(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, const cuDoubleComplex* csrValA, const int* csrRowPtrA, const int* csrColIndA, csrilu02Info_t info, cusparseSolvePolicy_t policy, void* pBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddVariableForArguments<int>(nnz);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusparseFrontend::AddDevicePointerForArguments(csrValA);
    CusparseFrontend::AddDevicePointerForArguments(csrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(csrColIndA);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) info);
    CusparseFrontend::AddVariableForArguments<cusparseSolvePolicy_t>(policy);
    CusparseFrontend::AddDevicePointerForArguments(pBuffer);
    CusparseFrontend::Execute("cusparseZcsrilu02_analysis");
    if (CusparseFrontend::Success()) {
        info = CusparseFrontend::GetOutputVariable<csrilu02Info_t>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseScsrilu02(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, float* csrValA, const int* csrRowPtrA, const int* csrColIndA, csrilu02Info_t info, cusparseSolvePolicy_t policy, void* pBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddVariableForArguments<int>(nnz);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusparseFrontend::AddDevicePointerForArguments(csrValA);
    CusparseFrontend::AddDevicePointerForArguments(csrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(csrColIndA);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) info);
    CusparseFrontend::AddVariableForArguments<cusparseSolvePolicy_t>(policy);
    CusparseFrontend::AddDevicePointerForArguments(pBuffer);
    CusparseFrontend::Execute("cusparseScsrilu02");
    if (CusparseFrontend::Success()) {
        csrValA = (float *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseDcsrilu02(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, double* csrValA, const int* csrRowPtrA, const int* csrColIndA, csrilu02Info_t info, cusparseSolvePolicy_t policy, void* pBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddVariableForArguments<int>(nnz);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusparseFrontend::AddDevicePointerForArguments(csrValA);
    CusparseFrontend::AddDevicePointerForArguments(csrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(csrColIndA);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) info);
    CusparseFrontend::AddVariableForArguments<cusparseSolvePolicy_t>(policy);
    CusparseFrontend::AddDevicePointerForArguments(pBuffer);
    CusparseFrontend::Execute("cusparseDcsrilu02");
    if (CusparseFrontend::Success()) {
        csrValA = (double *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseCcsrilu02(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, cuComplex* csrValA, const int* csrRowPtrA, const int* csrColIndA, csrilu02Info_t info, cusparseSolvePolicy_t policy, void* pBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddVariableForArguments<int>(nnz);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusparseFrontend::AddDevicePointerForArguments(csrValA);
    CusparseFrontend::AddDevicePointerForArguments(csrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(csrColIndA);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) info);
    CusparseFrontend::AddVariableForArguments<cusparseSolvePolicy_t>(policy);
    CusparseFrontend::AddDevicePointerForArguments(pBuffer);
    CusparseFrontend::Execute("cusparseCcsrilu02");
    if (CusparseFrontend::Success()) {
        csrValA = (cuComplex *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseZcsrilu02(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, cuDoubleComplex* csrValA, const int* csrRowPtrA, const int* csrColIndA, csrilu02Info_t info, cusparseSolvePolicy_t policy, void* pBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddVariableForArguments<int>(nnz);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusparseFrontend::AddDevicePointerForArguments(csrValA);
    CusparseFrontend::AddDevicePointerForArguments(csrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(csrColIndA);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) info);
    CusparseFrontend::AddVariableForArguments<cusparseSolvePolicy_t>(policy);
    CusparseFrontend::AddDevicePointerForArguments(pBuffer);
    CusparseFrontend::Execute("cusparseZcsrilu02");
    if (CusparseFrontend::Success()) {
        csrValA = (cuDoubleComplex *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseSbsrilu02_numericBoost(cusparseHandle_t handle, bsrilu02Info_t info, int enable_boost, double* tol, float* boost_val) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) info);
    CusparseFrontend::AddVariableForArguments<int>(enable_boost);
    CusparseFrontend::AddDevicePointerForArguments(tol);
    CusparseFrontend::AddDevicePointerForArguments(boost_val);
    CusparseFrontend::Execute("cusparseSbsrilu02_numericBoost");
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseDbsrilu02_numericBoost(cusparseHandle_t handle, bsrilu02Info_t info, int enable_boost, double* tol, double* boost_val) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) info);
    CusparseFrontend::AddVariableForArguments<int>(enable_boost);
    CusparseFrontend::AddDevicePointerForArguments(tol);
    CusparseFrontend::AddDevicePointerForArguments(boost_val);
    CusparseFrontend::Execute("cusparseDbsrilu02_numericBoost");
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseCbsrilu02_numericBoost(cusparseHandle_t handle, bsrilu02Info_t info, int enable_boost, double* tol, cuComplex* boost_val) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) info);
    CusparseFrontend::AddVariableForArguments<int>(enable_boost);
    CusparseFrontend::AddDevicePointerForArguments(tol);
    CusparseFrontend::AddDevicePointerForArguments(boost_val);
    CusparseFrontend::Execute("cusparseCbsrilu02_numericBoost");
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseZbsrilu02_numericBoost(cusparseHandle_t handle, bsrilu02Info_t info, int enable_boost, double* tol, cuDoubleComplex* boost_val) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) info);
    CusparseFrontend::AddVariableForArguments<int>(enable_boost);
    CusparseFrontend::AddDevicePointerForArguments(tol);
    CusparseFrontend::AddDevicePointerForArguments(boost_val);
    CusparseFrontend::Execute("cusparseZbsrilu02_numericBoost");
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseXcsrilu02_zeroPivot(cusparseHandle_t handle, csrilu02Info_t info, int* position) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) info);
    CusparseFrontend::Execute("cusparseXcsrilu02_zeroPivot");
    if (CusparseFrontend::Success()) {
        position = (int *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseSbsrilu02_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb, const cusparseMatDescr_t descrA, float* bsrValA, const int* bsrRowPtrA, const int* bsrColIndA, int blockDim, bsrilu02Info_t info, int* pBufferSize) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<cusparseDirection_t>(dirA);
    CusparseFrontend::AddVariableForArguments<int>(mb);
    CusparseFrontend::AddVariableForArguments<int>(nnzb);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrValA);
    CusparseFrontend::AddDevicePointerForArguments(bsrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrColIndA);
    CusparseFrontend::AddVariableForArguments<int>(blockDim);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) info);
    CusparseFrontend::Execute("cusparseSbsrilu02_bufferSize");
    if (CusparseFrontend::Success()) {
        info = CusparseFrontend::GetOutputVariable<bsrilu02Info_t>();
        pBufferSize = (int *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseDbsrilu02_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb, const cusparseMatDescr_t descrA, double* bsrValA, const int* bsrRowPtrA, const int* bsrColIndA, int blockDim, bsrilu02Info_t info, int* pBufferSize) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<cusparseDirection_t>(dirA);
    CusparseFrontend::AddVariableForArguments<int>(mb);
    CusparseFrontend::AddVariableForArguments<int>(nnzb);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrValA);
    CusparseFrontend::AddDevicePointerForArguments(bsrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrColIndA);
    CusparseFrontend::AddVariableForArguments<int>(blockDim);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) info);
    CusparseFrontend::Execute("cusparseDbsrilu02_bufferSize");
    if (CusparseFrontend::Success()) {
        info = CusparseFrontend::GetOutputVariable<bsrilu02Info_t>();
        pBufferSize = (int *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseCbsrilu02_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb, const cusparseMatDescr_t descrA, cuComplex* bsrValA, const int* bsrRowPtrA, const int* bsrColIndA, int blockDim, bsrilu02Info_t info, int* pBufferSize) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<cusparseDirection_t>(dirA);
    CusparseFrontend::AddVariableForArguments<int>(mb);
    CusparseFrontend::AddVariableForArguments<int>(nnzb);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrValA);
    CusparseFrontend::AddDevicePointerForArguments(bsrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrColIndA);
    CusparseFrontend::AddVariableForArguments<int>(blockDim);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) info);
    CusparseFrontend::Execute("cusparseCbsrilu02_bufferSize");
    if (CusparseFrontend::Success()) {
        info = CusparseFrontend::GetOutputVariable<bsrilu02Info_t>();
        pBufferSize = (int *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseZbsrilu02_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb, const cusparseMatDescr_t descrA, cuDoubleComplex* bsrValA, const int* bsrRowPtrA, const int* bsrColIndA, int blockDim, bsrilu02Info_t info, int* pBufferSize) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<cusparseDirection_t>(dirA);
    CusparseFrontend::AddVariableForArguments<int>(mb);
    CusparseFrontend::AddVariableForArguments<int>(nnzb);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrValA);
    CusparseFrontend::AddDevicePointerForArguments(bsrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrColIndA);
    CusparseFrontend::AddVariableForArguments<int>(blockDim);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) info);
    CusparseFrontend::Execute("cusparseZbsrilu02_bufferSize");
    if (CusparseFrontend::Success()) {
        info = CusparseFrontend::GetOutputVariable<bsrilu02Info_t>();
        pBufferSize = (int *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseSbsrilu02_analysis(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb, const cusparseMatDescr_t descrA, float* bsrValA, const int* bsrRowPtrA, const int* bsrColIndA, int blockDim, bsrilu02Info_t info, cusparseSolvePolicy_t policy, void* pBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<cusparseDirection_t>(dirA);
    CusparseFrontend::AddVariableForArguments<int>(mb);
    CusparseFrontend::AddVariableForArguments<int>(nnzb);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrValA);
    CusparseFrontend::AddDevicePointerForArguments(bsrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrColIndA);
    CusparseFrontend::AddVariableForArguments<int>(blockDim);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) info);
    CusparseFrontend::AddVariableForArguments<cusparseSolvePolicy_t>(policy);
    CusparseFrontend::AddDevicePointerForArguments(pBuffer);
    CusparseFrontend::Execute("cusparseSbsrilu02_analysis");
    if (CusparseFrontend::Success()) {
        info = CusparseFrontend::GetOutputVariable<bsrilu02Info_t>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseDbsrilu02_analysis(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb, const cusparseMatDescr_t descrA, double* bsrValA, const int* bsrRowPtrA, const int* bsrColIndA, int blockDim, bsrilu02Info_t info, cusparseSolvePolicy_t policy, void* pBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<cusparseDirection_t>(dirA);
    CusparseFrontend::AddVariableForArguments<int>(mb);
    CusparseFrontend::AddVariableForArguments<int>(nnzb);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrValA);
    CusparseFrontend::AddDevicePointerForArguments(bsrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrColIndA);
    CusparseFrontend::AddVariableForArguments<int>(blockDim);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) info);
    CusparseFrontend::AddVariableForArguments<cusparseSolvePolicy_t>(policy);
    CusparseFrontend::AddDevicePointerForArguments(pBuffer);
    CusparseFrontend::Execute("cusparseDbsrilu02_analysis");
    if (CusparseFrontend::Success()) {
        info = CusparseFrontend::GetOutputVariable<bsrilu02Info_t>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseCbsrilu02_analysis(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb, const cusparseMatDescr_t descrA, cuComplex* bsrValA, const int* bsrRowPtrA, const int* bsrColIndA, int blockDim, bsrilu02Info_t info, cusparseSolvePolicy_t policy, void* pBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<cusparseDirection_t>(dirA);
    CusparseFrontend::AddVariableForArguments<int>(mb);
    CusparseFrontend::AddVariableForArguments<int>(nnzb);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrValA);
    CusparseFrontend::AddDevicePointerForArguments(bsrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrColIndA);
    CusparseFrontend::AddVariableForArguments<int>(blockDim);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) info);
    CusparseFrontend::AddVariableForArguments<cusparseSolvePolicy_t>(policy);
    CusparseFrontend::AddDevicePointerForArguments(pBuffer);
    CusparseFrontend::Execute("cusparseCbsrilu02_analysis");
    if (CusparseFrontend::Success()) {
        info = CusparseFrontend::GetOutputVariable<bsrilu02Info_t>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseZbsrilu02_analysis(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb, const cusparseMatDescr_t descrA, cuDoubleComplex* bsrValA, const int* bsrRowPtrA, const int* bsrColIndA, int blockDim, bsrilu02Info_t info, cusparseSolvePolicy_t policy, void* pBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<cusparseDirection_t>(dirA);
    CusparseFrontend::AddVariableForArguments<int>(mb);
    CusparseFrontend::AddVariableForArguments<int>(nnzb);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrValA);
    CusparseFrontend::AddDevicePointerForArguments(bsrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrColIndA);
    CusparseFrontend::AddVariableForArguments<int>(blockDim);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) info);
    CusparseFrontend::AddVariableForArguments<cusparseSolvePolicy_t>(policy);
    CusparseFrontend::AddDevicePointerForArguments(pBuffer);
    CusparseFrontend::Execute("cusparseZbsrilu02_analysis");
    if (CusparseFrontend::Success()) {
        info = CusparseFrontend::GetOutputVariable<bsrilu02Info_t>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseSbsrilu02(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb, const cusparseMatDescr_t descrA, float* bsrValA, const int* bsrRowPtrA, const int* bsrColIndA, int blockDim, bsrilu02Info_t info, cusparseSolvePolicy_t policy, void* pBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<cusparseDirection_t>(dirA);
    CusparseFrontend::AddVariableForArguments<int>(mb);
    CusparseFrontend::AddVariableForArguments<int>(nnzb);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrValA);
    CusparseFrontend::AddDevicePointerForArguments(bsrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrColIndA);
    CusparseFrontend::AddVariableForArguments<int>(blockDim);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) info);
    CusparseFrontend::AddVariableForArguments<cusparseSolvePolicy_t>(policy);
    CusparseFrontend::AddDevicePointerForArguments(pBuffer);
    CusparseFrontend::Execute("cusparseSbsrilu02");
    if (CusparseFrontend::Success()) {
        bsrValA = (float *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseDbsrilu02(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb, const cusparseMatDescr_t descrA, double* bsrValA, const int* bsrRowPtrA, const int* bsrColIndA, int blockDim, bsrilu02Info_t info, cusparseSolvePolicy_t policy, void* pBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<cusparseDirection_t>(dirA);
    CusparseFrontend::AddVariableForArguments<int>(mb);
    CusparseFrontend::AddVariableForArguments<int>(nnzb);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrValA);
    CusparseFrontend::AddDevicePointerForArguments(bsrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrColIndA);
    CusparseFrontend::AddVariableForArguments<int>(blockDim);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) info);
    CusparseFrontend::AddVariableForArguments<cusparseSolvePolicy_t>(policy);
    CusparseFrontend::AddDevicePointerForArguments(pBuffer);
    CusparseFrontend::Execute("cusparseDbsrilu02");
    if (CusparseFrontend::Success()) {
        bsrValA = (double *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseCbsrilu02(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb, const cusparseMatDescr_t descrA, cuComplex* bsrValA, const int* bsrRowPtrA, const int* bsrColIndA, int blockDim, bsrilu02Info_t info, cusparseSolvePolicy_t policy, void* pBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<cusparseDirection_t>(dirA);
    CusparseFrontend::AddVariableForArguments<int>(mb);
    CusparseFrontend::AddVariableForArguments<int>(nnzb);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrValA);
    CusparseFrontend::AddDevicePointerForArguments(bsrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrColIndA);
    CusparseFrontend::AddVariableForArguments<int>(blockDim);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) info);
    CusparseFrontend::AddVariableForArguments<cusparseSolvePolicy_t>(policy);
    CusparseFrontend::AddDevicePointerForArguments(pBuffer);
    CusparseFrontend::Execute("cusparseCbsrilu02");
    if (CusparseFrontend::Success()) {
        bsrValA = (cuComplex *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseZbsrilu02(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb, const cusparseMatDescr_t descrA, cuDoubleComplex* bsrValA, const int* bsrRowPtrA, const int* bsrColIndA, int blockDim, bsrilu02Info_t info, cusparseSolvePolicy_t policy, void* pBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<cusparseDirection_t>(dirA);
    CusparseFrontend::AddVariableForArguments<int>(mb);
    CusparseFrontend::AddVariableForArguments<int>(nnzb);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrValA);
    CusparseFrontend::AddDevicePointerForArguments(bsrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrColIndA);
    CusparseFrontend::AddVariableForArguments<int>(blockDim);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) info);
    CusparseFrontend::AddVariableForArguments<cusparseSolvePolicy_t>(policy);
    CusparseFrontend::AddDevicePointerForArguments(pBuffer);
    CusparseFrontend::Execute("cusparseZbsrilu02");
    if (CusparseFrontend::Success()) {
        bsrValA = (cuDoubleComplex *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseXbsrilu02_zeroPivot(cusparseHandle_t handle, bsrilu02Info_t info, int* position) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) info);
    CusparseFrontend::Execute("cusparseXbsrilu02_zeroPivot");
    if (CusparseFrontend::Success()) {
        position = (int *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseSgtsv2_bufferSizeExt(cusparseHandle_t handle, int m, int n, const float * dl, const float * d, const float * du, const float * B, int ldb, size_t * pBufferSize) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddVariableForArguments<int>(n);
    CusparseFrontend::AddDevicePointerForArguments(dl);
    CusparseFrontend::AddDevicePointerForArguments(d);
    CusparseFrontend::AddDevicePointerForArguments(du);
    CusparseFrontend::AddDevicePointerForArguments(B);
    CusparseFrontend::AddVariableForArguments<int>(ldb);
    CusparseFrontend::AddHostPointerForArguments<size_t>(pBufferSize);
    CusparseFrontend::Execute("cusparseSgtsv2_bufferSizeExt");
    if (CusparseFrontend::Success()) {
        *pBufferSize = *CusparseFrontend::GetOutputHostPointer<size_t>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseDgtsv2_bufferSizeExt(cusparseHandle_t handle, int m, int n, const double * dl, const double * d, const double * du, const double * B, int ldb, size_t * pBufferSize) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddVariableForArguments<int>(n);
    CusparseFrontend::AddDevicePointerForArguments(dl);
    CusparseFrontend::AddDevicePointerForArguments(d);
    CusparseFrontend::AddDevicePointerForArguments(du);
    CusparseFrontend::AddDevicePointerForArguments(B);
    CusparseFrontend::AddVariableForArguments<int>(ldb);
    CusparseFrontend::AddHostPointerForArguments<size_t>(pBufferSize);
    CusparseFrontend::Execute("cusparseDgtsv2_bufferSizeExt");
    if (CusparseFrontend::Success()) {
        *pBufferSize = *CusparseFrontend::GetOutputHostPointer<size_t>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseCgtsv2_bufferSizeExt(cusparseHandle_t handle, int m, int n, const cuComplex * dl, const cuComplex * d, const cuComplex * du, const cuComplex * B, int ldb, size_t * pBufferSize) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddVariableForArguments<int>(n);
    CusparseFrontend::AddDevicePointerForArguments(dl);
    CusparseFrontend::AddDevicePointerForArguments(d);
    CusparseFrontend::AddDevicePointerForArguments(du);
    CusparseFrontend::AddDevicePointerForArguments(B);
    CusparseFrontend::AddVariableForArguments<int>(ldb);
    CusparseFrontend::AddHostPointerForArguments<size_t>(pBufferSize);
    CusparseFrontend::Execute("cusparseCgtsv2_bufferSizeExt");
    if (CusparseFrontend::Success()) {
        *pBufferSize = *CusparseFrontend::GetOutputHostPointer<size_t>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseZgtsv2_bufferSizeExt(cusparseHandle_t handle, int m, int n, const cuDoubleComplex * dl, const cuDoubleComplex * d, const cuDoubleComplex * du, const cuDoubleComplex * B, int ldb, size_t * pBufferSize) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddVariableForArguments<int>(n);
    CusparseFrontend::AddDevicePointerForArguments(dl);
    CusparseFrontend::AddDevicePointerForArguments(d);
    CusparseFrontend::AddDevicePointerForArguments(du);
    CusparseFrontend::AddDevicePointerForArguments(B);
    CusparseFrontend::AddVariableForArguments<int>(ldb);
    CusparseFrontend::AddHostPointerForArguments<size_t>(pBufferSize);
    CusparseFrontend::Execute("cusparseZgtsv2_bufferSizeExt");
    if (CusparseFrontend::Success()) {
        *pBufferSize = *CusparseFrontend::GetOutputHostPointer<size_t>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseSgtsv2(cusparseHandle_t handle, int m, int n, const float * dl, const float * d, const float * du, float * B, int ldb, void * pBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddVariableForArguments<int>(n);
    CusparseFrontend::AddDevicePointerForArguments(dl);
    CusparseFrontend::AddDevicePointerForArguments(d);
    CusparseFrontend::AddDevicePointerForArguments(du);
    CusparseFrontend::AddDevicePointerForArguments(B);
    CusparseFrontend::AddVariableForArguments<int>(ldb);
    CusparseFrontend::AddDevicePointerForArguments(pBuffer);
    CusparseFrontend::Execute("cusparseSgtsv2");
    if (CusparseFrontend::Success()) {
        B = (float *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseDgtsv2(cusparseHandle_t handle, int m, int n, const double * dl, const double * d, const double * du, double * B, int ldb, void * pBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddVariableForArguments<int>(n);
    CusparseFrontend::AddDevicePointerForArguments(dl);
    CusparseFrontend::AddDevicePointerForArguments(d);
    CusparseFrontend::AddDevicePointerForArguments(du);
    CusparseFrontend::AddDevicePointerForArguments(B);
    CusparseFrontend::AddVariableForArguments<int>(ldb);
    CusparseFrontend::AddDevicePointerForArguments(pBuffer);
    CusparseFrontend::Execute("cusparseDgtsv2");
    if (CusparseFrontend::Success()) {
        B = (double *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseCgtsv2(cusparseHandle_t handle, int m, int n, const cuComplex * dl, const cuComplex * d, const cuComplex * du, cuComplex * B, int ldb, void * pBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddVariableForArguments<int>(n);
    CusparseFrontend::AddDevicePointerForArguments(dl);
    CusparseFrontend::AddDevicePointerForArguments(d);
    CusparseFrontend::AddDevicePointerForArguments(du);
    CusparseFrontend::AddDevicePointerForArguments(B);
    CusparseFrontend::AddVariableForArguments<int>(ldb);
    CusparseFrontend::AddDevicePointerForArguments(pBuffer);
    CusparseFrontend::Execute("cusparseCgtsv2");
    if (CusparseFrontend::Success()) {
        B = (cuComplex *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseZgtsv2(cusparseHandle_t handle, int m, int n, const cuDoubleComplex * dl, const cuDoubleComplex * d, const cuDoubleComplex * du, cuDoubleComplex * B, int ldb, void * pBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddVariableForArguments<int>(n);
    CusparseFrontend::AddDevicePointerForArguments(dl);
    CusparseFrontend::AddDevicePointerForArguments(d);
    CusparseFrontend::AddDevicePointerForArguments(du);
    CusparseFrontend::AddDevicePointerForArguments(B);
    CusparseFrontend::AddVariableForArguments<int>(ldb);
    CusparseFrontend::AddDevicePointerForArguments(pBuffer);
    CusparseFrontend::Execute("cusparseZgtsv2");
    if (CusparseFrontend::Success()) {
        B = (cuDoubleComplex *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseSgtsv2_nopivot_bufferSizeExt(cusparseHandle_t handle, int m, int n, const float * dl, const float * d, const float * du, const float * B, int ldb, size_t * pBufferSize) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddVariableForArguments<int>(n);
    CusparseFrontend::AddDevicePointerForArguments(dl);
    CusparseFrontend::AddDevicePointerForArguments(d);
    CusparseFrontend::AddDevicePointerForArguments(du);
    CusparseFrontend::AddDevicePointerForArguments(B);
    CusparseFrontend::AddVariableForArguments<int>(ldb);
    CusparseFrontend::AddHostPointerForArguments<size_t>(pBufferSize);
    CusparseFrontend::Execute("cusparseSgtsv2_nopivot_bufferSizeExt");
    if (CusparseFrontend::Success()) {
        *pBufferSize = *CusparseFrontend::GetOutputHostPointer<size_t>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseDgtsv2_nopivot_bufferSizeExt(cusparseHandle_t handle, int m, int n, const double * dl, const double * d, const double * du, const double * B, int ldb, size_t * pBufferSize) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddVariableForArguments<int>(n);
    CusparseFrontend::AddDevicePointerForArguments(dl);
    CusparseFrontend::AddDevicePointerForArguments(d);
    CusparseFrontend::AddDevicePointerForArguments(du);
    CusparseFrontend::AddDevicePointerForArguments(B);
    CusparseFrontend::AddVariableForArguments<int>(ldb);
    CusparseFrontend::AddHostPointerForArguments<size_t>(pBufferSize);
    CusparseFrontend::Execute("cusparseDgtsv2_nopivot_bufferSizeExt");
    if (CusparseFrontend::Success()) {
        *pBufferSize = *CusparseFrontend::GetOutputHostPointer<size_t>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseCgtsv2_nopivot_bufferSizeExt(cusparseHandle_t handle, int m, int n, const cuComplex * dl, const cuComplex * d, const cuComplex * du, const cuComplex * B, int ldb, size_t * pBufferSize) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddVariableForArguments<int>(n);
    CusparseFrontend::AddDevicePointerForArguments(dl);
    CusparseFrontend::AddDevicePointerForArguments(d);
    CusparseFrontend::AddDevicePointerForArguments(du);
    CusparseFrontend::AddDevicePointerForArguments(B);
    CusparseFrontend::AddVariableForArguments<int>(ldb);
    CusparseFrontend::AddHostPointerForArguments<size_t>(pBufferSize);
    CusparseFrontend::Execute("cusparseCgtsv2_nopivot_bufferSizeExt");
    if (CusparseFrontend::Success()) {
        *pBufferSize = *CusparseFrontend::GetOutputHostPointer<size_t>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseZgtsv2_nopivot_bufferSizeExt(cusparseHandle_t handle, int m, int n, const cuDoubleComplex * dl, const cuDoubleComplex * d, const cuDoubleComplex * du, const cuDoubleComplex * B, int ldb, size_t * pBufferSize) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddVariableForArguments<int>(n);
    CusparseFrontend::AddDevicePointerForArguments(dl);
    CusparseFrontend::AddDevicePointerForArguments(d);
    CusparseFrontend::AddDevicePointerForArguments(du);
    CusparseFrontend::AddDevicePointerForArguments(B);
    CusparseFrontend::AddVariableForArguments<int>(ldb);
    CusparseFrontend::AddHostPointerForArguments<size_t>(pBufferSize);
    CusparseFrontend::Execute("cusparseZgtsv2_nopivot_bufferSizeExt");
    if (CusparseFrontend::Success()) {
        *pBufferSize = *CusparseFrontend::GetOutputHostPointer<size_t>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseSgtsv2_nopivot(cusparseHandle_t handle, int m, int n, const float * dl, const float * d, const float * du, float * B, int ldb, void * pBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddVariableForArguments<int>(n);
    CusparseFrontend::AddDevicePointerForArguments(dl);
    CusparseFrontend::AddDevicePointerForArguments(d);
    CusparseFrontend::AddDevicePointerForArguments(du);
    CusparseFrontend::AddDevicePointerForArguments(B);
    CusparseFrontend::AddVariableForArguments<int>(ldb);
    CusparseFrontend::AddDevicePointerForArguments(pBuffer);
    CusparseFrontend::Execute("cusparseSgtsv2_nopivot");
    if (CusparseFrontend::Success()) {
        B = (float *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseDgtsv2_nopivot(cusparseHandle_t handle, int m, int n, const double * dl, const double * d, const double * du, double * B, int ldb, void * pBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddVariableForArguments<int>(n);
    CusparseFrontend::AddDevicePointerForArguments(dl);
    CusparseFrontend::AddDevicePointerForArguments(d);
    CusparseFrontend::AddDevicePointerForArguments(du);
    CusparseFrontend::AddDevicePointerForArguments(B);
    CusparseFrontend::AddVariableForArguments<int>(ldb);
    CusparseFrontend::AddDevicePointerForArguments(pBuffer);
    CusparseFrontend::Execute("cusparseDgtsv2_nopivot");
    if (CusparseFrontend::Success()) {
        B = (double *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseCgtsv2_nopivot(cusparseHandle_t handle, int m, int n, const cuComplex * dl, const cuComplex * d, const cuComplex * du, cuComplex * B, int ldb, void * pBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddVariableForArguments<int>(n);
    CusparseFrontend::AddDevicePointerForArguments(dl);
    CusparseFrontend::AddDevicePointerForArguments(d);
    CusparseFrontend::AddDevicePointerForArguments(du);
    CusparseFrontend::AddDevicePointerForArguments(B);
    CusparseFrontend::AddVariableForArguments<int>(ldb);
    CusparseFrontend::AddDevicePointerForArguments(pBuffer);
    CusparseFrontend::Execute("cusparseCgtsv2_nopivot");
    if (CusparseFrontend::Success()) {
        B = (cuComplex *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseZgtsv2_nopivot(cusparseHandle_t handle, int m, int n, const cuDoubleComplex * dl, const cuDoubleComplex * d, const cuDoubleComplex * du, cuDoubleComplex * B, int ldb, void * pBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddVariableForArguments<int>(n);
    CusparseFrontend::AddDevicePointerForArguments(dl);
    CusparseFrontend::AddDevicePointerForArguments(d);
    CusparseFrontend::AddDevicePointerForArguments(du);
    CusparseFrontend::AddDevicePointerForArguments(B);
    CusparseFrontend::AddVariableForArguments<int>(ldb);
    CusparseFrontend::AddDevicePointerForArguments(pBuffer);
    CusparseFrontend::Execute("cusparseZgtsv2_nopivot");
    if (CusparseFrontend::Success()) {
        B = (cuDoubleComplex *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseSgtsv2StridedBatch_bufferSizeExt(cusparseHandle_t handle, int m, const float * dl, const float * d, const float * du, const float * x, int batchCount, int batchStride, size_t * pBufferSize) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddDevicePointerForArguments(dl);
    CusparseFrontend::AddDevicePointerForArguments(d);
    CusparseFrontend::AddDevicePointerForArguments(du);
    CusparseFrontend::AddDevicePointerForArguments(x);
    CusparseFrontend::AddVariableForArguments<int>(batchCount);
    CusparseFrontend::AddVariableForArguments<int>(batchStride);
    CusparseFrontend::AddHostPointerForArguments<size_t>(pBufferSize);
    CusparseFrontend::Execute("cusparseSgtsv2StridedBatch_bufferSizeExt");
    if (CusparseFrontend::Success()) {
        *pBufferSize = *CusparseFrontend::GetOutputHostPointer<size_t>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseDgtsv2StridedBatch_bufferSizeExt(cusparseHandle_t handle, int m, const double * dl, const double * d, const double * du, const double * x, int batchCount, int batchStride, size_t * pBufferSize) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddDevicePointerForArguments(dl);
    CusparseFrontend::AddDevicePointerForArguments(d);
    CusparseFrontend::AddDevicePointerForArguments(du);
    CusparseFrontend::AddDevicePointerForArguments(x);
    CusparseFrontend::AddVariableForArguments<int>(batchCount);
    CusparseFrontend::AddVariableForArguments<int>(batchStride);
    CusparseFrontend::AddHostPointerForArguments<size_t>(pBufferSize);
    CusparseFrontend::Execute("cusparseDgtsv2StridedBatch_bufferSizeExt");
    if (CusparseFrontend::Success()) {
        *pBufferSize = *CusparseFrontend::GetOutputHostPointer<size_t>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseCgtsv2StridedBatch_bufferSizeExt(cusparseHandle_t handle, int m, const cuComplex * dl, const cuComplex * d, const cuComplex * du, const cuComplex * x, int batchCount, int batchStride, size_t * pBufferSize) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddDevicePointerForArguments(dl);
    CusparseFrontend::AddDevicePointerForArguments(d);
    CusparseFrontend::AddDevicePointerForArguments(du);
    CusparseFrontend::AddDevicePointerForArguments(x);
    CusparseFrontend::AddVariableForArguments<int>(batchCount);
    CusparseFrontend::AddVariableForArguments<int>(batchStride);
    CusparseFrontend::AddHostPointerForArguments<size_t>(pBufferSize);
    CusparseFrontend::Execute("cusparseCgtsv2StridedBatch_bufferSizeExt");
    if (CusparseFrontend::Success()) {
        *pBufferSize = *CusparseFrontend::GetOutputHostPointer<size_t>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseZgtsv2StridedBatch_bufferSizeExt(cusparseHandle_t handle, int m, const cuDoubleComplex * dl, const cuDoubleComplex * d, const cuDoubleComplex * du, const cuDoubleComplex * x, int batchCount, int batchStride, size_t * pBufferSize) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddDevicePointerForArguments(dl);
    CusparseFrontend::AddDevicePointerForArguments(d);
    CusparseFrontend::AddDevicePointerForArguments(du);
    CusparseFrontend::AddDevicePointerForArguments(x);
    CusparseFrontend::AddVariableForArguments<int>(batchCount);
    CusparseFrontend::AddVariableForArguments<int>(batchStride);
    CusparseFrontend::AddHostPointerForArguments<size_t>(pBufferSize);
    CusparseFrontend::Execute("cusparseZgtsv2StridedBatch_bufferSizeExt");
    if (CusparseFrontend::Success()) {
        *pBufferSize = *CusparseFrontend::GetOutputHostPointer<size_t>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseSgtsv2StridedBatch(cusparseHandle_t handle, int m, const float * dl, const float * d, const float * du, float * x, int batchCount, int batchStride, void * pBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddDevicePointerForArguments(dl);
    CusparseFrontend::AddDevicePointerForArguments(d);
    CusparseFrontend::AddDevicePointerForArguments(du);
    CusparseFrontend::AddDevicePointerForArguments(x);
    CusparseFrontend::AddVariableForArguments<int>(batchCount);
    CusparseFrontend::AddVariableForArguments<int>(batchStride);
    CusparseFrontend::AddDevicePointerForArguments(pBuffer);
    CusparseFrontend::Execute("cusparseSgtsv2StridedBatch");
    if (CusparseFrontend::Success()) {
        x = (float *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseDgtsv2StridedBatch(cusparseHandle_t handle, int m, const double * dl, const double * d, const double * du, double * x, int batchCount, int batchStride, void * pBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddDevicePointerForArguments(dl);
    CusparseFrontend::AddDevicePointerForArguments(d);
    CusparseFrontend::AddDevicePointerForArguments(du);
    CusparseFrontend::AddDevicePointerForArguments(x);
    CusparseFrontend::AddVariableForArguments<int>(batchCount);
    CusparseFrontend::AddVariableForArguments<int>(batchStride);
    CusparseFrontend::AddDevicePointerForArguments(pBuffer);
    CusparseFrontend::Execute("cusparseDgtsv2StridedBatch");
    if (CusparseFrontend::Success()) {
        x = (double *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseCgtsv2StridedBatch(cusparseHandle_t handle, int m, const cuComplex * dl, const cuComplex * d, const cuComplex * du, cuComplex * x, int batchCount, int batchStride, void * pBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddDevicePointerForArguments(dl);
    CusparseFrontend::AddDevicePointerForArguments(d);
    CusparseFrontend::AddDevicePointerForArguments(du);
    CusparseFrontend::AddDevicePointerForArguments(x);
    CusparseFrontend::AddVariableForArguments<int>(batchCount);
    CusparseFrontend::AddVariableForArguments<int>(batchStride);
    CusparseFrontend::AddDevicePointerForArguments(pBuffer);
    CusparseFrontend::Execute("cusparseCgtsv2StridedBatch");
    if (CusparseFrontend::Success()) {
        x = (cuComplex *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseZgtsv2StridedBatch(cusparseHandle_t handle, int m, const cuDoubleComplex * dl, const cuDoubleComplex * d, const cuDoubleComplex * du, cuDoubleComplex * x, int batchCount, int batchStride, void * pBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddDevicePointerForArguments(dl);
    CusparseFrontend::AddDevicePointerForArguments(d);
    CusparseFrontend::AddDevicePointerForArguments(du);
    CusparseFrontend::AddDevicePointerForArguments(x);
    CusparseFrontend::AddVariableForArguments<int>(batchCount);
    CusparseFrontend::AddVariableForArguments<int>(batchStride);
    CusparseFrontend::AddDevicePointerForArguments(pBuffer);
    CusparseFrontend::Execute("cusparseZgtsv2StridedBatch");
    if (CusparseFrontend::Success()) {
        x = (cuDoubleComplex *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseSgtsvInterleavedBatch_bufferSizeExt(cusparseHandle_t handle, int algo, int m, const float * dl, const float * d, const float * du, const float * x, int batchCount, size_t * pBufferSize) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<int>(algo);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddDevicePointerForArguments(dl);
    CusparseFrontend::AddDevicePointerForArguments(d);
    CusparseFrontend::AddDevicePointerForArguments(du);
    CusparseFrontend::AddDevicePointerForArguments(x);
    CusparseFrontend::AddVariableForArguments<int>(batchCount);
    CusparseFrontend::AddHostPointerForArguments<size_t>(pBufferSize);
    CusparseFrontend::Execute("cusparseSgtsvInterleavedBatch_bufferSizeExt");
    if (CusparseFrontend::Success()) {
        *pBufferSize = *CusparseFrontend::GetOutputHostPointer<size_t>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseDgtsvInterleavedBatch_bufferSizeExt(cusparseHandle_t handle, int algo, int m, const double * dl, const double * d, const double * du, const double * x, int batchCount, size_t * pBufferSize) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<int>(algo);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddDevicePointerForArguments(dl);
    CusparseFrontend::AddDevicePointerForArguments(d);
    CusparseFrontend::AddDevicePointerForArguments(du);
    CusparseFrontend::AddDevicePointerForArguments(x);
    CusparseFrontend::AddVariableForArguments<int>(batchCount);
    CusparseFrontend::AddHostPointerForArguments<size_t>(pBufferSize);
    CusparseFrontend::Execute("cusparseDgtsvInterleavedBatch_bufferSizeExt");
    if (CusparseFrontend::Success()) {
        *pBufferSize = *CusparseFrontend::GetOutputHostPointer<size_t>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseCgtsvInterleavedBatch_bufferSizeExt(cusparseHandle_t handle, int algo, int m, const cuComplex * dl, const cuComplex * d, const cuComplex * du, const cuComplex * x, int batchCount, size_t * pBufferSize) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<int>(algo);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddDevicePointerForArguments(dl);
    CusparseFrontend::AddDevicePointerForArguments(d);
    CusparseFrontend::AddDevicePointerForArguments(du);
    CusparseFrontend::AddDevicePointerForArguments(x);
    CusparseFrontend::AddVariableForArguments<int>(batchCount);
    CusparseFrontend::AddHostPointerForArguments<size_t>(pBufferSize);
    CusparseFrontend::Execute("cusparseCgtsvInterleavedBatch_bufferSizeExt");
    if (CusparseFrontend::Success()) {
        *pBufferSize = *CusparseFrontend::GetOutputHostPointer<size_t>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseZgtsvInterleavedBatch_bufferSizeExt(cusparseHandle_t handle, int algo, int m, const cuDoubleComplex * dl, const cuDoubleComplex * d, const cuDoubleComplex * du, const cuDoubleComplex * x, int batchCount, size_t * pBufferSize) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<int>(algo);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddDevicePointerForArguments(dl);
    CusparseFrontend::AddDevicePointerForArguments(d);
    CusparseFrontend::AddDevicePointerForArguments(du);
    CusparseFrontend::AddDevicePointerForArguments(x);
    CusparseFrontend::AddVariableForArguments<int>(batchCount);
    CusparseFrontend::AddHostPointerForArguments<size_t>(pBufferSize);
    CusparseFrontend::Execute("cusparseZgtsvInterleavedBatch_bufferSizeExt");
    if (CusparseFrontend::Success()) {
        *pBufferSize = *CusparseFrontend::GetOutputHostPointer<size_t>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseSgtsvInterleavedBatch(cusparseHandle_t handle, int algo, int m, float * dl, float * d, float * du, float * x, int batchCount, void * pBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<int>(algo);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddDevicePointerForArguments(dl);
    CusparseFrontend::AddDevicePointerForArguments(d);
    CusparseFrontend::AddDevicePointerForArguments(du);
    CusparseFrontend::AddDevicePointerForArguments(x);
    CusparseFrontend::AddVariableForArguments<int>(batchCount);
    CusparseFrontend::AddDevicePointerForArguments(pBuffer);
    CusparseFrontend::Execute("cusparseSgtsvInterleavedBatch");
    if (CusparseFrontend::Success()) {
        x = (float *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseDgtsvInterleavedBatch(cusparseHandle_t handle, int algo, int m, double * dl, double * d, double * du, double * x, int batchCount, void * pBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<int>(algo);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddDevicePointerForArguments(dl);
    CusparseFrontend::AddDevicePointerForArguments(d);
    CusparseFrontend::AddDevicePointerForArguments(du);
    CusparseFrontend::AddDevicePointerForArguments(x);
    CusparseFrontend::AddVariableForArguments<int>(batchCount);
    CusparseFrontend::AddDevicePointerForArguments(pBuffer);
    CusparseFrontend::Execute("cusparseDgtsvInterleavedBatch");
    if (CusparseFrontend::Success()) {
        x = (double *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseCgtsvInterleavedBatch(cusparseHandle_t handle, int algo, int m, cuComplex * dl, cuComplex * d, cuComplex * du, cuComplex * x, int batchCount, void * pBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<int>(algo);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddDevicePointerForArguments(dl);
    CusparseFrontend::AddDevicePointerForArguments(d);
    CusparseFrontend::AddDevicePointerForArguments(du);
    CusparseFrontend::AddDevicePointerForArguments(x);
    CusparseFrontend::AddVariableForArguments<int>(batchCount);
    CusparseFrontend::AddDevicePointerForArguments(pBuffer);
    CusparseFrontend::Execute("cusparseCgtsvInterleavedBatch");
    if (CusparseFrontend::Success()) {
        x = (cuComplex *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseZgtsvInterleavedBatch(cusparseHandle_t handle, int algo, int m, cuDoubleComplex * dl, cuDoubleComplex * d, cuDoubleComplex * du, cuDoubleComplex * x, int batchCount, void * pBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<int>(algo);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddDevicePointerForArguments(dl);
    CusparseFrontend::AddDevicePointerForArguments(d);
    CusparseFrontend::AddDevicePointerForArguments(du);
    CusparseFrontend::AddDevicePointerForArguments(x);
    CusparseFrontend::AddVariableForArguments<int>(batchCount);
    CusparseFrontend::AddDevicePointerForArguments(pBuffer);
    CusparseFrontend::Execute("cusparseZgtsvInterleavedBatch");
    if (CusparseFrontend::Success()) {
        x = (cuDoubleComplex *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseSgpsvInterleavedBatch_bufferSizeExt(cusparseHandle_t handle, int algo, int m, const float * ds, const float * dl, const float * d, const float * du, const float * dw, const float * x, int batchCount, size_t * pBufferSize) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<int>(algo);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddDevicePointerForArguments(ds);
    CusparseFrontend::AddDevicePointerForArguments(dl);
    CusparseFrontend::AddDevicePointerForArguments(d);
    CusparseFrontend::AddDevicePointerForArguments(du);
    CusparseFrontend::AddDevicePointerForArguments(dw);
    CusparseFrontend::AddDevicePointerForArguments(x);
    CusparseFrontend::AddVariableForArguments<int>(batchCount);
    CusparseFrontend::AddHostPointerForArguments<size_t>(pBufferSize);
    CusparseFrontend::Execute("cusparseSgpsvInterleavedBatch_bufferSizeExt");
    if (CusparseFrontend::Success()) {
        *pBufferSize = *CusparseFrontend::GetOutputHostPointer<size_t>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseDgpsvInterleavedBatch_bufferSizeExt(cusparseHandle_t handle, int algo, int m, const double * ds, const double * dl, const double * d, const double * du, const double * dw, const double * x, int batchCount, size_t * pBufferSize) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<int>(algo);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddDevicePointerForArguments(ds);
    CusparseFrontend::AddDevicePointerForArguments(dl);
    CusparseFrontend::AddDevicePointerForArguments(d);
    CusparseFrontend::AddDevicePointerForArguments(du);
    CusparseFrontend::AddDevicePointerForArguments(dw);
    CusparseFrontend::AddDevicePointerForArguments(x);
    CusparseFrontend::AddVariableForArguments<int>(batchCount);
    CusparseFrontend::AddHostPointerForArguments<size_t>(pBufferSize);
    CusparseFrontend::Execute("cusparseDgpsvInterleavedBatch_bufferSizeExt");
    if (CusparseFrontend::Success()) {
        *pBufferSize = *CusparseFrontend::GetOutputHostPointer<size_t>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseCgpsvInterleavedBatch_bufferSizeExt(cusparseHandle_t handle, int algo, int m, const cuComplex * ds, const cuComplex * dl, const cuComplex * d, const cuComplex * du, const cuComplex * dw, const cuComplex * x, int batchCount, size_t * pBufferSize) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<int>(algo);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddDevicePointerForArguments(ds);
    CusparseFrontend::AddDevicePointerForArguments(dl);
    CusparseFrontend::AddDevicePointerForArguments(d);
    CusparseFrontend::AddDevicePointerForArguments(du);
    CusparseFrontend::AddDevicePointerForArguments(dw);
    CusparseFrontend::AddDevicePointerForArguments(x);
    CusparseFrontend::AddVariableForArguments<int>(batchCount);
    CusparseFrontend::AddHostPointerForArguments<size_t>(pBufferSize);
    CusparseFrontend::Execute("cusparseCgpsvInterleavedBatch_bufferSizeExt");
    if (CusparseFrontend::Success()) {
        *pBufferSize = *CusparseFrontend::GetOutputHostPointer<size_t>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseZgpsvInterleavedBatch_bufferSizeExt(cusparseHandle_t handle, int algo, int m, const cuDoubleComplex * ds, const cuDoubleComplex * dl, const cuDoubleComplex * d, const cuDoubleComplex * du, const cuDoubleComplex * dw, const cuDoubleComplex * x, int batchCount, size_t * pBufferSize) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<int>(algo);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddDevicePointerForArguments(ds);
    CusparseFrontend::AddDevicePointerForArguments(dl);
    CusparseFrontend::AddDevicePointerForArguments(d);
    CusparseFrontend::AddDevicePointerForArguments(du);
    CusparseFrontend::AddDevicePointerForArguments(dw);
    CusparseFrontend::AddDevicePointerForArguments(x);
    CusparseFrontend::AddVariableForArguments<int>(batchCount);
    CusparseFrontend::AddHostPointerForArguments<size_t>(pBufferSize);
    CusparseFrontend::Execute("cusparseZgpsvInterleavedBatch_bufferSizeExt");
    if (CusparseFrontend::Success()) {
        *pBufferSize = *CusparseFrontend::GetOutputHostPointer<size_t>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseSgpsvInterleavedBatch(cusparseHandle_t handle, int algo, int m, float * ds, float * dl, float * d, float * du, float * dw, float * x, int batchCount, void * pBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<int>(algo);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddDevicePointerForArguments(ds);
    CusparseFrontend::AddDevicePointerForArguments(dl);
    CusparseFrontend::AddDevicePointerForArguments(d);
    CusparseFrontend::AddDevicePointerForArguments(du);
    CusparseFrontend::AddDevicePointerForArguments(dw);
    CusparseFrontend::AddDevicePointerForArguments(x);
    CusparseFrontend::AddVariableForArguments<int>(batchCount);
    CusparseFrontend::AddDevicePointerForArguments(pBuffer);
    CusparseFrontend::Execute("cusparseSgpsvInterleavedBatch");
    if (CusparseFrontend::Success()) {
        x = (float *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseDgpsvInterleavedBatch(cusparseHandle_t handle, int algo, int m, double * ds, double * dl, double * d, double * du, double * dw, double * x, int batchCount, void * pBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<int>(algo);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddDevicePointerForArguments(ds);
    CusparseFrontend::AddDevicePointerForArguments(dl);
    CusparseFrontend::AddDevicePointerForArguments(d);
    CusparseFrontend::AddDevicePointerForArguments(du);
    CusparseFrontend::AddDevicePointerForArguments(dw);
    CusparseFrontend::AddDevicePointerForArguments(x);
    CusparseFrontend::AddVariableForArguments<int>(batchCount);
    CusparseFrontend::AddDevicePointerForArguments(pBuffer);
    CusparseFrontend::Execute("cusparseDgpsvInterleavedBatch");
    if (CusparseFrontend::Success()) {
        x = (double *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseCgpsvInterleavedBatch(cusparseHandle_t handle, int algo, int m, cuComplex * ds, cuComplex * dl, cuComplex * d, cuComplex * du, cuComplex * dw, cuComplex * x, int batchCount, void * pBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<int>(algo);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddDevicePointerForArguments(ds);
    CusparseFrontend::AddDevicePointerForArguments(dl);
    CusparseFrontend::AddDevicePointerForArguments(d);
    CusparseFrontend::AddDevicePointerForArguments(du);
    CusparseFrontend::AddDevicePointerForArguments(dw);
    CusparseFrontend::AddDevicePointerForArguments(x);
    CusparseFrontend::AddVariableForArguments<int>(batchCount);
    CusparseFrontend::AddDevicePointerForArguments(pBuffer);
    CusparseFrontend::Execute("cusparseCgpsvInterleavedBatch");
    if (CusparseFrontend::Success()) {
        x = (cuComplex *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseZgpsvInterleavedBatch(cusparseHandle_t handle, int algo, int m, cuDoubleComplex * ds, cuDoubleComplex * dl, cuDoubleComplex * d, cuDoubleComplex * du, cuDoubleComplex * dw, cuDoubleComplex * x, int batchCount, void * pBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<int>(algo);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddDevicePointerForArguments(ds);
    CusparseFrontend::AddDevicePointerForArguments(dl);
    CusparseFrontend::AddDevicePointerForArguments(d);
    CusparseFrontend::AddDevicePointerForArguments(du);
    CusparseFrontend::AddDevicePointerForArguments(dw);
    CusparseFrontend::AddDevicePointerForArguments(x);
    CusparseFrontend::AddVariableForArguments<int>(batchCount);
    CusparseFrontend::AddDevicePointerForArguments(pBuffer);
    CusparseFrontend::Execute("cusparseZgpsvInterleavedBatch");
    if (CusparseFrontend::Success()) {
        x = (cuDoubleComplex *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}