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
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<cusparseDirection_t>(dir);
    CusparseFrontend::AddVariableForArguments<int>(mb);
    CusparseFrontend::AddVariableForArguments<int>(nb);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrValA);
    CusparseFrontend::AddDevicePointerForArguments(bsrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrColIndA);
    CusparseFrontend::AddVariableForArguments<int>(blockDim);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrC);
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
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<cusparseDirection_t>(dir);
    CusparseFrontend::AddVariableForArguments<int>(mb);
    CusparseFrontend::AddVariableForArguments<int>(nb);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrValA);
    CusparseFrontend::AddDevicePointerForArguments(bsrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrColIndA);
    CusparseFrontend::AddVariableForArguments<int>(blockDim);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrC);
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
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<cusparseDirection_t>(dir);
    CusparseFrontend::AddVariableForArguments<int>(mb);
    CusparseFrontend::AddVariableForArguments<int>(nb);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrValA);
    CusparseFrontend::AddDevicePointerForArguments(bsrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrColIndA);
    CusparseFrontend::AddVariableForArguments<int>(blockDim);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrC);
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
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<cusparseDirection_t>(dir);
    CusparseFrontend::AddVariableForArguments<int>(mb);
    CusparseFrontend::AddVariableForArguments<int>(nb);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrValA);
    CusparseFrontend::AddDevicePointerForArguments(bsrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrColIndA);
    CusparseFrontend::AddVariableForArguments<int>(blockDim);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrC);
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

extern "C" cusparseStatus_t CUSPARSEAPI cusparseSgebsr2gebsc_bufferSize(cusparseHandle_t handle, int mb, int nb, int nnzb, const float* bsrValA, const int* bsrRowPtrA, const int* bsrColIndA, int rowBlockDim, int colBlockDim, int* pBufferSize) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<int>(mb);
    CusparseFrontend::AddVariableForArguments<int>(nb);
    CusparseFrontend::AddVariableForArguments<int>(nnzb);
    CusparseFrontend::AddDevicePointerForArguments(bsrValA);
    CusparseFrontend::AddDevicePointerForArguments(bsrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrColIndA);
    CusparseFrontend::AddVariableForArguments<int>(rowBlockDim);
    CusparseFrontend::AddVariableForArguments<int>(colBlockDim);
    CusparseFrontend::Execute("cusparseSgebsr2gebsc_bufferSize");
    if (CusparseFrontend::Success()) {
        pBufferSize = (int *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseDgebsr2gebsc_bufferSize(cusparseHandle_t handle, int mb, int nb, int nnzb, const double* bsrValA, const int* bsrRowPtrA, const int* bsrColIndA, int rowBlockDim, int colBlockDim, int* pBufferSize) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<int>(mb);
    CusparseFrontend::AddVariableForArguments<int>(nb);
    CusparseFrontend::AddVariableForArguments<int>(nnzb);
    CusparseFrontend::AddDevicePointerForArguments(bsrValA);
    CusparseFrontend::AddDevicePointerForArguments(bsrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrColIndA);
    CusparseFrontend::AddVariableForArguments<int>(rowBlockDim);
    CusparseFrontend::AddVariableForArguments<int>(colBlockDim);
    CusparseFrontend::Execute("cusparseDgebsr2gebsc_bufferSize");
    if (CusparseFrontend::Success()) {
        pBufferSize = (int *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseCgebsr2gebsc_bufferSize(cusparseHandle_t handle, int mb, int nb, int nnzb, const cuComplex* bsrValA, const int* bsrRowPtrA, const int* bsrColIndA, int rowBlockDim, int colBlockDim, int* pBufferSize) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<int>(mb);
    CusparseFrontend::AddVariableForArguments<int>(nb);
    CusparseFrontend::AddVariableForArguments<int>(nnzb);
    CusparseFrontend::AddDevicePointerForArguments(bsrValA);
    CusparseFrontend::AddDevicePointerForArguments(bsrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrColIndA);
    CusparseFrontend::AddVariableForArguments<int>(rowBlockDim);
    CusparseFrontend::AddVariableForArguments<int>(colBlockDim);
    CusparseFrontend::Execute("cusparseCgebsr2gebsc_bufferSize");
    if (CusparseFrontend::Success()) {
        pBufferSize = (int *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseZgebsr2gebsc_bufferSize(cusparseHandle_t handle, int mb, int nb, int nnzb, const cuDoubleComplex* bsrValA, const int* bsrRowPtrA, const int* bsrColIndA, int rowBlockDim, int colBlockDim, int* pBufferSize) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<int>(mb);
    CusparseFrontend::AddVariableForArguments<int>(nb);
    CusparseFrontend::AddVariableForArguments<int>(nnzb);
    CusparseFrontend::AddDevicePointerForArguments(bsrValA);
    CusparseFrontend::AddDevicePointerForArguments(bsrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrColIndA);
    CusparseFrontend::AddVariableForArguments<int>(rowBlockDim);
    CusparseFrontend::AddVariableForArguments<int>(colBlockDim);
    CusparseFrontend::Execute("cusparseZgebsr2gebsc_bufferSize");
    if (CusparseFrontend::Success()) {
        pBufferSize = (int *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseSgebsr2gebsc(cusparseHandle_t handle, int mb, int nb, int nnzb, const float* bsrValA, const int* bsrRowPtrA, const int* bsrColIndA, int rowBlockDim, int colBlockDim, float* bsCVal, int* bscRowInd, int* bscColPtr, cusparseAction_t copyValues, cusparseIndexBase_t baseIdx, void* pBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<int>(mb);
    CusparseFrontend::AddVariableForArguments<int>(nb);
    CusparseFrontend::AddVariableForArguments<int>(nnzb);
    CusparseFrontend::AddDevicePointerForArguments(bsrValA);
    CusparseFrontend::AddDevicePointerForArguments(bsrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrColIndA);
    CusparseFrontend::AddVariableForArguments<int>(rowBlockDim);
    CusparseFrontend::AddVariableForArguments<int>(colBlockDim);
    CusparseFrontend::AddDevicePointerForArguments(bsCVal);
    CusparseFrontend::AddDevicePointerForArguments(bscRowInd);
    CusparseFrontend::AddDevicePointerForArguments(bscColPtr);
    CusparseFrontend::AddVariableForArguments<cusparseAction_t>(copyValues);
    CusparseFrontend::AddVariableForArguments<cusparseIndexBase_t>(baseIdx);
    CusparseFrontend::AddDevicePointerForArguments(pBuffer);
    CusparseFrontend::Execute("cusparseSgebsr2gebsc");
    if (CusparseFrontend::Success()) {
        bsCVal = (float *)CusparseFrontend::GetOutputDevicePointer();
        bscRowInd = (int *)CusparseFrontend::GetOutputDevicePointer();
        bscColPtr = (int *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseDgebsr2gebsc(cusparseHandle_t handle, int mb, int nb, int nnzb, const double* bsrValA, const int* bsrRowPtrA, const int* bsrColIndA, int rowBlockDim, int colBlockDim, double* bsCVal, int* bscRowInd, int* bscColPtr, cusparseAction_t copyValues, cusparseIndexBase_t baseIdx, void* pBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<int>(mb);
    CusparseFrontend::AddVariableForArguments<int>(nb);
    CusparseFrontend::AddVariableForArguments<int>(nnzb);
    CusparseFrontend::AddDevicePointerForArguments(bsrValA);
    CusparseFrontend::AddDevicePointerForArguments(bsrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrColIndA);
    CusparseFrontend::AddVariableForArguments<int>(rowBlockDim);
    CusparseFrontend::AddVariableForArguments<int>(colBlockDim);
    CusparseFrontend::AddDevicePointerForArguments(bsCVal);
    CusparseFrontend::AddDevicePointerForArguments(bscRowInd);
    CusparseFrontend::AddDevicePointerForArguments(bscColPtr);
    CusparseFrontend::AddVariableForArguments<cusparseAction_t>(copyValues);
    CusparseFrontend::AddVariableForArguments<cusparseIndexBase_t>(baseIdx);
    CusparseFrontend::AddDevicePointerForArguments(pBuffer);
    CusparseFrontend::Execute("cusparseDgebsr2gebsc");
    if (CusparseFrontend::Success()) {
        bsCVal = (double *)CusparseFrontend::GetOutputDevicePointer();
        bscRowInd = (int *)CusparseFrontend::GetOutputDevicePointer();
        bscColPtr = (int *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseCgebsr2gebsc(cusparseHandle_t handle, int mb, int nb, int nnzb, const cuComplex* bsrValA, const int* bsrRowPtrA, const int* bsrColIndA, int rowBlockDim, int colBlockDim, cuComplex* bsCVal, int* bscRowInd, int* bscColPtr, cusparseAction_t copyValues, cusparseIndexBase_t baseIdx, void* pBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<int>(mb);
    CusparseFrontend::AddVariableForArguments<int>(nb);
    CusparseFrontend::AddVariableForArguments<int>(nnzb);
    CusparseFrontend::AddDevicePointerForArguments(bsrValA);
    CusparseFrontend::AddDevicePointerForArguments(bsrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrColIndA);
    CusparseFrontend::AddVariableForArguments<int>(rowBlockDim);
    CusparseFrontend::AddVariableForArguments<int>(colBlockDim);
    CusparseFrontend::AddDevicePointerForArguments(bsCVal);
    CusparseFrontend::AddDevicePointerForArguments(bscRowInd);
    CusparseFrontend::AddDevicePointerForArguments(bscColPtr);
    CusparseFrontend::AddVariableForArguments<cusparseAction_t>(copyValues);
    CusparseFrontend::AddVariableForArguments<cusparseIndexBase_t>(baseIdx);
    CusparseFrontend::AddDevicePointerForArguments(pBuffer);
    CusparseFrontend::Execute("cusparseCgebsr2gebsc");
    if (CusparseFrontend::Success()) {
        bsCVal = (cuComplex *)CusparseFrontend::GetOutputDevicePointer();
        bscRowInd = (int *)CusparseFrontend::GetOutputDevicePointer();
        bscColPtr = (int *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseZgebsr2gebsc(cusparseHandle_t handle, int mb, int nb, int nnzb, const cuDoubleComplex* bsrValA, const int* bsrRowPtrA, const int* bsrColIndA, int rowBlockDim, int colBlockDim, cuDoubleComplex* bsCVal, int* bscRowInd, int* bscColPtr, cusparseAction_t copyValues, cusparseIndexBase_t baseIdx, void* pBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<int>(mb);
    CusparseFrontend::AddVariableForArguments<int>(nb);
    CusparseFrontend::AddVariableForArguments<int>(nnzb);
    CusparseFrontend::AddDevicePointerForArguments(bsrValA);
    CusparseFrontend::AddDevicePointerForArguments(bsrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrColIndA);
    CusparseFrontend::AddVariableForArguments<int>(rowBlockDim);
    CusparseFrontend::AddVariableForArguments<int>(colBlockDim);
    CusparseFrontend::AddDevicePointerForArguments(bsCVal);
    CusparseFrontend::AddDevicePointerForArguments(bscRowInd);
    CusparseFrontend::AddDevicePointerForArguments(bscColPtr);
    CusparseFrontend::AddVariableForArguments<cusparseAction_t>(copyValues);
    CusparseFrontend::AddVariableForArguments<cusparseIndexBase_t>(baseIdx);
    CusparseFrontend::AddDevicePointerForArguments(pBuffer);
    CusparseFrontend::Execute("cusparseZgebsr2gebsc");
    if (CusparseFrontend::Success()) {
        bsCVal = (cuDoubleComplex *)CusparseFrontend::GetOutputDevicePointer();
        bscRowInd = (int *)CusparseFrontend::GetOutputDevicePointer();
        bscColPtr = (int *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseSgebsr2gebsr_bufferSize(cusparseHandle_t handle, cusparseDirection_t dir, int mb, int nb, int nnzb, const cusparseMatDescr_t descrA, const float* bsrValA, const int* bsrRowPtrA, const int* bsrColIndA, int rowBlockDimA, int colBlockDimA, int rowBlockDimC, int colBlockDimC, int* pBufferSize) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<cusparseDirection_t>(dir);
    CusparseFrontend::AddVariableForArguments<int>(mb);
    CusparseFrontend::AddVariableForArguments<int>(nb);
    CusparseFrontend::AddVariableForArguments<int>(nnzb);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) descrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrValA);
    CusparseFrontend::AddDevicePointerForArguments(bsrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrColIndA);
    CusparseFrontend::AddVariableForArguments<int>(rowBlockDimA);
    CusparseFrontend::AddVariableForArguments<int>(colBlockDimA);
    CusparseFrontend::AddVariableForArguments<int>(rowBlockDimC);
    CusparseFrontend::AddVariableForArguments<int>(colBlockDimC);
    CusparseFrontend::Execute("cusparseSgebsr2gebsr_bufferSize");
    if (CusparseFrontend::Success()) {
        pBufferSize = (int *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseDgebsr2gebsr_bufferSize(cusparseHandle_t handle, cusparseDirection_t dir, int mb, int nb, int nnzb, const cusparseMatDescr_t descrA, const double* bsrValA, const int* bsrRowPtrA, const int* bsrColIndA, int rowBlockDimA, int colBlockDimA, int rowBlockDimC, int colBlockDimC, int* pBufferSize) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<cusparseDirection_t>(dir);
    CusparseFrontend::AddVariableForArguments<int>(mb);
    CusparseFrontend::AddVariableForArguments<int>(nb);
    CusparseFrontend::AddVariableForArguments<int>(nnzb);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) descrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrValA);
    CusparseFrontend::AddDevicePointerForArguments(bsrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrColIndA);
    CusparseFrontend::AddVariableForArguments<int>(rowBlockDimA);
    CusparseFrontend::AddVariableForArguments<int>(colBlockDimA);
    CusparseFrontend::AddVariableForArguments<int>(rowBlockDimC);
    CusparseFrontend::AddVariableForArguments<int>(colBlockDimC);
    CusparseFrontend::Execute("cusparseDgebsr2gebsr_bufferSize");
    if (CusparseFrontend::Success()) {
        pBufferSize = (int *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseCgebsr2gebsr_bufferSize(cusparseHandle_t handle, cusparseDirection_t dir, int mb, int nb, int nnzb, const cusparseMatDescr_t descrA, const cuComplex* bsrValA, const int* bsrRowPtrA, const int* bsrColIndA, int rowBlockDimA, int colBlockDimA, int rowBlockDimC, int colBlockDimC, int* pBufferSize) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<cusparseDirection_t>(dir);
    CusparseFrontend::AddVariableForArguments<int>(mb);
    CusparseFrontend::AddVariableForArguments<int>(nb);
    CusparseFrontend::AddVariableForArguments<int>(nnzb);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) descrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrValA);
    CusparseFrontend::AddDevicePointerForArguments(bsrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrColIndA);
    CusparseFrontend::AddVariableForArguments<int>(rowBlockDimA);
    CusparseFrontend::AddVariableForArguments<int>(colBlockDimA);
    CusparseFrontend::AddVariableForArguments<int>(rowBlockDimC);
    CusparseFrontend::AddVariableForArguments<int>(colBlockDimC);
    CusparseFrontend::Execute("cusparseCgebsr2gebsr_bufferSize");
    if (CusparseFrontend::Success()) {
        pBufferSize = (int *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseZgebsr2gebsr_bufferSize(cusparseHandle_t handle, cusparseDirection_t dir, int mb, int nb, int nnzb, const cusparseMatDescr_t descrA, const cuDoubleComplex* bsrValA, const int* bsrRowPtrA, const int* bsrColIndA, int rowBlockDimA, int colBlockDimA, int rowBlockDimC, int colBlockDimC, int* pBufferSize) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<cusparseDirection_t>(dir);
    CusparseFrontend::AddVariableForArguments<int>(mb);
    CusparseFrontend::AddVariableForArguments<int>(nb);
    CusparseFrontend::AddVariableForArguments<int>(nnzb);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) descrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrValA);
    CusparseFrontend::AddDevicePointerForArguments(bsrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrColIndA);
    CusparseFrontend::AddVariableForArguments<int>(rowBlockDimA);
    CusparseFrontend::AddVariableForArguments<int>(colBlockDimA);
    CusparseFrontend::AddVariableForArguments<int>(rowBlockDimC);
    CusparseFrontend::AddVariableForArguments<int>(colBlockDimC);
    CusparseFrontend::Execute("cusparseZgebsr2gebsr_bufferSize");
    if (CusparseFrontend::Success()) {
        pBufferSize = (int *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseXgebsr2gebsrNnz(cusparseHandle_t handle, cusparseDirection_t dir, int mb, int nb, int nnzb, const cusparseMatDescr_t descrA, const int* bsrRowPtrA, const int* bsrColIndA, int rowBlockDimA, int colBlockDimA, const cusparseMatDescr_t descrC, int* bsrRowPtrC, int rowBlockDimC, int colBlockDimC, int * nnzTotalDevHostPtr, void* pBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<cusparseDirection_t>(dir);
    CusparseFrontend::AddVariableForArguments<int>(mb);
    CusparseFrontend::AddVariableForArguments<int>(nb);
    CusparseFrontend::AddVariableForArguments<int>(nnzb);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) descrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrColIndA);
    CusparseFrontend::AddVariableForArguments<int>(rowBlockDimA);
    CusparseFrontend::AddVariableForArguments<int>(colBlockDimA);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) descrC);
    CusparseFrontend::AddDevicePointerForArguments(bsrRowPtrC);
    CusparseFrontend::AddVariableForArguments<int>(rowBlockDimC);
    CusparseFrontend::AddVariableForArguments<int>(colBlockDimC);
    CusparseFrontend::AddDevicePointerForArguments(pBuffer);
    CusparseFrontend::AddHostPointerForArguments<int>(nnzTotalDevHostPtr);
    CusparseFrontend::Execute("cusparseXgebsr2gebsrNnz");
    if (CusparseFrontend::Success()) {
        bsrRowPtrC = (int *)CusparseFrontend::GetOutputDevicePointer();
        *nnzTotalDevHostPtr = (int) CusparseFrontend::GetOutputVariable<int>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseSgebsr2gebsr(cusparseHandle_t handle, cusparseDirection_t dir, int mb, int nb, int nnzb, const cusparseMatDescr_t descrA, const float* bsrValA, const int* bsrRowPtrA, const int* bsrColIndA, int rowBlockDimA, int colBlockDimA, const cusparseMatDescr_t descrC, float* bsrValC, int* bsrRowPtrC, int* bsrColIndC, int rowBlockDimC, int colBlockDimC, void* pBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<cusparseDirection_t>(dir);
    CusparseFrontend::AddVariableForArguments<int>(mb);
    CusparseFrontend::AddVariableForArguments<int>(nb);
    CusparseFrontend::AddVariableForArguments<int>(nnzb);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) descrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrValA);
    CusparseFrontend::AddDevicePointerForArguments(bsrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrColIndA);
    CusparseFrontend::AddVariableForArguments<int>(rowBlockDimA);
    CusparseFrontend::AddVariableForArguments<int>(colBlockDimA);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) descrC);
    CusparseFrontend::AddDevicePointerForArguments(bsrValC);
    CusparseFrontend::AddDevicePointerForArguments(bsrRowPtrC);
    CusparseFrontend::AddDevicePointerForArguments(bsrColIndC);
    CusparseFrontend::AddVariableForArguments<int>(rowBlockDimC);
    CusparseFrontend::AddVariableForArguments<int>(colBlockDimC);
    CusparseFrontend::AddDevicePointerForArguments(pBuffer);
    CusparseFrontend::Execute("cusparseSgebsr2gebsr");
    if (CusparseFrontend::Success()) {
        bsrValC = (float *)CusparseFrontend::GetOutputDevicePointer();
        bsrRowPtrC = (int *)CusparseFrontend::GetOutputDevicePointer();
        bsrColIndC = (int *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseDgebsr2gebsr(cusparseHandle_t handle, cusparseDirection_t dir, int mb, int nb, int nnzb, const cusparseMatDescr_t descrA, const double* bsrValA, const int* bsrRowPtrA, const int* bsrColIndA, int rowBlockDimA, int colBlockDimA, const cusparseMatDescr_t descrC, double* bsrValC, int* bsrRowPtrC, int* bsrColIndC, int rowBlockDimC, int colBlockDimC, void* pBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<cusparseDirection_t>(dir);
    CusparseFrontend::AddVariableForArguments<int>(mb);
    CusparseFrontend::AddVariableForArguments<int>(nb);
    CusparseFrontend::AddVariableForArguments<int>(nnzb);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) descrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrValA);
    CusparseFrontend::AddDevicePointerForArguments(bsrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrColIndA);
    CusparseFrontend::AddVariableForArguments<int>(rowBlockDimA);
    CusparseFrontend::AddVariableForArguments<int>(colBlockDimA);

    CusparseFrontend::AddVariableForArguments<size_t>((size_t) descrC);
    CusparseFrontend::AddDevicePointerForArguments(bsrValC);
    CusparseFrontend::AddDevicePointerForArguments(bsrRowPtrC);
    CusparseFrontend::AddDevicePointerForArguments(bsrColIndC);
    CusparseFrontend::AddVariableForArguments<int>(rowBlockDimC);
    CusparseFrontend::AddVariableForArguments<int>(colBlockDimC);
    CusparseFrontend::AddDevicePointerForArguments(pBuffer);
    CusparseFrontend::Execute("cusparseDgebsr2gebsr");
    if (CusparseFrontend::Success()) {
        bsrValC = (double *)CusparseFrontend::GetOutputDevicePointer();
        bsrRowPtrC = (int *)CusparseFrontend::GetOutputDevicePointer();
        bsrColIndC = (int *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseCgebsr2gebsr(cusparseHandle_t handle, cusparseDirection_t dir, int mb, int nb, int nnzb, const cusparseMatDescr_t descrA, const cuComplex* bsrValA, const int* bsrRowPtrA, const int* bsrColIndA, int rowBlockDimA, int colBlockDimA, const cusparseMatDescr_t descrC, cuComplex* bsrValC, int* bsrRowPtrC, int* bsrColIndC, int rowBlockDimC, int colBlockDimC, void* pBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<cusparseDirection_t>(dir);
    CusparseFrontend::AddVariableForArguments<int>(mb);
    CusparseFrontend::AddVariableForArguments<int>(nb);
    CusparseFrontend::AddVariableForArguments<int>(nnzb);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) descrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrValA);
    CusparseFrontend::AddDevicePointerForArguments(bsrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrColIndA);
    CusparseFrontend::AddVariableForArguments<int>(rowBlockDimA);
    CusparseFrontend::AddVariableForArguments<int>(colBlockDimA);

    CusparseFrontend::AddVariableForArguments<size_t>((size_t) descrC);
    CusparseFrontend::AddDevicePointerForArguments(bsrValC);
    CusparseFrontend::AddDevicePointerForArguments(bsrRowPtrC);
    CusparseFrontend::AddDevicePointerForArguments(bsrColIndC);
    CusparseFrontend::AddVariableForArguments<int>(rowBlockDimC);
    CusparseFrontend::AddVariableForArguments<int>(colBlockDimC);
    CusparseFrontend::AddDevicePointerForArguments(pBuffer);
    CusparseFrontend::Execute("cusparseCgebsr2gebsr");
    if (CusparseFrontend::Success()) {
        bsrValC = (cuComplex *)CusparseFrontend::GetOutputDevicePointer();
        bsrRowPtrC = (int *)CusparseFrontend::GetOutputDevicePointer();
        bsrColIndC = (int *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseZgebsr2gebsr(cusparseHandle_t handle, cusparseDirection_t dir, int mb, int nb, int nnzb, const cusparseMatDescr_t descrA, const cuDoubleComplex* bsrValA, const int* bsrRowPtrA, const int* bsrColIndA, int rowBlockDimA, int colBlockDimA, const cusparseMatDescr_t descrC, cuDoubleComplex* bsrValC, int* bsrRowPtrC, int* bsrColIndC, int rowBlockDimC, int colBlockDimC, void* pBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<cusparseDirection_t>(dir);
    CusparseFrontend::AddVariableForArguments<int>(mb);
    CusparseFrontend::AddVariableForArguments<int>(nb);
    CusparseFrontend::AddVariableForArguments<int>(nnzb);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) descrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrValA);
    CusparseFrontend::AddDevicePointerForArguments(bsrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrColIndA);
    CusparseFrontend::AddVariableForArguments<int>(rowBlockDimA);
    CusparseFrontend::AddVariableForArguments<int>(colBlockDimA);

    CusparseFrontend::AddVariableForArguments<size_t>((size_t) descrC);
    CusparseFrontend::AddDevicePointerForArguments(bsrValC);
    CusparseFrontend::AddDevicePointerForArguments(bsrRowPtrC);
    CusparseFrontend::AddDevicePointerForArguments(bsrColIndC);
    CusparseFrontend::AddVariableForArguments<int>(rowBlockDimC);
    CusparseFrontend::AddVariableForArguments<int>(colBlockDimC);
    CusparseFrontend::AddDevicePointerForArguments(pBuffer);
    CusparseFrontend::Execute("cusparseZgebsr2gebsr");
    if (CusparseFrontend::Success()) {
        bsrValC = (cuDoubleComplex *)CusparseFrontend::GetOutputDevicePointer();
        bsrRowPtrC = (int *)CusparseFrontend::GetOutputDevicePointer();
        bsrColIndC = (int *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseSgebsr2csr(cusparseHandle_t handle, cusparseDirection_t dir, int mb, int nb, const cusparseMatDescr_t descrA, const float* bsrValA, const int* bsrRowPtrA, const int* bsrColIndA, int rowBlockDim, int colBlockDim, const cusparseMatDescr_t descrC, float* csrValC, int* csrRowPtrC, int* csrColIndC) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<cusparseDirection_t>(dir);
    CusparseFrontend::AddVariableForArguments<int>(mb);
    CusparseFrontend::AddVariableForArguments<int>(nb);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) descrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrValA);
    CusparseFrontend::AddDevicePointerForArguments(bsrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrColIndA);
    CusparseFrontend::AddVariableForArguments<int>(rowBlockDim);
    CusparseFrontend::AddVariableForArguments<int>(colBlockDim);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) descrC);
    CusparseFrontend::AddDevicePointerForArguments(csrValC);
    CusparseFrontend::AddDevicePointerForArguments(csrRowPtrC);
    CusparseFrontend::AddDevicePointerForArguments(csrColIndC);
    CusparseFrontend::Execute("cusparseSgebsr2csr");
    if (CusparseFrontend::Success()) {
        csrValC = (float *)CusparseFrontend::GetOutputDevicePointer();
        csrRowPtrC = (int *)CusparseFrontend::GetOutputDevicePointer();
        csrColIndC = (int *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseDgebsr2csr(cusparseHandle_t handle, cusparseDirection_t dir, int mb, int nb, const cusparseMatDescr_t descrA, const double* bsrValA, const int* bsrRowPtrA, const int* bsrColIndA, int rowBlockDim, int colBlockDim, const cusparseMatDescr_t descrC, double* csrValC, int* csrRowPtrC, int* csrColIndC) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<cusparseDirection_t>(dir);
    CusparseFrontend::AddVariableForArguments<int>(mb);
    CusparseFrontend::AddVariableForArguments<int>(nb);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) descrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrValA);
    CusparseFrontend::AddDevicePointerForArguments(bsrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrColIndA);
    CusparseFrontend::AddVariableForArguments<int>(rowBlockDim);
    CusparseFrontend::AddVariableForArguments<int>(colBlockDim);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) descrC);
    CusparseFrontend::AddDevicePointerForArguments(csrValC);
    CusparseFrontend::AddDevicePointerForArguments(csrRowPtrC);
    CusparseFrontend::AddDevicePointerForArguments(csrColIndC);
    CusparseFrontend::Execute("cusparseDgebsr2csr");
    if (CusparseFrontend::Success()) {
        csrValC = (double *)CusparseFrontend::GetOutputDevicePointer();
        csrRowPtrC = (int *)CusparseFrontend::GetOutputDevicePointer();
        csrColIndC = (int *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseCgebsr2csr(cusparseHandle_t handle, cusparseDirection_t dir, int mb, int nb, const cusparseMatDescr_t descrA, const cuComplex* bsrValA, const int* bsrRowPtrA, const int* bsrColIndA, int rowBlockDim, int colBlockDim, const cusparseMatDescr_t descrC, cuComplex* csrValC, int* csrRowPtrC, int* csrColIndC) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<cusparseDirection_t>(dir);
    CusparseFrontend::AddVariableForArguments<int>(mb);
    CusparseFrontend::AddVariableForArguments<int>(nb);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) descrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrValA);
    CusparseFrontend::AddDevicePointerForArguments(bsrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrColIndA);
    CusparseFrontend::AddVariableForArguments<int>(rowBlockDim);
    CusparseFrontend::AddVariableForArguments<int>(colBlockDim);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) descrC);
    CusparseFrontend::AddDevicePointerForArguments(csrValC);
    CusparseFrontend::AddDevicePointerForArguments(csrRowPtrC);
    CusparseFrontend::AddDevicePointerForArguments(csrColIndC);
    CusparseFrontend::Execute("cusparseCgebsr2csr");
    if (CusparseFrontend::Success()) {
        csrValC = (cuComplex *)CusparseFrontend::GetOutputDevicePointer();
        csrRowPtrC = (int *)CusparseFrontend::GetOutputDevicePointer();
        csrColIndC = (int *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseZgebsr2csr(cusparseHandle_t handle, cusparseDirection_t dir, int mb, int nb, const cusparseMatDescr_t descrA, const cuDoubleComplex* bsrValA, const int* bsrRowPtrA, const int* bsrColIndA, int rowBlockDim, int colBlockDim, const cusparseMatDescr_t descrC, cuDoubleComplex* csrValC, int* csrRowPtrC, int* csrColIndC) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<cusparseDirection_t>(dir);
    CusparseFrontend::AddVariableForArguments<int>(mb);
    CusparseFrontend::AddVariableForArguments<int>(nb);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) descrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrValA);
    CusparseFrontend::AddDevicePointerForArguments(bsrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrColIndA);
    CusparseFrontend::AddVariableForArguments<int>(rowBlockDim);
    CusparseFrontend::AddVariableForArguments<int>(colBlockDim);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) descrC);
    CusparseFrontend::AddDevicePointerForArguments(csrValC);
    CusparseFrontend::AddDevicePointerForArguments(csrRowPtrC);
    CusparseFrontend::AddDevicePointerForArguments(csrColIndC);
    CusparseFrontend::Execute("cusparseZgebsr2csr");
    if (CusparseFrontend::Success()) {
        csrValC = (cuDoubleComplex *)CusparseFrontend::GetOutputDevicePointer();
        csrRowPtrC = (int *)CusparseFrontend::GetOutputDevicePointer();
        csrColIndC = (int *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseScsr2gebsr_bufferSize(cusparseHandle_t handle, cusparseDirection_t dir, int m, int n, const cusparseMatDescr_t descrA, const float* csrValA, const int* csrRowPtrA, const int* csrColIndA, int rowBlockDim, int colBlockDim, int* pBufferSize) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<cusparseDirection_t>(dir);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddVariableForArguments<int>(n);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) descrA);
    CusparseFrontend::AddDevicePointerForArguments(csrValA);
    CusparseFrontend::AddDevicePointerForArguments(csrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(csrColIndA);
    CusparseFrontend::AddVariableForArguments<int>(rowBlockDim);
    CusparseFrontend::AddVariableForArguments<int>(colBlockDim);
    CusparseFrontend::Execute("cusparseScsr2gebsr_bufferSize");
    if (CusparseFrontend::Success()) {
        pBufferSize = (int *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseDcsr2gebsr_bufferSize(cusparseHandle_t handle, cusparseDirection_t dir, int m, int n, const cusparseMatDescr_t descrA, const double* csrValA, const int* csrRowPtrA, const int* csrColIndA, int rowBlockDim, int colBlockDim, int* pBufferSize) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<cusparseDirection_t>(dir);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddVariableForArguments<int>(n);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) descrA);
    CusparseFrontend::AddDevicePointerForArguments(csrValA);
    CusparseFrontend::AddDevicePointerForArguments(csrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(csrColIndA);
    CusparseFrontend::AddVariableForArguments<int>(rowBlockDim);
    CusparseFrontend::AddVariableForArguments<int>(colBlockDim);
    CusparseFrontend::Execute("cusparseDcsr2gebsr_bufferSize");
    if (CusparseFrontend::Success()) {
        pBufferSize = (int *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseCcsr2gebsr_bufferSize(cusparseHandle_t handle, cusparseDirection_t dir, int m, int n, const cusparseMatDescr_t descrA, const cuComplex* csrValA, const int* csrRowPtrA, const int* csrColIndA, int rowBlockDim, int colBlockDim, int* pBufferSize) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<cusparseDirection_t>(dir);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddVariableForArguments<int>(n);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) descrA);
    CusparseFrontend::AddDevicePointerForArguments(csrValA);
    CusparseFrontend::AddDevicePointerForArguments(csrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(csrColIndA);
    CusparseFrontend::AddVariableForArguments<int>(rowBlockDim);
    CusparseFrontend::AddVariableForArguments<int>(colBlockDim);
    CusparseFrontend::Execute("cusparseCcsr2gebsr_bufferSize");
    if (CusparseFrontend::Success()) {
        pBufferSize = (int *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseZcsr2gebsr_bufferSize(cusparseHandle_t handle, cusparseDirection_t dir, int m, int n, const cusparseMatDescr_t descrA, const cuDoubleComplex* csrValA, const int* csrRowPtrA, const int* csrColIndA, int rowBlockDim, int colBlockDim, int* pBufferSize) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<cusparseDirection_t>(dir);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddVariableForArguments<int>(n);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) descrA);
    CusparseFrontend::AddDevicePointerForArguments(csrValA);
    CusparseFrontend::AddDevicePointerForArguments(csrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(csrColIndA);
    CusparseFrontend::AddVariableForArguments<int>(rowBlockDim);
    CusparseFrontend::AddVariableForArguments<int>(colBlockDim);
    CusparseFrontend::Execute("cusparseZcsr2gebsr_bufferSize");
    if (CusparseFrontend::Success()) {
        pBufferSize = (int *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseXcsr2gebsrNnz(cusparseHandle_t handle, cusparseDirection_t dir, int m, int n, const cusparseMatDescr_t descrA, const int* csrRowPtrA, const int* csrColIndA, const cusparseMatDescr_t descrC, int* bsrRowPtrC, int rowBlockDim, int colBlockDim, int * nnzTotalDevHostPtr, void* pBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<cusparseDirection_t>(dir);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddVariableForArguments<int>(n);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) descrA);
    CusparseFrontend::AddDevicePointerForArguments(csrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(csrColIndA);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) descrC);
    CusparseFrontend::AddDevicePointerForArguments(bsrRowPtrC);
    CusparseFrontend::AddVariableForArguments<int>(rowBlockDim);
    CusparseFrontend::AddVariableForArguments<int>(colBlockDim);
    CusparseFrontend::AddDevicePointerForArguments(pBuffer);
    CusparseFrontend::AddHostPointerForArguments<int>(nnzTotalDevHostPtr);
    CusparseFrontend::Execute("cusparseXcsr2gebsrNnz");
    if (CusparseFrontend::Success()) {
        bsrRowPtrC = (int *)CusparseFrontend::GetOutputDevicePointer();
        *nnzTotalDevHostPtr = (int) CusparseFrontend::GetOutputVariable<int>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseScsr2gebsr(cusparseHandle_t handle, cusparseDirection_t dir, int m, int n, const cusparseMatDescr_t descrA, const float* csrValA, const int* csrRowPtrA, const int* csrColIndA, const cusparseMatDescr_t descrC, float* bsrValC, int* bsrRowPtrC, int* bsrColIndC, int rowBlockDim, int colBlockDim, void* pBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<cusparseDirection_t>(dir);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddVariableForArguments<int>(n);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) descrA);
    CusparseFrontend::AddDevicePointerForArguments(csrValA);
    CusparseFrontend::AddDevicePointerForArguments(csrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(csrColIndA);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) descrC);
    CusparseFrontend::AddDevicePointerForArguments(bsrValC);
    CusparseFrontend::AddDevicePointerForArguments(bsrRowPtrC);
    CusparseFrontend::AddDevicePointerForArguments(bsrColIndC);
    CusparseFrontend::AddVariableForArguments<int>(rowBlockDim);
    CusparseFrontend::AddVariableForArguments<int>(colBlockDim);
    CusparseFrontend::AddDevicePointerForArguments(pBuffer);
    CusparseFrontend::Execute("cusparseScsr2gebsr");
    if (CusparseFrontend::Success()) {
        bsrValC = (float *)CusparseFrontend::GetOutputDevicePointer();
        bsrRowPtrC = (int *)CusparseFrontend::GetOutputDevicePointer();
        bsrColIndC = (int *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseDcsr2gebsr(cusparseHandle_t handle, cusparseDirection_t dir, int m, int n, const cusparseMatDescr_t descrA, const double* csrValA, const int* csrRowPtrA, const int* csrColIndA, const cusparseMatDescr_t descrC, double* bsrValC, int* bsrRowPtrC, int* bsrColIndC, int rowBlockDim, int colBlockDim, void* pBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<cusparseDirection_t>(dir);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddVariableForArguments<int>(n);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) descrA);
    CusparseFrontend::AddDevicePointerForArguments(csrValA);
    CusparseFrontend::AddDevicePointerForArguments(csrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(csrColIndA);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) descrC);
    CusparseFrontend::AddDevicePointerForArguments(bsrValC);
    CusparseFrontend::AddDevicePointerForArguments(bsrRowPtrC);
    CusparseFrontend::AddDevicePointerForArguments(bsrColIndC);
    CusparseFrontend::AddVariableForArguments<int>(rowBlockDim);
    CusparseFrontend::AddVariableForArguments<int>(colBlockDim);
    CusparseFrontend::AddDevicePointerForArguments(pBuffer);
    CusparseFrontend::Execute("cusparseDcsr2gebsr");
    if (CusparseFrontend::Success()) {
        bsrValC = (double *)CusparseFrontend::GetOutputDevicePointer();
        bsrRowPtrC = (int *)CusparseFrontend::GetOutputDevicePointer();
        bsrColIndC = (int *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseCcsr2gebsr(cusparseHandle_t handle, cusparseDirection_t dir, int m, int n, const cusparseMatDescr_t descrA, const cuComplex* csrValA, const int* csrRowPtrA, const int* csrColIndA, const cusparseMatDescr_t descrC, cuComplex* bsrValC, int* bsrRowPtrC, int* bsrColIndC, int rowBlockDim, int colBlockDim, void* pBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<cusparseDirection_t>(dir);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddVariableForArguments<int>(n);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) descrA);
    CusparseFrontend::AddDevicePointerForArguments(csrValA);
    CusparseFrontend::AddDevicePointerForArguments(csrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(csrColIndA);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) descrC);
    CusparseFrontend::AddDevicePointerForArguments(bsrValC);
    CusparseFrontend::AddDevicePointerForArguments(bsrRowPtrC);
    CusparseFrontend::AddDevicePointerForArguments(bsrColIndC);
    CusparseFrontend::AddVariableForArguments<int>(rowBlockDim);
    CusparseFrontend::AddVariableForArguments<int>(colBlockDim);
    CusparseFrontend::AddDevicePointerForArguments(pBuffer);
    CusparseFrontend::Execute("cusparseCcsr2gebsr");
    if (CusparseFrontend::Success()) {
        bsrValC = (cuComplex *)CusparseFrontend::GetOutputDevicePointer();
        bsrRowPtrC = (int *)CusparseFrontend::GetOutputDevicePointer();
        bsrColIndC = (int *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseZcsr2gebsr(cusparseHandle_t handle, cusparseDirection_t dir, int m, int n, const cusparseMatDescr_t descrA, const cuDoubleComplex* csrValA, const int* csrRowPtrA, const int* csrColIndA, const cusparseMatDescr_t descrC, cuDoubleComplex* bsrValC, int* bsrRowPtrC, int* bsrColIndC, int rowBlockDim, int colBlockDim, void* pBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<cusparseDirection_t>(dir);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddVariableForArguments<int>(n);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) descrA);
    CusparseFrontend::AddDevicePointerForArguments(csrValA);
    CusparseFrontend::AddDevicePointerForArguments(csrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(csrColIndA);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) descrC);
    CusparseFrontend::AddDevicePointerForArguments(bsrValC);
    CusparseFrontend::AddDevicePointerForArguments(bsrRowPtrC);
    CusparseFrontend::AddDevicePointerForArguments(bsrColIndC);
    CusparseFrontend::AddVariableForArguments<int>(rowBlockDim);
    CusparseFrontend::AddVariableForArguments<int>(colBlockDim);
    CusparseFrontend::AddDevicePointerForArguments(pBuffer);
    CusparseFrontend::Execute("cusparseZcsr2gebsr");
    if (CusparseFrontend::Success()) {
        bsrValC = (cuDoubleComplex *)CusparseFrontend::GetOutputDevicePointer();
        bsrRowPtrC = (int *)CusparseFrontend::GetOutputDevicePointer();
        bsrColIndC = (int *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseXcoo2csr(cusparseHandle_t handle, const int* cooRowInd, int nnz, int m, int* csrRowPtr, cusparseIndexBase_t idxBase) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddDevicePointerForArguments(cooRowInd);
    CusparseFrontend::AddVariableForArguments<int>(nnz);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddDevicePointerForArguments(csrRowPtr);
    CusparseFrontend::AddVariableForArguments<cusparseIndexBase_t>(idxBase);
    CusparseFrontend::Execute("cusparseXcoo2csr");
    if (CusparseFrontend::Success()) {
        csrRowPtr = (int *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseScsc2dense(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA, const float* cscValA, const int* cscRowIndA, const int* cscColPtrA, float* A, int lda) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddVariableForArguments<int>(n);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) descrA);
    CusparseFrontend::AddDevicePointerForArguments(cscValA);
    CusparseFrontend::AddDevicePointerForArguments(cscRowIndA);
    CusparseFrontend::AddDevicePointerForArguments(cscColPtrA);
    CusparseFrontend::AddDevicePointerForArguments(A);
    CusparseFrontend::AddVariableForArguments<int>(lda);
    CusparseFrontend::Execute("cusparseScsc2dense");
    if (CusparseFrontend::Success()) {
        A = (float *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseDcsc2dense(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA, const double* cscValA, const int* cscRowIndA, const int* cscColPtrA, double* A, int lda) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddVariableForArguments<int>(n);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) descrA);
    CusparseFrontend::AddDevicePointerForArguments(cscValA);
    CusparseFrontend::AddDevicePointerForArguments(cscRowIndA);
    CusparseFrontend::AddDevicePointerForArguments(cscColPtrA);
    CusparseFrontend::AddDevicePointerForArguments(A);
    CusparseFrontend::AddVariableForArguments<int>(lda);
    CusparseFrontend::Execute("cusparseDcsc2dense");
    if (CusparseFrontend::Success()) {
        A = (double *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseCcsc2dense(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA, const cuComplex* cscValA, const int* cscRowIndA, const int* cscColPtrA, cuComplex* A, int lda) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddVariableForArguments<int>(n);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) descrA);
    CusparseFrontend::AddDevicePointerForArguments(cscValA);
    CusparseFrontend::AddDevicePointerForArguments(cscRowIndA);
    CusparseFrontend::AddDevicePointerForArguments(cscColPtrA);
    CusparseFrontend::AddDevicePointerForArguments(A);
    CusparseFrontend::AddVariableForArguments<int>(lda);
    CusparseFrontend::Execute("cusparseCcsc2dense");
    if (CusparseFrontend::Success()) {
        A = (cuComplex *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseZcsc2dense(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA, const cuDoubleComplex* cscValA, const int* cscRowIndA, const int* cscColPtrA, cuDoubleComplex* A, int lda) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddVariableForArguments<int>(n);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) descrA);
    CusparseFrontend::AddDevicePointerForArguments(cscValA);
    CusparseFrontend::AddDevicePointerForArguments(cscRowIndA);
    CusparseFrontend::AddDevicePointerForArguments(cscColPtrA);
    CusparseFrontend::AddDevicePointerForArguments(A);
    CusparseFrontend::AddVariableForArguments<int>(lda);
    CusparseFrontend::Execute("cusparseZcsc2dense");
    if (CusparseFrontend::Success()) {
        A = (cuDoubleComplex *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseXcsr2bsrNnz(cusparseHandle_t handle, cusparseDirection_t dir, int m, int n, const cusparseMatDescr_t descrA, const int* csrRowPtrA, const int* csrColIndA, int blockDim, const cusparseMatDescr_t descrC, int* bsrRowPtrC, int* nnzTotalDevHostPtr) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<cusparseDirection_t>(dir);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddVariableForArguments<int>(n);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusparseFrontend::AddDevicePointerForArguments(csrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(csrColIndA);
    CusparseFrontend::AddVariableForArguments<int>(blockDim);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrC);
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
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<cusparseDirection_t>(dir);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddVariableForArguments<int>(n);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusparseFrontend::AddDevicePointerForArguments(csrValA);
    CusparseFrontend::AddDevicePointerForArguments(csrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(csrColIndA);
    CusparseFrontend::AddVariableForArguments<int>(blockDim);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrC);
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
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<cusparseDirection_t>(dir);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddVariableForArguments<int>(n);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusparseFrontend::AddDevicePointerForArguments(csrValA);
    CusparseFrontend::AddDevicePointerForArguments(csrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(csrColIndA);
    CusparseFrontend::AddVariableForArguments<int>(blockDim);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrC);
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
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<cusparseDirection_t>(dir);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddVariableForArguments<int>(n);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusparseFrontend::AddDevicePointerForArguments(csrValA);
    CusparseFrontend::AddDevicePointerForArguments(csrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(csrColIndA);
    CusparseFrontend::AddVariableForArguments<int>(blockDim);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrC);
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
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<cusparseDirection_t>(dir);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddVariableForArguments<int>(n);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusparseFrontend::AddDevicePointerForArguments(csrValA);
    CusparseFrontend::AddDevicePointerForArguments(csrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(csrColIndA);
    CusparseFrontend::AddVariableForArguments<int>(blockDim);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrC);
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

extern "C" cusparseStatus_t CUSPARSEAPI cusparseXcsr2coo(cusparseHandle_t handle, const int* csrRowPtr, int nnz, int m, int* cooRowInd, cusparseIndexBase_t idxBase) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddDevicePointerForArguments(csrRowPtr);
    CusparseFrontend::AddVariableForArguments<int>(nnz);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddDevicePointerForArguments(cooRowInd);
    CusparseFrontend::AddVariableForArguments<cusparseIndexBase_t>(idxBase);
    CusparseFrontend::Execute("cusparseXcsr2coo");
    if (CusparseFrontend::Success()) {
        cooRowInd = (int *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseCsr2cscEx2_bufferSize(cusparseHandle_t handle, int m, int n, int nnz, const void* csrVal, const int* csrRowPtr, const int* csrColInd, void* cscVal, int* cscColPtr, int* cscRowInd, cudaDataType valType, cusparseAction_t copyValues, cusparseIndexBase_t idxBase, cusparseCsr2CscAlg_t alg, size_t* bufferSize) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddVariableForArguments<int>(n);
    CusparseFrontend::AddVariableForArguments<int>(nnz);
    CusparseFrontend::AddDevicePointerForArguments(csrRowPtr);
    CusparseFrontend::AddDevicePointerForArguments(csrColInd);
    CusparseFrontend::AddDevicePointerForArguments(cscColPtr);
    CusparseFrontend::AddDevicePointerForArguments(cscRowInd);
    CusparseFrontend::AddVariableForArguments<cudaDataType>(valType);
    CusparseFrontend::AddVariableForArguments<cusparseAction_t>(copyValues);
    CusparseFrontend::AddVariableForArguments<cusparseIndexBase_t>(idxBase);
    CusparseFrontend::AddVariableForArguments<cusparseCsr2CscAlg_t>(alg);
    switch(valType){
        case CUDA_R_32F:
            //float
            CusparseFrontend::AddDevicePointerForArguments((float *)csrVal);
            CusparseFrontend::AddDevicePointerForArguments((float *)cscVal);
            break;
        case CUDA_R_64F:
            //double
            CusparseFrontend::AddDevicePointerForArguments((double *)csrVal);
            CusparseFrontend::AddDevicePointerForArguments((double *)cscVal);
            break;
        case CUDA_C_32F:
            //cuComplex
            CusparseFrontend::AddDevicePointerForArguments((cuComplex *)csrVal);
            CusparseFrontend::AddDevicePointerForArguments((cuComplex *)cscVal);
            break;
        case CUDA_C_64F:
            //cuDoubleComplex
            CusparseFrontend::AddDevicePointerForArguments((cuDoubleComplex *)csrVal);
            CusparseFrontend::AddDevicePointerForArguments((cuDoubleComplex *)cscVal);
            break;
        default:
            throw "Type not supported by GVirtus!";
    }
    CusparseFrontend::Execute("cusparseCsr2cscEx2_bufferSize");
    if (CusparseFrontend::Success()) {
        bufferSize = (size_t *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseCsr2cscEx2(cusparseHandle_t handle, int m, int n, int nnz, const void* csrVal, const int* csrRowPtr, const int* csrColInd, void* cscVal, int* cscColPtr, int* cscRowInd, cudaDataType valType, cusparseAction_t copyValues, cusparseIndexBase_t idxBase, cusparseCsr2CscAlg_t alg, void* buffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddVariableForArguments<int>(n);
    CusparseFrontend::AddVariableForArguments<int>(nnz);
    CusparseFrontend::AddDevicePointerForArguments(csrRowPtr);
    CusparseFrontend::AddDevicePointerForArguments(csrColInd);
    CusparseFrontend::AddDevicePointerForArguments(cscColPtr);
    CusparseFrontend::AddDevicePointerForArguments(cscRowInd);
    CusparseFrontend::AddVariableForArguments<cudaDataType>(valType);
    CusparseFrontend::AddVariableForArguments<cusparseAction_t>(copyValues);
    CusparseFrontend::AddVariableForArguments<cusparseIndexBase_t>(idxBase);
    CusparseFrontend::AddVariableForArguments<cusparseCsr2CscAlg_t>(alg);
    CusparseFrontend::AddDevicePointerForArguments(buffer);
    switch(valType){
        case CUDA_R_32F:
            //float
            CusparseFrontend::AddDevicePointerForArguments((float *)csrVal);
            CusparseFrontend::AddDevicePointerForArguments((float *)cscVal);
            break;
        case CUDA_R_64F:
            //double
            CusparseFrontend::AddDevicePointerForArguments((double *)csrVal);
            CusparseFrontend::AddDevicePointerForArguments((double *)cscVal);
            break;
        case CUDA_C_32F:
            //cuComplex
            CusparseFrontend::AddDevicePointerForArguments((cuComplex *)csrVal);
            CusparseFrontend::AddDevicePointerForArguments((cuComplex *)cscVal);
            break;
        case CUDA_C_64F:
            //cuDoubleComplex
            CusparseFrontend::AddDevicePointerForArguments((cuDoubleComplex *)csrVal);
            CusparseFrontend::AddDevicePointerForArguments((cuDoubleComplex *)cscVal);
            break;
        default:
            throw "Type not supported by GVirtus!";
    }
    CusparseFrontend::Execute("cusparseCsr2cscEx2");
    if (CusparseFrontend::Success()) {
        switch(valType){
            case CUDA_R_32F:
                //float
                cscVal = (float *)CusparseFrontend::GetOutputDevicePointer();
                break;
            case CUDA_R_64F:
                //double
                cscVal = (double *)CusparseFrontend::GetOutputDevicePointer();
                break;
            case CUDA_C_32F:
                //cuComplex
                cscVal = (cuComplex *)CusparseFrontend::GetOutputDevicePointer();
                break;
            case CUDA_C_64F:
                //cuDoubleComplex
                cscVal = (cuDoubleComplex *)CusparseFrontend::GetOutputDevicePointer();
                break;
            default:
                throw "Type not supported by GVirtus!";
        }
        cscColPtr = (int *)CusparseFrontend::GetOutputDevicePointer();
        cscRowInd = (int *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseSdense2csr(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA, const float* A, int lda, const int* nnzPerRow, float* csrValA, int* csrRowPtrA, int* csrColIndA) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddVariableForArguments<int>(n);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrA);
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
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddVariableForArguments<int>(n);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrA);
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
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddVariableForArguments<int>(n);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrA);
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
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddVariableForArguments<int>(n);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrA);
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
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<cusparseDirection_t>(dirA);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddVariableForArguments<int>(n);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrA);
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
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<cusparseDirection_t>(dirA);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddVariableForArguments<int>(n);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrA);
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
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<cusparseDirection_t>(dirA);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddVariableForArguments<int>(n);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrA);
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
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<cusparseDirection_t>(dirA);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddVariableForArguments<int>(n);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrA);
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