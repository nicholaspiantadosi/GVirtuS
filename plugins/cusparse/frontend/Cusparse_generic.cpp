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
#include "Utilities.h"

using namespace std;

extern "C" cusparseStatus_t CUSPARSEAPI cusparseCreateSpVec(cusparseSpVecDescr_t* spVecDescr, int64_t size, int64_t nnz, void* indices, void* values, cusparseIndexType_t idxType, cusparseIndexBase_t idxBase, cudaDataType valueType) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<int64_t>(size);
    CusparseFrontend::AddVariableForArguments<int64_t>(nnz);
    CusparseFrontend::AddDevicePointerForArguments(indices);
    CusparseFrontend::AddDevicePointerForArguments(values);
    CusparseFrontend::AddVariableForArguments<cusparseIndexType_t>(idxType);
    CusparseFrontend::AddVariableForArguments<cusparseIndexBase_t>(idxBase);
    CusparseFrontend::AddVariableForArguments<cudaDataType>(valueType);
    CusparseFrontend::Execute("cusparseCreateSpVec");
    if (CusparseFrontend::Success()) {
        *spVecDescr = *(CusparseFrontend::GetOutputHostPointer<cusparseSpVecDescr_t>());
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseDestroySpVec(cusparseSpVecDescr_t spVecDescr) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) spVecDescr);
    CusparseFrontend::Execute("cusparseDestroySpVec");
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseSpVecGet(cusparseSpVecDescr_t spVecDescr, int64_t* size, int64_t* nnz, void** indices, void** values, cusparseIndexType_t* idxType, cusparseIndexBase_t* idxBase, cudaDataType* valueType) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)spVecDescr);
    CusparseFrontend::Execute("cusparseSpVecGet");
    if (CusparseFrontend::Success()) {
        *size = (int64_t) CusparseFrontend::GetOutputVariable<size_t>();
        *nnz = (int64_t) CusparseFrontend::GetOutputVariable<size_t>();
        *indices = (void *) CusparseFrontend::GetOutputDevicePointer();
        *values = (void *) CusparseFrontend::GetOutputDevicePointer();
        *idxType = CusparseFrontend::GetOutputVariable<cusparseIndexType_t>();
        *idxBase = CusparseFrontend::GetOutputVariable<cusparseIndexBase_t>();
        *valueType = CusparseFrontend::GetOutputVariable<cudaDataType>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseSpVecGetIndexBase(cusparseSpVecDescr_t spVecDescr, cusparseIndexBase_t* idxBase) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)spVecDescr);
    CusparseFrontend::Execute("cusparseSpVecGetIndexBase");
    if (CusparseFrontend::Success()) {
        *idxBase = CusparseFrontend::GetOutputVariable<cusparseIndexBase_t>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseSpVecGetValues(cusparseSpVecDescr_t spVecDescr, void** values) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)spVecDescr);
    CusparseFrontend::Execute("cusparseSpVecGetValues");
    if (CusparseFrontend::Success()) {
        *values = (void *) CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseSpVecSetValues(cusparseSpVecDescr_t spVecDescr, void* values) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)spVecDescr);
    CusparseFrontend::AddDevicePointerForArguments(values);
    CusparseFrontend::Execute("cusparseSpVecSetValues");
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseCreateCoo(cusparseSpMatDescr_t* spMatDescr, int64_t rows, int64_t cols, int64_t nnz, void* cooRowInd, void* cooColInd, void* cooValues, cusparseIndexType_t cooIdxType, cusparseIndexBase_t idxBase, cudaDataType valueType) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<int64_t>(rows);
    CusparseFrontend::AddVariableForArguments<int64_t>(cols);
    CusparseFrontend::AddVariableForArguments<int64_t>(nnz);
    CusparseFrontend::AddDevicePointerForArguments(cooRowInd);
    CusparseFrontend::AddDevicePointerForArguments(cooColInd);
    CusparseFrontend::AddDevicePointerForArguments(cooValues);
    CusparseFrontend::AddVariableForArguments<cusparseIndexType_t>(cooIdxType);
    CusparseFrontend::AddVariableForArguments<cusparseIndexBase_t>(idxBase);
    CusparseFrontend::AddVariableForArguments<cudaDataType>(valueType);
    CusparseFrontend::AddHostPointerForArguments(spMatDescr);
    CusparseFrontend::Execute("cusparseCreateCoo");
    if (CusparseFrontend::Success()) {
        *spMatDescr = *(CusparseFrontend::GetOutputHostPointer<cusparseSpMatDescr_t>());
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseCreateCooAoS(cusparseSpMatDescr_t* spMatDescr, int64_t rows, int64_t cols, int64_t nnz, void* cooInd, void* cooValues, cusparseIndexType_t cooIdxType, cusparseIndexBase_t idxBase, cudaDataType valueType) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<int64_t>(rows);
    CusparseFrontend::AddVariableForArguments<int64_t>(cols);
    CusparseFrontend::AddVariableForArguments<int64_t>(nnz);
    CusparseFrontend::AddDevicePointerForArguments(cooInd);
    CusparseFrontend::AddDevicePointerForArguments(cooValues);
    CusparseFrontend::AddVariableForArguments<cusparseIndexType_t>(cooIdxType);
    CusparseFrontend::AddVariableForArguments<cusparseIndexBase_t>(idxBase);
    CusparseFrontend::AddVariableForArguments<cudaDataType>(valueType);
    CusparseFrontend::AddHostPointerForArguments(spMatDescr);
    CusparseFrontend::Execute("cusparseCreateCooAoS");
    if (CusparseFrontend::Success()) {
        *spMatDescr = *(CusparseFrontend::GetOutputHostPointer<cusparseSpMatDescr_t>());
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseCreateCsr(cusparseSpMatDescr_t* spMatDescr, int64_t rows, int64_t cols, int64_t nnz, void* csrRowOffsets, void* csrColInd, void* csrValues, cusparseIndexType_t csrRowOffsetsType, cusparseIndexType_t csrColIndType, cusparseIndexBase_t idxBase, cudaDataType valueType) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<int64_t>(rows);
    CusparseFrontend::AddVariableForArguments<int64_t>(cols);
    CusparseFrontend::AddVariableForArguments<int64_t>(nnz);
    CusparseFrontend::AddDevicePointerForArguments(csrRowOffsets);
    CusparseFrontend::AddDevicePointerForArguments(csrColInd);
    CusparseFrontend::AddDevicePointerForArguments(csrValues);
    CusparseFrontend::AddVariableForArguments<cusparseIndexType_t>(csrRowOffsetsType);
    CusparseFrontend::AddVariableForArguments<cusparseIndexType_t>(csrColIndType);
    CusparseFrontend::AddVariableForArguments<cusparseIndexBase_t>(idxBase);
    CusparseFrontend::AddVariableForArguments<cudaDataType>(valueType);
    CusparseFrontend::AddHostPointerForArguments(spMatDescr);
    CusparseFrontend::Execute("cusparseCreateCsr");
    if (CusparseFrontend::Success()) {
        *spMatDescr = *(CusparseFrontend::GetOutputHostPointer<cusparseSpMatDescr_t>());
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseCreateCsc(cusparseSpMatDescr_t* spMatDescr, int64_t rows, int64_t cols, int64_t nnz, void* cscColOffsets, void* cscRowInd, void* cscValues, cusparseIndexType_t cscColOffsetsType, cusparseIndexType_t cscRowIndType, cusparseIndexBase_t idxBase, cudaDataType valueType) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<int64_t>(rows);
    CusparseFrontend::AddVariableForArguments<int64_t>(cols);
    CusparseFrontend::AddVariableForArguments<int64_t>(nnz);
    CusparseFrontend::AddDevicePointerForArguments(cscColOffsets);
    CusparseFrontend::AddDevicePointerForArguments(cscRowInd);
    CusparseFrontend::AddDevicePointerForArguments(cscValues);
    CusparseFrontend::AddVariableForArguments<cusparseIndexType_t>(cscColOffsetsType);
    CusparseFrontend::AddVariableForArguments<cusparseIndexType_t>(cscRowIndType);
    CusparseFrontend::AddVariableForArguments<cusparseIndexBase_t>(idxBase);
    CusparseFrontend::AddVariableForArguments<cudaDataType>(valueType);
    CusparseFrontend::AddHostPointerForArguments(spMatDescr);
    CusparseFrontend::Execute("cusparseCreateCsc");
    if (CusparseFrontend::Success()) {
        *spMatDescr = *(CusparseFrontend::GetOutputHostPointer<cusparseSpMatDescr_t>());
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseCreateBlockedEll(cusparseSpMatDescr_t* spMatDescr, int64_t rows, int64_t cols, int64_t ellBlockSize, int64_t ellCols, void* ellColInd, void* ellValue, cusparseIndexType_t ellIdxType, cusparseIndexBase_t idxBase, cudaDataType valueType) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<int64_t>(rows);
    CusparseFrontend::AddVariableForArguments<int64_t>(cols);
    CusparseFrontend::AddVariableForArguments<int64_t>(ellBlockSize);
    CusparseFrontend::AddVariableForArguments<int64_t>(ellCols);
    CusparseFrontend::AddDevicePointerForArguments(ellColInd);
    CusparseFrontend::AddDevicePointerForArguments(ellValue);
    CusparseFrontend::AddVariableForArguments<cusparseIndexType_t>(ellIdxType);
    CusparseFrontend::AddVariableForArguments<cusparseIndexBase_t>(idxBase);
    CusparseFrontend::AddVariableForArguments<cudaDataType>(valueType);
    CusparseFrontend::AddHostPointerForArguments(spMatDescr);
    CusparseFrontend::Execute("cusparseCreateBlockedEll");
    if (CusparseFrontend::Success()) {
        *spMatDescr = *(CusparseFrontend::GetOutputHostPointer<cusparseSpMatDescr_t>());
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseDestroySpMat(cusparseSpMatDescr_t spMatDescr) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) spMatDescr);
    CusparseFrontend::Execute("cusparseDestroySpMat");
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseCooGet(cusparseSpMatDescr_t spMatDescr, int64_t* rows, int64_t* cols, int64_t* nnz, void** cooRowInd, void** cooColInd, void** cooValues, cusparseIndexType_t* idxType, cusparseIndexBase_t* idxBase, cudaDataType* valueType) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)spMatDescr);
    CusparseFrontend::Execute("cusparseCooGet");
    if (CusparseFrontend::Success()) {
        *rows = (int64_t) CusparseFrontend::GetOutputVariable<size_t>();
        *cols = (int64_t) CusparseFrontend::GetOutputVariable<size_t>();
        *nnz = (int64_t) CusparseFrontend::GetOutputVariable<size_t>();
        *cooRowInd = (void *) CusparseFrontend::GetOutputDevicePointer();
        *cooColInd = (void *) CusparseFrontend::GetOutputDevicePointer();
        *cooValues = (void *) CusparseFrontend::GetOutputDevicePointer();
        *idxType = CusparseFrontend::GetOutputVariable<cusparseIndexType_t>();
        *idxBase = CusparseFrontend::GetOutputVariable<cusparseIndexBase_t>();
        *valueType = CusparseFrontend::GetOutputVariable<cudaDataType>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseCooAoSGet(cusparseSpMatDescr_t spMatDescr, int64_t* rows, int64_t* cols, int64_t* nnz, void** cooInd, void** cooValues, cusparseIndexType_t* idxType, cusparseIndexBase_t* idxBase, cudaDataType* valueType) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)spMatDescr);
    CusparseFrontend::Execute("cusparseCooAoSGet");
    if (CusparseFrontend::Success()) {
        *rows = (int64_t) CusparseFrontend::GetOutputVariable<size_t>();
        *cols = (int64_t) CusparseFrontend::GetOutputVariable<size_t>();
        *nnz = (int64_t) CusparseFrontend::GetOutputVariable<size_t>();
        *cooInd = (void *) CusparseFrontend::GetOutputDevicePointer();
        *cooValues = (void *) CusparseFrontend::GetOutputDevicePointer();
        *idxType = CusparseFrontend::GetOutputVariable<cusparseIndexType_t>();
        *idxBase = CusparseFrontend::GetOutputVariable<cusparseIndexBase_t>();
        *valueType = CusparseFrontend::GetOutputVariable<cudaDataType>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseCsrGet(cusparseSpMatDescr_t spMatDescr, int64_t* rows, int64_t* cols, int64_t* nnz, void** csrRowOffsets, void** csrColInd, void** csrValues, cusparseIndexType_t* csrRowOffsetsType, cusparseIndexType_t* csrColIndType, cusparseIndexBase_t* idxBase, cudaDataType* valueType) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)spMatDescr);
    CusparseFrontend::Execute("cusparseCsrGet");
    if (CusparseFrontend::Success()) {
        *rows = (int64_t) CusparseFrontend::GetOutputVariable<size_t>();
        *cols = (int64_t) CusparseFrontend::GetOutputVariable<size_t>();
        *nnz = (int64_t) CusparseFrontend::GetOutputVariable<size_t>();
        *csrRowOffsets = (void *) CusparseFrontend::GetOutputDevicePointer();
        *csrColInd = (void *) CusparseFrontend::GetOutputDevicePointer();
        *csrValues = (void *) CusparseFrontend::GetOutputDevicePointer();
        *csrRowOffsetsType = CusparseFrontend::GetOutputVariable<cusparseIndexType_t>();
        *csrColIndType = CusparseFrontend::GetOutputVariable<cusparseIndexType_t>();
        *idxBase = CusparseFrontend::GetOutputVariable<cusparseIndexBase_t>();
        *valueType = CusparseFrontend::GetOutputVariable<cudaDataType>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseCsrSetPointers(cusparseSpMatDescr_t spMatDescr, void* csrRowOffsets, void* csrColInd, void* csrValues) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)spMatDescr);
    CusparseFrontend::AddDevicePointerForArguments(csrRowOffsets);
    CusparseFrontend::AddDevicePointerForArguments(csrColInd);
    CusparseFrontend::AddDevicePointerForArguments(csrValues);
    CusparseFrontend::Execute("cusparseCsrSetPointers");
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseCscSetPointers(cusparseSpMatDescr_t spMatDescr, void* cscColOffsets, void* cscRowInd, void* cscValues) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)spMatDescr);
    CusparseFrontend::AddDevicePointerForArguments(cscColOffsets);
    CusparseFrontend::AddDevicePointerForArguments(cscRowInd);
    CusparseFrontend::AddDevicePointerForArguments(cscValues);
    CusparseFrontend::Execute("cusparseCscSetPointers");
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseCooSetPointers(cusparseSpMatDescr_t spMatDescr, void* cooRows, void* cooColumns, void* cooValues) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)spMatDescr);
    CusparseFrontend::AddDevicePointerForArguments(cooRows);
    CusparseFrontend::AddDevicePointerForArguments(cooColumns);
    CusparseFrontend::AddDevicePointerForArguments(cooValues);
    CusparseFrontend::Execute("cusparseCooSetPointers");
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseBlockedEllGet(cusparseSpMatDescr_t spMatDescr, int64_t* rows, int64_t* cols, int64_t* ellBlockSize, int64_t* ellCols, void** ellColInd, void** ellValue, cusparseIndexType_t* ellIdxType, cusparseIndexBase_t* idxBase, cudaDataType* valueType) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)spMatDescr);
    CusparseFrontend::Execute("cusparseBlockedEllGet");
    if (CusparseFrontend::Success()) {
        *rows = (int64_t) CusparseFrontend::GetOutputVariable<size_t>();
        *cols = (int64_t) CusparseFrontend::GetOutputVariable<size_t>();
        *ellBlockSize = (int64_t) CusparseFrontend::GetOutputVariable<size_t>();
        *ellCols = (int64_t) CusparseFrontend::GetOutputVariable<size_t>();
        *ellColInd = (void *) CusparseFrontend::GetOutputDevicePointer();
        *ellValue = (void *) CusparseFrontend::GetOutputDevicePointer();
        *ellIdxType = CusparseFrontend::GetOutputVariable<cusparseIndexType_t>();
        *idxBase = CusparseFrontend::GetOutputVariable<cusparseIndexBase_t>();
        *valueType = CusparseFrontend::GetOutputVariable<cudaDataType>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseSpMatGetSize(cusparseSpMatDescr_t spMatDescr, int64_t* rows, int64_t* cols, int64_t* nnz) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)spMatDescr);
    CusparseFrontend::Execute("cusparseSpMatGetSize");
    if (CusparseFrontend::Success()) {
        *rows = CusparseFrontend::GetOutputVariable<int64_t>();
        *cols = CusparseFrontend::GetOutputVariable<int64_t>();
        *nnz = CusparseFrontend::GetOutputVariable<int64_t>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseSpMatGetFormat(cusparseSpMatDescr_t spMatDescr, cusparseFormat_t* format) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)spMatDescr);
    CusparseFrontend::Execute("cusparseSpMatGetFormat");
    if (CusparseFrontend::Success()) {
        *format = CusparseFrontend::GetOutputVariable<cusparseFormat_t>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseSpMatGetIndexBase(cusparseSpMatDescr_t spMatDescr, cusparseIndexBase_t* idxBase) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)spMatDescr);
    CusparseFrontend::Execute("cusparseSpMatGetIndexBase");
    if (CusparseFrontend::Success()) {
        *idxBase = CusparseFrontend::GetOutputVariable<cusparseIndexBase_t>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseSpMatGetValues(cusparseSpMatDescr_t spMatDescr, void** values) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)spMatDescr);
    CusparseFrontend::Execute("cusparseSpMatGetValues");
    if (CusparseFrontend::Success()) {
        *values = (void *) CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseSpMatSetValues(cusparseSpMatDescr_t spMatDescr, void* values) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)spMatDescr);
    CusparseFrontend::AddDevicePointerForArguments(values);
    CusparseFrontend::Execute("cusparseSpMatSetValues");
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseSpMatGetStridedBatch(cusparseSpMatDescr_t spMatDescr, int* batchCount) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)spMatDescr);
    CusparseFrontend::Execute("cusparseSpMatGetStridedBatch");
    if (CusparseFrontend::Success()) {
        *batchCount = CusparseFrontend::GetOutputVariable<int>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseSpMatSetStridedBatch(cusparseSpMatDescr_t spMatDescr, int batchCount) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)spMatDescr);
    CusparseFrontend::AddVariableForArguments<int>(batchCount);
    CusparseFrontend::Execute("cusparseSpMatSetStridedBatch");
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseCooSetStridedBatch(cusparseSpMatDescr_t spMatDescr, int batchCount, int64_t batchStride) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)spMatDescr);
    CusparseFrontend::AddVariableForArguments<int>(batchCount);
    CusparseFrontend::AddVariableForArguments<int64_t>(batchStride);
    CusparseFrontend::Execute("cusparseCooSetStridedBatch");
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseCsrSetStridedBatch(cusparseSpMatDescr_t spMatDescr, int batchCount, int64_t offsetsBatchStride, int64_t columnsValuesBatchStride) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)spMatDescr);
    CusparseFrontend::AddVariableForArguments<int>(batchCount);
    CusparseFrontend::AddVariableForArguments<int64_t>(offsetsBatchStride);
    CusparseFrontend::AddVariableForArguments<int64_t>(columnsValuesBatchStride);
    CusparseFrontend::Execute("cusparseCsrSetStridedBatch");
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseSpMatGetAttribute(cusparseSpMatDescr_t spMatDescr, cusparseSpMatAttribute_t attribute, void* data, size_t dataSize) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)spMatDescr);
    CusparseFrontend::AddVariableForArguments<cusparseSpMatAttribute_t>(attribute);
    CusparseFrontend::AddVariableForArguments<size_t>(dataSize);
    CusparseFrontend::Execute("cusparseSpMatGetAttribute");
    if (CusparseFrontend::Success()) {
        data = (void *) CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseSpMatSetAttribute(cusparseSpMatDescr_t spMatDescr, cusparseSpMatAttribute_t attribute, void* data, size_t dataSize) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<cusparseSpMatAttribute_t>(attribute);
    CusparseFrontend::AddDevicePointerForArguments(data);
    CusparseFrontend::AddVariableForArguments<size_t>(dataSize);
    CusparseFrontend::Execute("cusparseSpMatSetAttribute");
    if (CusparseFrontend::Success()) {
        spMatDescr = (cusparseSpMatDescr_t) CusparseFrontend::GetOutputHostPointer<size_t>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseCreateDnVec(cusparseDnVecDescr_t* dnVecDescr, int64_t size, void* values, cudaDataType valueType) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<int64_t>(size);
    CusparseFrontend::AddDevicePointerForArguments(values);
    CusparseFrontend::AddVariableForArguments<cudaDataType>(valueType);
    CusparseFrontend::AddHostPointerForArguments(dnVecDescr);
    //CusparseFrontend::AddVariableForArguments(dnVecDescr);
    //CusparseFrontend::AddHostPointerForArguments(dnVecDescr);
    //CusparseFrontend::AddHostPointerForArguments<cusparseDnVecDescr_t>(dnVecDescr);
    CusparseFrontend::Execute("cusparseCreateDnVec");
    if (CusparseFrontend::Success()) {
        //*dnVecDescr = *(CusparseFrontend::GetOutputHostPointer<cusparseDnVecDescr_t>());
        *dnVecDescr = (cusparseDnVecDescr_t) CusparseFrontend::GetOutputVariable<size_t>();
        //printf("\n\tFE - CreateDnVec - dnVecDescr pointer: %p\n", *dnVecDescr);
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseDestroyDnVec(cusparseDnVecDescr_t dnVecDescr) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) dnVecDescr);
    CusparseFrontend::Execute("cusparseDestroyDnVec");
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseDnVecGet(cusparseDnVecDescr_t dnVecDescr, int64_t* size, void** values, cudaDataType* valueType) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)dnVecDescr);
    CusparseFrontend::Execute("cusparseDnVecGet");
    if (CusparseFrontend::Success()) {
        *size = (int64_t) CusparseFrontend::GetOutputVariable<size_t>();
        *values = (void *) CusparseFrontend::GetOutputDevicePointer();
        *valueType = CusparseFrontend::GetOutputVariable<cudaDataType>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseDnVecGetValues(cusparseDnVecDescr_t dnVecDescr, void** values) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)dnVecDescr);
    CusparseFrontend::Execute("cusparseDnVecGetValues");
    if (CusparseFrontend::Success()) {
        *values = (void *) CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseDnVecSetValues(cusparseDnVecDescr_t dnVecDescr, void* values) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)dnVecDescr);
    CusparseFrontend::AddDevicePointerForArguments(values);
    CusparseFrontend::Execute("cusparseDnVecSetValues");
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseSpMV_bufferSize(cusparseHandle_t handle, cusparseOperation_t opA, const void* alpha, cusparseSpMatDescr_t matA,
                                                                cusparseDnVecDescr_t vecX, const void* beta, cusparseDnVecDescr_t vecY, cudaDataType computeType, cusparseSpMVAlg_t alg, size_t* bufferSize) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<cusparseOperation_t>(opA);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) matA);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) vecX);
    //printf("\n\tFE - SpMV_bufferSize - vecX pointer: %p\n", vecX);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) vecY);
    //printf("\n\tFE - SpMV_bufferSize - vecY pointer: %p\n", vecY);
    CusparseFrontend::AddVariableForArguments<cudaDataType>(computeType);
    CusparseFrontend::AddVariableForArguments<cusparseSpMVAlg_t>(alg);
    //CusparseFrontend::AddHostPointerForArguments(bufferSize);
    switch(computeType){
        case CUDA_R_32F:
            //float
            CusparseFrontend::AddVariableForArguments(*(float *)alpha);
            CusparseFrontend::AddVariableForArguments(*(float *)beta);
            break;
        case CUDA_R_64F:
            //double
            CusparseFrontend::AddVariableForArguments(*(double *)alpha);
            CusparseFrontend::AddVariableForArguments(*(double *)beta);
            break;
        case CUDA_C_32F:
            //cuComplex
            CusparseFrontend::AddVariableForArguments(*(cuComplex *)alpha);
            CusparseFrontend::AddVariableForArguments(*(cuComplex *)beta);
            break;
        case CUDA_C_64F:
            //cuDoubleComplex
            CusparseFrontend::AddVariableForArguments(*(cuDoubleComplex *)alpha);
            CusparseFrontend::AddVariableForArguments(*(cuDoubleComplex *)beta);
            break;
        default:
            throw "Type not supported by GVirtus!";
    }
    CusparseFrontend::Execute("cusparseSpMV_bufferSize");
    if (CusparseFrontend::Success()) {
        *bufferSize = *(CusparseFrontend::GetOutputHostPointer<size_t>());
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseSpMV(cusparseHandle_t handle, cusparseOperation_t opA, const void* alpha, cusparseSpMatDescr_t matA, cusparseDnVecDescr_t vecX, const void* beta, cusparseDnVecDescr_t vecY, cudaDataType computeType, cusparseSpMVAlg_t alg, void* externalBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<cusparseOperation_t>(opA);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) matA);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) vecX);
    /*
    float hX[4];
    cudaMemcpy(hX, vecX, 4 * sizeof(float), cudaMemcpyDeviceToHost);
    printf("\tSpMV - vecX: [");
    for (int i = 0; i < 4; i++) {
        printf("%f", hX[i]);
        if (i < 3) {
            printf(", ");
        }
    }
    printf("]\n");
    */
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) vecY);
    /*
    float hY[4];
    cudaMemcpy(hY, vecY, 4 * sizeof(float), cudaMemcpyDeviceToHost);
    printf("\tSpMV - BEFORE - vecY: [");
    for (int i = 0; i < 4; i++) {
        printf("%f", hY[i]);
        if (i < 3) {
            printf(", ");
        }
    }
    printf("]\n");
    */
    CusparseFrontend::AddVariableForArguments<cudaDataType>(computeType);
    CusparseFrontend::AddVariableForArguments<cusparseSpMVAlg_t>(alg);
    CusparseFrontend::AddDevicePointerForArguments(externalBuffer);
    switch(computeType){
        case CUDA_R_32F:
            //float
            CusparseFrontend::AddVariableForArguments(*(float *)alpha);
            CusparseFrontend::AddVariableForArguments(*(float *)beta);
            break;
        case CUDA_R_64F:
            //double
            CusparseFrontend::AddVariableForArguments(*(double *)alpha);
            CusparseFrontend::AddVariableForArguments(*(double *)beta);
            break;
        case CUDA_C_32F:
            //cuComplex
            CusparseFrontend::AddVariableForArguments(*(cuComplex *)alpha);
            CusparseFrontend::AddVariableForArguments(*(cuComplex *)beta);
            break;
        case CUDA_C_64F:
            //cuDoubleComplex
            CusparseFrontend::AddVariableForArguments(*(cuDoubleComplex *)alpha);
            CusparseFrontend::AddVariableForArguments(*(cuDoubleComplex *)beta);
            break;
        default:
            throw "Type not supported by GVirtus!";
    }
    CusparseFrontend::Execute("cusparseSpMV");
    if (CusparseFrontend::Success()) {
        vecY = CusparseFrontend::GetOutputVariable<cusparseDnVecDescr_t>();
        /*
        cudaMemcpy(hY, vecY, 4 * sizeof(float), cudaMemcpyDeviceToHost);
        printf("\tSpMV - AFTER - vecY: [");
        for (int i = 0; i < 4; i++) {
            printf("%f", hY[i]);
            if (i < 3) {
                printf(", ");
            }
        }
        printf("]\n");
        */
        //printf("\nvecY address: %d\n", vecY);
    }
    return CusparseFrontend::GetExitCode();
}