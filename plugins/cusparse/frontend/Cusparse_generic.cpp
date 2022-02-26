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

extern "C" cusparseStatus_t CUSPARSEAPI cusparseSpMatSetAttribute(cusparseSpMatDescr_t spMatDescr, cusparseSpMatAttribute_t attribute, void *data, size_t dataSize) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)spMatDescr);
    CusparseFrontend::AddVariableForArguments<cusparseSpMatAttribute_t>(attribute);
    CusparseFrontend::AddVariableForArguments<size_t>(dataSize);
    //CusparseFrontend::AddDevicePointerForArguments(data);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)data);
    CusparseFrontend::Execute("cusparseSpMatSetAttribute");
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

extern "C" cusparseStatus_t CUSPARSEAPI cusparseCreateDnMat(cusparseDnMatDescr_t* dnMatDescr, int64_t rows, int64_t cols, int64_t ld, void* values, cudaDataType valueType, cusparseOrder_t order) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<int64_t>(rows);
    CusparseFrontend::AddVariableForArguments<int64_t>(cols);
    CusparseFrontend::AddVariableForArguments<int64_t>(ld);
    CusparseFrontend::AddDevicePointerForArguments(values);
    CusparseFrontend::AddVariableForArguments<cudaDataType>(valueType);
    CusparseFrontend::AddVariableForArguments<cusparseOrder_t>(order);
    CusparseFrontend::AddHostPointerForArguments(dnMatDescr);
    CusparseFrontend::Execute("cusparseCreateDnMat");
    if (CusparseFrontend::Success()) {
        *dnMatDescr = *(CusparseFrontend::GetOutputHostPointer<cusparseDnMatDescr_t>());
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseDestroyDnMat(cusparseDnMatDescr_t dnMatDescr) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) dnMatDescr);
    CusparseFrontend::Execute("cusparseDestroyDnMat");
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseDnMatGet(cusparseDnMatDescr_t dnMatDescr, int64_t* rows, int64_t* cols, int64_t* ld, void** values, cudaDataType* type, cusparseOrder_t* order) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)dnMatDescr);
    CusparseFrontend::Execute("cusparseDnMatGet");
    if (CusparseFrontend::Success()) {
        *rows = (int64_t) CusparseFrontend::GetOutputVariable<size_t>();
        *cols = (int64_t) CusparseFrontend::GetOutputVariable<size_t>();
        *ld = (int64_t) CusparseFrontend::GetOutputVariable<size_t>();
        *values = (void *) CusparseFrontend::GetOutputDevicePointer();
        *type = CusparseFrontend::GetOutputVariable<cudaDataType>();
        *order = CusparseFrontend::GetOutputVariable<cusparseOrder_t>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseDnMatGetValues(cusparseDnMatDescr_t dnMatDescr, void** values) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)dnMatDescr);
    CusparseFrontend::Execute("cusparseDnMatGetValues");
    if (CusparseFrontend::Success()) {
        *values = (void *) CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseDnMatSetValues(cusparseDnMatDescr_t dnMatDescr, void* values) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)dnMatDescr);
    CusparseFrontend::AddDevicePointerForArguments(values);
    CusparseFrontend::Execute("cusparseDnMatSetValues");
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseDnMatGetStridedBatch(cusparseDnMatDescr_t dnMatDescr, int* batchCount, int64_t* batchStride) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)dnMatDescr);
    CusparseFrontend::Execute("cusparseDnMatGetStridedBatch");
    if (CusparseFrontend::Success()) {
        *batchCount = CusparseFrontend::GetOutputVariable<int>();
        *batchStride = CusparseFrontend::GetOutputVariable<int64_t>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseDnMatSetStridedBatch(cusparseDnMatDescr_t dnMatDescr, int batchCount, int64_t batchStride) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)dnMatDescr);
    CusparseFrontend::AddVariableForArguments<int>(batchCount);
    CusparseFrontend::AddVariableForArguments<int64_t>(batchStride);
    CusparseFrontend::Execute("cusparseDnMatSetStridedBatch");
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseSparseToDense_bufferSize(cusparseHandle_t handle, cusparseSpMatDescr_t matA, cusparseDnMatDescr_t matB, cusparseSparseToDenseAlg_t alg, size_t* bufferSize) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) matA);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) matB);
    CusparseFrontend::AddVariableForArguments<cusparseSparseToDenseAlg_t>(alg);
    CusparseFrontend::Execute("cusparseSparseToDense_bufferSize");
    if (CusparseFrontend::Success()) {
        *bufferSize = *(CusparseFrontend::GetOutputHostPointer<size_t>());
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseSparseToDense(cusparseHandle_t handle, cusparseSpMatDescr_t matA, cusparseDnMatDescr_t matB, cusparseSparseToDenseAlg_t alg, void* buffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) matA);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) matB);
    CusparseFrontend::AddVariableForArguments<cusparseSparseToDenseAlg_t>(alg);
    CusparseFrontend::AddDevicePointerForArguments(buffer);
    CusparseFrontend::Execute("cusparseSparseToDense");
    if (CusparseFrontend::Success()) {
        matB = CusparseFrontend::GetOutputVariable<cusparseDnMatDescr_t>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseDenseToSparse_bufferSize(cusparseHandle_t handle, cusparseDnMatDescr_t matA, cusparseSpMatDescr_t matB, cusparseDenseToSparseAlg_t alg, size_t* bufferSize) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) matA);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) matB);
    CusparseFrontend::AddVariableForArguments<cusparseDenseToSparseAlg_t>(alg);
    CusparseFrontend::Execute("cusparseDenseToSparse_bufferSize");
    if (CusparseFrontend::Success()) {
        *bufferSize = *(CusparseFrontend::GetOutputHostPointer<size_t>());
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseDenseToSparse_analysis(cusparseHandle_t handle, cusparseDnMatDescr_t matA, cusparseSpMatDescr_t matB, cusparseDenseToSparseAlg_t alg, void* buffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) matA);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) matB);
    CusparseFrontend::AddVariableForArguments<cusparseDenseToSparseAlg_t>(alg);
    CusparseFrontend::AddDevicePointerForArguments(buffer);
    CusparseFrontend::Execute("cusparseDenseToSparse_analysis");
    if (CusparseFrontend::Success()) {
        matB = CusparseFrontend::GetOutputVariable<cusparseSpMatDescr_t>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseDenseToSparse_convert(cusparseHandle_t handle, cusparseDnMatDescr_t matA, cusparseSpMatDescr_t matB, cusparseDenseToSparseAlg_t alg, void* buffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) matA);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) matB);
    CusparseFrontend::AddVariableForArguments<cusparseDenseToSparseAlg_t>(alg);
    CusparseFrontend::AddDevicePointerForArguments(buffer);
    CusparseFrontend::Execute("cusparseDenseToSparse_convert");
    if (CusparseFrontend::Success()) {
        matB = CusparseFrontend::GetOutputVariable<cusparseSpMatDescr_t>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseAxpby(cusparseHandle_t handle, const void* alpha, cusparseSpVecDescr_t vecX, const void* beta, cusparseDnVecDescr_t vecY) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddHostPointerForArguments(const_cast<float *>((float *)alpha));
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) vecX);
    CusparseFrontend::AddHostPointerForArguments(const_cast<float *>((float *)beta));
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) vecY);
    CusparseFrontend::Execute("cusparseAxpby");
    if (CusparseFrontend::Success()) {
        vecY = CusparseFrontend::GetOutputVariable<cusparseDnVecDescr_t>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseGather(cusparseHandle_t handle, cusparseDnVecDescr_t vecY, cusparseSpVecDescr_t vecX) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) vecY);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) vecX);
    CusparseFrontend::Execute("cusparseGather");
    if (CusparseFrontend::Success()) {
        vecX = CusparseFrontend::GetOutputVariable<cusparseSpVecDescr_t>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseScatter(cusparseHandle_t handle, cusparseSpVecDescr_t vecX, cusparseDnVecDescr_t vecY) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) vecX);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) vecY);
    CusparseFrontend::Execute("cusparseScatter");
    if (CusparseFrontend::Success()) {
        vecY = CusparseFrontend::GetOutputVariable<cusparseDnVecDescr_t>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseRot(cusparseHandle_t handle, const void* c_coeff, const void* s_coeff, cusparseSpVecDescr_t vecX, cusparseDnVecDescr_t vecY) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddHostPointerForArguments(const_cast<float *>((float *)c_coeff));
    CusparseFrontend::AddHostPointerForArguments(const_cast<float *>((float *)s_coeff));
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) vecX);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) vecY);
    CusparseFrontend::Execute("cusparseRot");
    if (CusparseFrontend::Success()) {
        vecX = CusparseFrontend::GetOutputVariable<cusparseSpVecDescr_t>();
        vecY = CusparseFrontend::GetOutputVariable<cusparseDnVecDescr_t>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseSpVV_bufferSize(cusparseHandle_t handle, cusparseOperation_t opX, cusparseSpVecDescr_t vecX, cusparseDnVecDescr_t vecY, const void* result, cudaDataType computeType, size_t* bufferSize) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<cusparseOperation_t>(opX);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) vecX);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) vecY);
    CusparseFrontend::AddDevicePointerForArguments(result);
    CusparseFrontend::AddVariableForArguments<cudaDataType>(computeType);
    CusparseFrontend::Execute("cusparseSpVV_bufferSize");
    if (CusparseFrontend::Success()) {
        *bufferSize = *(CusparseFrontend::GetOutputHostPointer<size_t>());
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseSpVV(cusparseHandle_t handle, cusparseOperation_t opX, cusparseSpVecDescr_t vecX, cusparseDnVecDescr_t vecY, void* result, cudaDataType computeType, void* buffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<cusparseOperation_t>(opX);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) vecX);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) vecY);
    CusparseFrontend::AddVariableForArguments<cudaDataType>(computeType);
    CusparseFrontend::AddDevicePointerForArguments(buffer);
    //printf("%p\n", result);
    //printf("%f\n", *((float*)result));
    //CusparseFrontend::AddDevicePointerForArguments((float*)result);
    //CusparseFrontend::AddHostPointerForArguments((float*)result);
    CusparseFrontend::Execute("cusparseSpVV");
    if (CusparseFrontend::Success()) {
        if (computeType == CUDA_R_32F) {
            //float

            //result = (float*) CusparseFrontend::GetOutputDevicePointer();

            //float* res = (float*) CusparseFrontend::GetOutputDevicePointer();
            //printf("%p\n", res);
            //printf("%f\n", *res);
            //result = (void*)res;

            float resF = CusparseFrontend::GetOutputVariable<float>();
            result = &resF;

            //printf("%p\n", result);
            //printf("%f\n", *((float*)result));
        } else if (computeType == CUDA_R_64F) {
            //double
            double resD = CusparseFrontend::GetOutputVariable<double>();
            result = &resD;
        } else if (computeType == CUDA_C_32F) {
            //cuComplex
            cuComplex resC = CusparseFrontend::GetOutputVariable<cuComplex>();
            result = &resC;
        } else if (computeType == CUDA_C_64F) {
            //cuDoubleComplex
            cuDoubleComplex resZ = CusparseFrontend::GetOutputVariable<cuDoubleComplex>();
            result = &resZ;
        } else {
            throw "Type not supported by GVirtus!";
        }
    }
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

extern "C" cusparseStatus_t CUSPARSEAPI cusparseSpSV_createDescr(cusparseSpSVDescr_t* spsvDescr) {
    CusparseFrontend::Prepare();
    CusparseFrontend::Execute("cusparseSpSV_createDescr");
    if (CusparseFrontend::Success()) {
        *spsvDescr = *(CusparseFrontend::GetOutputHostPointer<cusparseSpSVDescr_t>());
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseSpSV_destroyDescr(cusparseSpSVDescr_t spsvDescr) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) spsvDescr);
    CusparseFrontend::Execute("cusparseSpSV_destroyDescr");
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseSpSV_bufferSize(cusparseHandle_t handle, cusparseOperation_t opA, const void* alpha, cusparseSpMatDescr_t matA,
                                                                cusparseDnVecDescr_t vecX, cusparseDnVecDescr_t vecY, cudaDataType computeType, cusparseSpSVAlg_t alg,
                                                                cusparseSpSVDescr_t  spsvDescr, size_t* bufferSize) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<cusparseOperation_t>(opA);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) matA);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) vecX);
    //printf("\n\tFE - SpMV_bufferSize - vecX pointer: %p\n", vecX);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) vecY);
    //printf("\n\tFE - SpMV_bufferSize - vecY pointer: %p\n", vecY);
    CusparseFrontend::AddVariableForArguments<cudaDataType>(computeType);
    CusparseFrontend::AddVariableForArguments<cusparseSpSVAlg_t>(alg);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)spsvDescr);
    //CusparseFrontend::AddHostPointerForArguments(bufferSize);
    switch(computeType){
        case CUDA_R_32F:
            //float
            CusparseFrontend::AddVariableForArguments(*(float *)alpha);
            break;
        case CUDA_R_64F:
            //double
            CusparseFrontend::AddVariableForArguments(*(double *)alpha);
            break;
        case CUDA_C_32F:
            //cuComplex
            CusparseFrontend::AddVariableForArguments(*(cuComplex *)alpha);
            break;
        case CUDA_C_64F:
            //cuDoubleComplex
            CusparseFrontend::AddVariableForArguments(*(cuDoubleComplex *)alpha);
            break;
        default:
            throw "Type not supported by GVirtus!";
    }
    CusparseFrontend::Execute("cusparseSpSV_bufferSize");
    if (CusparseFrontend::Success()) {
        *bufferSize = *(CusparseFrontend::GetOutputHostPointer<size_t>());
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseSpSV_analysis(cusparseHandle_t handle, cusparseOperation_t opA, const void* alpha, cusparseSpMatDescr_t matA,
                                                              cusparseDnVecDescr_t vecX, cusparseDnVecDescr_t vecY, cudaDataType computeType, cusparseSpSVAlg_t alg,
                                                              cusparseSpSVDescr_t  spsvDescr, void* externalBuffer) {
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
    CusparseFrontend::AddVariableForArguments<cusparseSpSVAlg_t>(alg);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)spsvDescr);
    CusparseFrontend::AddDevicePointerForArguments(externalBuffer);
    switch(computeType){
        case CUDA_R_32F:
            //float
            CusparseFrontend::AddVariableForArguments(*(float *)alpha);
            break;
        case CUDA_R_64F:
            //double
            CusparseFrontend::AddVariableForArguments(*(double *)alpha);
            break;
        case CUDA_C_32F:
            //cuComplex
            CusparseFrontend::AddVariableForArguments(*(cuComplex *)alpha);
            break;
        case CUDA_C_64F:
            //cuDoubleComplex
            CusparseFrontend::AddVariableForArguments(*(cuDoubleComplex *)alpha);
            break;
        default:
            throw "Type not supported by GVirtus!";
    }
    CusparseFrontend::Execute("cusparseSpSV_analysis");
    if (CusparseFrontend::Success()) {
        vecY = CusparseFrontend::GetOutputVariable<cusparseDnVecDescr_t>();
        spsvDescr = CusparseFrontend::GetOutputVariable<cusparseSpSVDescr_t>();
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

extern "C" cusparseStatus_t CUSPARSEAPI cusparseSpSV_solve(cusparseHandle_t handle, cusparseOperation_t opA, const void* alpha, cusparseSpMatDescr_t matA,
                                                              cusparseDnVecDescr_t vecX, cusparseDnVecDescr_t vecY, cudaDataType computeType, cusparseSpSVAlg_t alg,
                                                              cusparseSpSVDescr_t  spsvDescr) {
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
    CusparseFrontend::AddVariableForArguments<cusparseSpSVAlg_t>(alg);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)spsvDescr);
    switch(computeType){
        case CUDA_R_32F:
            //float
            CusparseFrontend::AddVariableForArguments(*(float *)alpha);
            break;
        case CUDA_R_64F:
            //double
            CusparseFrontend::AddVariableForArguments(*(double *)alpha);
            break;
        case CUDA_C_32F:
            //cuComplex
            CusparseFrontend::AddVariableForArguments(*(cuComplex *)alpha);
            break;
        case CUDA_C_64F:
            //cuDoubleComplex
            CusparseFrontend::AddVariableForArguments(*(cuDoubleComplex *)alpha);
            break;
        default:
            throw "Type not supported by GVirtus!";
    }
    CusparseFrontend::Execute("cusparseSpSV_solve");
    if (CusparseFrontend::Success()) {
        vecY = CusparseFrontend::GetOutputVariable<cusparseDnVecDescr_t>();
        spsvDescr = CusparseFrontend::GetOutputVariable<cusparseSpSVDescr_t>();
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

extern "C" cusparseStatus_t CUSPARSEAPI cusparseSpMM_bufferSize(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, const void* alpha, cusparseSpMatDescr_t matA,
                                                                cusparseDnMatDescr_t matB, const void* beta, cusparseDnMatDescr_t matC, cudaDataType computeType, cusparseSpMMAlg_t alg, size_t* bufferSize) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<cusparseOperation_t>(opA);
    CusparseFrontend::AddVariableForArguments<cusparseOperation_t>(opB);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) matA);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) matB);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) matC);
    CusparseFrontend::AddVariableForArguments<cudaDataType>(computeType);
    CusparseFrontend::AddVariableForArguments<cusparseSpMMAlg_t>(alg);
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
    CusparseFrontend::Execute("cusparseSpMM_bufferSize");
    if (CusparseFrontend::Success()) {
        *bufferSize = *(CusparseFrontend::GetOutputHostPointer<size_t>());
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseSpMM_preprocess(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, const void* alpha, cusparseSpMatDescr_t matA,
                                                                cusparseDnMatDescr_t matB, const void* beta, cusparseDnMatDescr_t matC, cudaDataType computeType, cusparseSpMMAlg_t alg, void* externalBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<cusparseOperation_t>(opA);
    CusparseFrontend::AddVariableForArguments<cusparseOperation_t>(opB);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) matA);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) matB);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) matC);
    CusparseFrontend::AddVariableForArguments<cudaDataType>(computeType);
    CusparseFrontend::AddVariableForArguments<cusparseSpMMAlg_t>(alg);
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
    CusparseFrontend::Execute("cusparseSpMM_preprocess");
    if (CusparseFrontend::Success()) {
        matC = CusparseFrontend::GetOutputVariable<cusparseDnMatDescr_t>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseSpMM(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, const void* alpha, cusparseSpMatDescr_t matA,
                                                                cusparseDnMatDescr_t matB, const void* beta, cusparseDnMatDescr_t matC, cudaDataType computeType, cusparseSpMMAlg_t alg, void* externalBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<cusparseOperation_t>(opA);
    CusparseFrontend::AddVariableForArguments<cusparseOperation_t>(opB);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) matA);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) matB);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) matC);
    CusparseFrontend::AddVariableForArguments<cudaDataType>(computeType);
    CusparseFrontend::AddVariableForArguments<cusparseSpMMAlg_t>(alg);
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
    CusparseFrontend::Execute("cusparseSpMM");
    if (CusparseFrontend::Success()) {
        matC = CusparseFrontend::GetOutputVariable<cusparseDnMatDescr_t>();
    }
    return CusparseFrontend::GetExitCode();
}