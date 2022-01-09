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
        //spMatDescr = CusparseFrontend::GetOutputVariable<cusparseSpMatDescr_t*>();
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
        *dnVecDescr = *(CusparseFrontend::GetOutputHostPointer<cusparseDnVecDescr_t>());
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseDestroyDnVec(cusparseDnVecDescr_t dnVecDescr) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) dnVecDescr);
    CusparseFrontend::Execute("cusparseDestroyDnVec");
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseSpMV_bufferSize(cusparseHandle_t handle, cusparseOperation_t opA, const void* alpha, cusparseSpMatDescr_t matA, cusparseDnVecDescr_t vecX, const void* beta, cusparseDnVecDescr_t vecY, cudaDataType computeType, cusparseSpMVAlg_t alg, size_t* bufferSize) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<cusparseOperation_t>(opA);
    //printf("\nalpha address: %d\n", alpha);
    //printf("\nalpha value: %f\n", *(float*) alpha);
    CusparseFrontend::AddDevicePointerForArguments(alpha);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) matA);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) vecX);
    CusparseFrontend::AddDevicePointerForArguments(beta);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) vecY);
    CusparseFrontend::AddVariableForArguments<cudaDataType>(computeType);
    CusparseFrontend::AddVariableForArguments<cusparseSpMVAlg_t>(alg);
    CusparseFrontend::AddHostPointerForArguments(bufferSize);
    CusparseFrontend::Execute("cusparseSpMV_bufferSize");
    if (CusparseFrontend::Success()) {
        //printf("\nFE-PRE-bufferSize address: %d\n", bufferSize);
        *bufferSize = *(CusparseFrontend::GetOutputHostPointer<size_t>());
        //printf("\nFE-POST-bufferSize address: %d\n", bufferSize);
        //printf("\nbufferSize value: %d\n", *bufferSize);
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseSpMV(cusparseHandle_t handle, cusparseOperation_t opA, const void* alpha, cusparseSpMatDescr_t matA, cusparseDnVecDescr_t vecX, const void* beta, cusparseDnVecDescr_t vecY, cudaDataType computeType, cusparseSpMVAlg_t alg, void* externalBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<cusparseOperation_t>(opA);
    //printf("\nalpha address: %d\n", alpha);
    //printf("\nalpha value: %f\n", *(float*) alpha);
    CusparseFrontend::AddDevicePointerForArguments(alpha);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) matA);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) vecX);
    CusparseFrontend::AddDevicePointerForArguments(beta);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t) vecY);
    CusparseFrontend::AddVariableForArguments<cudaDataType>(computeType);
    CusparseFrontend::AddVariableForArguments<cusparseSpMVAlg_t>(alg);
    CusparseFrontend::AddDevicePointerForArguments(externalBuffer);
    CusparseFrontend::Execute("cusparseSpMV");
    if (CusparseFrontend::Success()) {
        vecY = CusparseFrontend::GetOutputVariable<cusparseDnVecDescr_t>();
        //printf("\nvecY address: %d\n", vecY);
    }
    return CusparseFrontend::GetExitCode();
}