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
 * Written by: Antonio Pilato <antonio.pilato001@studenti.uniparthenope.it>,
 *             Nicholas Piantadosi <nicholas.piantadosi@studenti.uniparthenope.it>,
 *              Department of Science andTechnology
 */

#include <iostream>
#include <cstdio>
#include <string>

#include "CusolverFrontend.h"

using namespace std;

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnXpotrf_bufferSize(cusolverDnHandle_t handle, cusolverDnParams_t params, cublasFillMode_t uplo, int64_t n, cudaDataType dataTypeA, const void *A, int64_t lda, cudaDataType computeType, size_t *workspaceInBytesOnDevice, size_t *workspaceInBytesOnHost) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) params);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<int64_t>(n);
    CusolverFrontend::AddVariableForArguments<cudaDataType_t>(dataTypeA);
    CusolverFrontend::AddVariableForArguments<int64_t>(lda);
    CusolverFrontend::AddVariableForArguments<cudaDataType_t>(computeType);
    switch(dataTypeA){
        case CUDA_R_32F:
            //float
            CusolverFrontend::AddDevicePointerForArguments((float *)A);
            break;
        case CUDA_R_64F:
            //double
            CusolverFrontend::AddDevicePointerForArguments((double *)A);
            break;
        case CUDA_C_32F:
            //cuComplex
            CusolverFrontend::AddDevicePointerForArguments((cuComplex *)A);
            break;
        case CUDA_C_64F:
            //cuDoubleComplex
            CusolverFrontend::AddDevicePointerForArguments((cuDoubleComplex *)A);
            break;
        default:
            throw "Type not supported by GVirtus!";
    }
    CusolverFrontend::Execute("cusolverDnXpotrf_bufferSize");
    if (CusolverFrontend::Success()) {
        *workspaceInBytesOnDevice = CusolverFrontend::GetOutputVariable<int>();
        *workspaceInBytesOnHost = CusolverFrontend::GetOutputVariable<int>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnXpotrf(cusolverDnHandle_t handle, cusolverDnParams_t params, cublasFillMode_t uplo, int64_t n, cudaDataType dataTypeA, void *A, int64_t lda, cudaDataType computeType, void *bufferOnDevice, size_t workspaceInBytesOnDevice, void *bufferOnHost, size_t workspaceInBytesOnHost, int *info) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) params);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<int64_t>(n);
    CusolverFrontend::AddVariableForArguments<cudaDataType_t>(dataTypeA);
    CusolverFrontend::AddVariableForArguments<int64_t>(lda);
    CusolverFrontend::AddVariableForArguments<cudaDataType_t>(computeType);
    CusolverFrontend::AddDevicePointerForArguments(bufferOnDevice);
    CusolverFrontend::AddVariableForArguments<size_t>(workspaceInBytesOnDevice);
    CusolverFrontend::AddHostPointerForArguments((size_t*)bufferOnHost);
    CusolverFrontend::AddVariableForArguments<size_t>(workspaceInBytesOnHost);
    CusolverFrontend::AddDevicePointerForArguments(info);
    switch(dataTypeA){
        case CUDA_R_32F:
            //float
            CusolverFrontend::AddDevicePointerForArguments((float *)A);
            break;
        case CUDA_R_64F:
            //double
            CusolverFrontend::AddDevicePointerForArguments((double *)A);
            break;
        case CUDA_C_32F:
            //cuComplex
            CusolverFrontend::AddDevicePointerForArguments((cuComplex *)A);
            break;
        case CUDA_C_64F:
            //cuDoubleComplex
            CusolverFrontend::AddDevicePointerForArguments((cuDoubleComplex *)A);
            break;
        default:
            throw "Type not supported by GVirtus!";
    }
    CusolverFrontend::Execute("cusolverDnXpotrf");
    if (CusolverFrontend::Success()) {
        info = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnXpotrs(cusolverDnHandle_t handle, cusolverDnParams_t params, cublasFillMode_t uplo, int64_t n, int64_t nrhs, cudaDataType dataTypeA, const void *A, int64_t lda, cudaDataType dataTypeB, void *B, int64_t ldb, int *info) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) params);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<int64_t>(n);
    CusolverFrontend::AddVariableForArguments<int64_t>(nrhs);
    CusolverFrontend::AddVariableForArguments<cudaDataType_t>(dataTypeA);
    CusolverFrontend::AddVariableForArguments<int64_t>(lda);
    CusolverFrontend::AddVariableForArguments<cudaDataType_t>(dataTypeB);
    CusolverFrontend::AddVariableForArguments<int64_t>(ldb);
    CusolverFrontend::AddDevicePointerForArguments(info);
    switch(dataTypeA){
        case CUDA_R_32F:
            //float
            CusolverFrontend::AddDevicePointerForArguments((float *)A);
            break;
        case CUDA_R_64F:
            //double
            CusolverFrontend::AddDevicePointerForArguments((double *)A);
            break;
        case CUDA_C_32F:
            //cuComplex
            CusolverFrontend::AddDevicePointerForArguments((cuComplex *)A);
            break;
        case CUDA_C_64F:
            //cuDoubleComplex
            CusolverFrontend::AddDevicePointerForArguments((cuDoubleComplex *)A);
            break;
        default:
            throw "Type not supported by GVirtus!";
    }
    switch(dataTypeB){
        case CUDA_R_32F:
            //float
            CusolverFrontend::AddDevicePointerForArguments((float *)B);
            break;
        case CUDA_R_64F:
            //double
            CusolverFrontend::AddDevicePointerForArguments((double *)B);
            break;
        case CUDA_C_32F:
            //cuComplex
            CusolverFrontend::AddDevicePointerForArguments((cuComplex *)B);
            break;
        case CUDA_C_64F:
            //cuDoubleComplex
            CusolverFrontend::AddDevicePointerForArguments((cuDoubleComplex *)B);
            break;
        default:
            throw "Type not supported by GVirtus!";
    }
    CusolverFrontend::Execute("cusolverDnXpotrs");
    if (CusolverFrontend::Success()) {
        info = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnXgetrf_bufferSize(cusolverDnHandle_t handle, cusolverDnParams_t params, int64_t m, int64_t n, cudaDataType dataTypeA, const void *A, int64_t lda, cudaDataType computeType, size_t *workspaceInBytesOnDevice, size_t *workspaceInBytesOnHost) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) params);
    CusolverFrontend::AddVariableForArguments<int64_t>(m);
    CusolverFrontend::AddVariableForArguments<int64_t>(n);
    CusolverFrontend::AddVariableForArguments<cudaDataType_t>(dataTypeA);
    CusolverFrontend::AddVariableForArguments<int64_t>(lda);
    CusolverFrontend::AddVariableForArguments<cudaDataType_t>(computeType);
    switch(dataTypeA){
        case CUDA_R_32F:
            //float
            CusolverFrontend::AddDevicePointerForArguments((float *)A);
            break;
        case CUDA_R_64F:
            //double
            CusolverFrontend::AddDevicePointerForArguments((double *)A);
            break;
        case CUDA_C_32F:
            //cuComplex
            CusolverFrontend::AddDevicePointerForArguments((cuComplex *)A);
            break;
        case CUDA_C_64F:
            //cuDoubleComplex
            CusolverFrontend::AddDevicePointerForArguments((cuDoubleComplex *)A);
            break;
        default:
            throw "Type not supported by GVirtus!";
    }
    CusolverFrontend::Execute("cusolverDnXgetrf_bufferSize");
    if (CusolverFrontend::Success()) {
        *workspaceInBytesOnDevice = CusolverFrontend::GetOutputVariable<int>();
        *workspaceInBytesOnHost = CusolverFrontend::GetOutputVariable<int>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnXgetrf(cusolverDnHandle_t handle, cusolverDnParams_t params, int64_t m, int64_t n, cudaDataType dataTypeA, void *A, int64_t lda, int64_t *ipiv, cudaDataType computeType, void *bufferOnDevice, size_t workspaceInBytesOnDevice, void *bufferOnHost, size_t workspaceInBytesOnHost, int *info) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) params);
    CusolverFrontend::AddVariableForArguments<int64_t>(m);
    CusolverFrontend::AddVariableForArguments<int64_t>(n);
    CusolverFrontend::AddVariableForArguments<cudaDataType_t>(dataTypeA);
    CusolverFrontend::AddVariableForArguments<int64_t>(lda);
    CusolverFrontend::AddDevicePointerForArguments(ipiv);
    CusolverFrontend::AddVariableForArguments<cudaDataType_t>(computeType);
    CusolverFrontend::AddDevicePointerForArguments(bufferOnDevice);
    CusolverFrontend::AddVariableForArguments<size_t>(workspaceInBytesOnDevice);
    CusolverFrontend::AddHostPointerForArguments((size_t*)bufferOnHost);
    CusolverFrontend::AddVariableForArguments<size_t>(workspaceInBytesOnHost);
    CusolverFrontend::AddDevicePointerForArguments(info);
    switch(dataTypeA){
        case CUDA_R_32F:
            //float
            CusolverFrontend::AddDevicePointerForArguments((float *)A);
            break;
        case CUDA_R_64F:
            //double
            CusolverFrontend::AddDevicePointerForArguments((double *)A);
            break;
        case CUDA_C_32F:
            //cuComplex
            CusolverFrontend::AddDevicePointerForArguments((cuComplex *)A);
            break;
        case CUDA_C_64F:
            //cuDoubleComplex
            CusolverFrontend::AddDevicePointerForArguments((cuDoubleComplex *)A);
            break;
        default:
            throw "Type not supported by GVirtus!";
    }
    CusolverFrontend::Execute("cusolverDnXgetrf");
    if (CusolverFrontend::Success()) {
        ipiv = (int64_t*) CusolverFrontend::GetOutputDevicePointer();
        info = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}