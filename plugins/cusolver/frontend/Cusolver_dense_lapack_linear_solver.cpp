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

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnSpotrf_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, float *A, int lda, int *Lwork) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::Execute("cusolverDnSpotrf_bufferSize");
    if (CusolverFrontend::Success()) {
        *Lwork = CusolverFrontend::GetOutputVariable<int>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnDpotrf_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, double *A, int lda, int *Lwork) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::Execute("cusolverDnDpotrf_bufferSize");
    if (CusolverFrontend::Success()) {
        *Lwork = CusolverFrontend::GetOutputVariable<int>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnCpotrf_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuComplex *A, int lda, int *Lwork) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::Execute("cusolverDnCpotrf_bufferSize");
    if (CusolverFrontend::Success()) {
        *Lwork = CusolverFrontend::GetOutputVariable<int>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnZpotrf_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuDoubleComplex *A, int lda, int *Lwork) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::Execute("cusolverDnZpotrf_bufferSize");
    if (CusolverFrontend::Success()) {
        *Lwork = CusolverFrontend::GetOutputVariable<int>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnSpotrf(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, float *A, int lda, float *Workspace, int Lwork, int *devInfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(Workspace);
    CusolverFrontend::AddVariableForArguments<int>(Lwork);
    CusolverFrontend::AddDevicePointerForArguments(devInfo);
    CusolverFrontend::Execute("cusolverDnSpotrf");
    if (CusolverFrontend::Success()) {
        devInfo = (int*) CusolverFrontend::GetOutputDevicePointer();
        A = (float*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnDpotrf(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, double *A, int lda, double *Workspace, int Lwork, int *devInfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(Workspace);
    CusolverFrontend::AddVariableForArguments<int>(Lwork);
    CusolverFrontend::AddDevicePointerForArguments(devInfo);
    CusolverFrontend::Execute("cusolverDnDpotrf");
    if (CusolverFrontend::Success()) {
        devInfo = (int*) CusolverFrontend::GetOutputDevicePointer();
        A = (double*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnCpotrf(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuComplex *A, int lda, cuComplex *Workspace, int Lwork, int *devInfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(Workspace);
    CusolverFrontend::AddVariableForArguments<int>(Lwork);
    CusolverFrontend::AddDevicePointerForArguments(devInfo);
    CusolverFrontend::Execute("cusolverDnCpotrf");
    if (CusolverFrontend::Success()) {
        devInfo = (int*) CusolverFrontend::GetOutputDevicePointer();
        A = (cuComplex*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnZpotrf(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuDoubleComplex *A, int lda, cuDoubleComplex *Workspace, int Lwork, int *devInfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(Workspace);
    CusolverFrontend::AddVariableForArguments<int>(Lwork);
    CusolverFrontend::AddDevicePointerForArguments(devInfo);
    CusolverFrontend::Execute("cusolverDnZpotrf");
    if (CusolverFrontend::Success()) {
        devInfo = (int*) CusolverFrontend::GetOutputDevicePointer();
        A = (cuDoubleComplex *) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnPotrf_bufferSize(cusolverDnHandle_t handle, cusolverDnParams_t params, cublasFillMode_t uplo, int64_t n, cudaDataType dataTypeA, const void *A, int64_t lda, cudaDataType computeType, size_t *workspaceInBytes) {
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
    CusolverFrontend::Execute("cusolverDnPotrf_bufferSize");
    if (CusolverFrontend::Success()) {
        *workspaceInBytes = CusolverFrontend::GetOutputVariable<int>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnPotrf(cusolverDnHandle_t handle, cusolverDnParams_t params, cublasFillMode_t uplo, int64_t n, cudaDataType dataTypeA, void *A, int64_t lda, cudaDataType computeType, void *pBuffer, size_t workspaceInBytes, int *info) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) params);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<int64_t>(n);
    CusolverFrontend::AddVariableForArguments<cudaDataType_t>(dataTypeA);
    CusolverFrontend::AddVariableForArguments<int64_t>(lda);
    CusolverFrontend::AddVariableForArguments<cudaDataType_t>(computeType);
    CusolverFrontend::AddDevicePointerForArguments(pBuffer);
    CusolverFrontend::AddVariableForArguments<size_t>(workspaceInBytes);
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
    CusolverFrontend::Execute("cusolverDnPotrf");
    if (CusolverFrontend::Success()) {
        info = (int*) CusolverFrontend::GetOutputDevicePointer();
        A = CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnSpotrs(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, int nrhs, const float *A, int lda, float *B, int ldb, int *devInfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nrhs);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(B);
    CusolverFrontend::AddVariableForArguments<int>(ldb);
    CusolverFrontend::AddDevicePointerForArguments(devInfo);
    CusolverFrontend::Execute("cusolverDnSpotrs");
    if (CusolverFrontend::Success()) {
        devInfo = (int*) CusolverFrontend::GetOutputDevicePointer();
        B = (float*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnDpotrs(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, int nrhs, const double *A, int lda, double *B, int ldb, int *devInfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nrhs);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(B);
    CusolverFrontend::AddVariableForArguments<int>(ldb);
    CusolverFrontend::AddDevicePointerForArguments(devInfo);
    CusolverFrontend::Execute("cusolverDnDpotrs");
    if (CusolverFrontend::Success()) {
        devInfo = (int*) CusolverFrontend::GetOutputDevicePointer();
        B = (double*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnCpotrs(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, int nrhs, const cuComplex *A, int lda, cuComplex *B, int ldb, int *devInfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nrhs);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(B);
    CusolverFrontend::AddVariableForArguments<int>(ldb);
    CusolverFrontend::AddDevicePointerForArguments(devInfo);
    CusolverFrontend::Execute("cusolverDnCpotrs");
    if (CusolverFrontend::Success()) {
        devInfo = (int*) CusolverFrontend::GetOutputDevicePointer();
        B = (cuComplex*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnZpotrs(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, int nrhs, const cuDoubleComplex *A, int lda, cuDoubleComplex *B, int ldb, int *devInfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nrhs);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(B);
    CusolverFrontend::AddVariableForArguments<int>(ldb);
    CusolverFrontend::AddDevicePointerForArguments(devInfo);
    CusolverFrontend::Execute("cusolverDnZpotrs");
    if (CusolverFrontend::Success()) {
        devInfo = (int*) CusolverFrontend::GetOutputDevicePointer();
        B = (cuDoubleComplex*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnPotrs(cusolverDnHandle_t handle, cusolverDnParams_t params, cublasFillMode_t uplo, int64_t n, int64_t nrhs, cudaDataType dataTypeA, const void *A, int64_t lda, cudaDataType dataTypeB, void *B, int64_t ldb, int *info) {
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
            CusolverFrontend::AddDevicePointerForArguments((float *)B);
            break;
        case CUDA_R_64F:
            //double
            CusolverFrontend::AddDevicePointerForArguments((double *)A);
            CusolverFrontend::AddDevicePointerForArguments((double *)B);
            break;
        case CUDA_C_32F:
            //cuComplex
            CusolverFrontend::AddDevicePointerForArguments((cuComplex *)A);
            CusolverFrontend::AddDevicePointerForArguments((cuComplex *)B);
            break;
        case CUDA_C_64F:
            //cuDoubleComplex
            CusolverFrontend::AddDevicePointerForArguments((cuDoubleComplex *)A);
            CusolverFrontend::AddDevicePointerForArguments((cuDoubleComplex *)B);
            break;
        default:
            throw "Type not supported by GVirtus!";
    }
    CusolverFrontend::Execute("cusolverDnPotrs");
    if (CusolverFrontend::Success()) {
        info = (int*) CusolverFrontend::GetOutputDevicePointer();
        B = CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnSpotri_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, float *A, int lda, int *Lwork) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::Execute("cusolverDnSpotri_bufferSize");
    if (CusolverFrontend::Success()) {
        *Lwork = CusolverFrontend::GetOutputVariable<int>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnDpotri_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, double *A, int lda, int *Lwork) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::Execute("cusolverDnDpotri_bufferSize");
    if (CusolverFrontend::Success()) {
        *Lwork = CusolverFrontend::GetOutputVariable<int>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnCpotri_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuComplex *A, int lda, int *Lwork) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::Execute("cusolverDnCpotri_bufferSize");
    if (CusolverFrontend::Success()) {
        *Lwork = CusolverFrontend::GetOutputVariable<int>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnZpotri_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuDoubleComplex *A, int lda, int *Lwork) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::Execute("cusolverDnZpotri_bufferSize");
    if (CusolverFrontend::Success()) {
        *Lwork = CusolverFrontend::GetOutputVariable<int>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnSpotri(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, float *A, int lda, float *Workspace, int Lwork, int *devInfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(Workspace);
    CusolverFrontend::AddVariableForArguments<int>(Lwork);
    CusolverFrontend::AddDevicePointerForArguments(devInfo);
    CusolverFrontend::Execute("cusolverDnSpotri");
    if (CusolverFrontend::Success()) {
        devInfo = (int*) CusolverFrontend::GetOutputDevicePointer();
        A = (float*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnDpotri(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, double *A, int lda, double *Workspace, int Lwork, int *devInfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(Workspace);
    CusolverFrontend::AddVariableForArguments<int>(Lwork);
    CusolverFrontend::AddDevicePointerForArguments(devInfo);
    CusolverFrontend::Execute("cusolverDnDpotri");
    if (CusolverFrontend::Success()) {
        devInfo = (int*) CusolverFrontend::GetOutputDevicePointer();
        A = (double*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnCpotri(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuComplex *A, int lda, cuComplex *Workspace, int Lwork, int *devInfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(Workspace);
    CusolverFrontend::AddVariableForArguments<int>(Lwork);
    CusolverFrontend::AddDevicePointerForArguments(devInfo);
    CusolverFrontend::Execute("cusolverDnCpotri");
    if (CusolverFrontend::Success()) {
        devInfo = (int*) CusolverFrontend::GetOutputDevicePointer();
        A = (cuComplex*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnZpotri(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuDoubleComplex *A, int lda, cuDoubleComplex *Workspace, int Lwork, int *devInfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(Workspace);
    CusolverFrontend::AddVariableForArguments<int>(Lwork);
    CusolverFrontend::AddDevicePointerForArguments(devInfo);
    CusolverFrontend::Execute("cusolverDnZpotri");
    if (CusolverFrontend::Success()) {
        devInfo = (int*) CusolverFrontend::GetOutputDevicePointer();
        A = (cuDoubleComplex *) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnSgetrf_bufferSize(cusolverDnHandle_t handle, int m, int n, float *A, int lda, int *Lwork) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::Execute("cusolverDnSgetrf_bufferSize");
    if (CusolverFrontend::Success()) {
        *Lwork = CusolverFrontend::GetOutputVariable<int>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnDgetrf_bufferSize(cusolverDnHandle_t handle, int m, int n, double *A, int lda, int *Lwork) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::Execute("cusolverDnDgetrf_bufferSize");
    if (CusolverFrontend::Success()) {
        *Lwork = CusolverFrontend::GetOutputVariable<int>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnCgetrf_bufferSize(cusolverDnHandle_t handle, int m, int n, cuComplex *A, int lda, int *Lwork) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::Execute("cusolverDnCgetrf_bufferSize");
    if (CusolverFrontend::Success()) {
        *Lwork = CusolverFrontend::GetOutputVariable<int>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnZgetrf_bufferSize(cusolverDnHandle_t handle, int m, int n, cuDoubleComplex *A, int lda, int *Lwork) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::Execute("cusolverDnZgetrf_bufferSize");
    if (CusolverFrontend::Success()) {
        *Lwork = CusolverFrontend::GetOutputVariable<int>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnSgetrf(cusolverDnHandle_t handle, int m, int n, float *A, int lda, float *Workspace, int *devIpiv, int *devInfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(Workspace);
    CusolverFrontend::AddDevicePointerForArguments(devIpiv);
    CusolverFrontend::AddDevicePointerForArguments(devInfo);
    CusolverFrontend::Execute("cusolverDnSgetrf");
    if (CusolverFrontend::Success()) {
        devIpiv = (int*) CusolverFrontend::GetOutputDevicePointer();
        devInfo = (int*) CusolverFrontend::GetOutputDevicePointer();
        A = (float*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnDgetrf(cusolverDnHandle_t handle, int m, int n, double *A, int lda, double *Workspace, int *devIpiv, int *devInfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(Workspace);
    CusolverFrontend::AddDevicePointerForArguments(devIpiv);
    CusolverFrontend::AddDevicePointerForArguments(devInfo);
    CusolverFrontend::Execute("cusolverDnDgetrf");
    if (CusolverFrontend::Success()) {
        devIpiv = (int*) CusolverFrontend::GetOutputDevicePointer();
        devInfo = (int*) CusolverFrontend::GetOutputDevicePointer();
        A = (double*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnCgetrf(cusolverDnHandle_t handle, int m, int n, cuComplex *A, int lda, cuComplex *Workspace, int *devIpiv, int *devInfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(Workspace);
    CusolverFrontend::AddDevicePointerForArguments(devIpiv);
    CusolverFrontend::AddDevicePointerForArguments(devInfo);
    CusolverFrontend::Execute("cusolverDnCgetrf");
    if (CusolverFrontend::Success()) {
        devIpiv = (int*) CusolverFrontend::GetOutputDevicePointer();
        devInfo = (int*) CusolverFrontend::GetOutputDevicePointer();
        A = (cuComplex*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnZgetrf(cusolverDnHandle_t handle, int m, int n, cuDoubleComplex *A, int lda, cuDoubleComplex *Workspace, int *devIpiv, int *devInfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(Workspace);
    CusolverFrontend::AddDevicePointerForArguments(devIpiv);
    CusolverFrontend::AddDevicePointerForArguments(devInfo);
    CusolverFrontend::Execute("cusolverDnZgetrf");
    if (CusolverFrontend::Success()) {
        devIpiv = (int*) CusolverFrontend::GetOutputDevicePointer();
        devInfo = (int*) CusolverFrontend::GetOutputDevicePointer();
        A = (cuDoubleComplex*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnGetrf_bufferSize(cusolverDnHandle_t handle, cusolverDnParams_t params, int64_t m, int64_t n, cudaDataType dataTypeA, const void *A, int64_t lda, cudaDataType computeType, size_t *workspaceInBytes) {
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
    CusolverFrontend::Execute("cusolverDnGetrf_bufferSize");
    if (CusolverFrontend::Success()) {
        *workspaceInBytes = CusolverFrontend::GetOutputVariable<int>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnGetrf(cusolverDnHandle_t handle, cusolverDnParams_t params, int64_t m, int64_t n, cudaDataType dataTypeA, void *A, int64_t lda, int64_t *ipiv, cudaDataType computeType, void *pBuffer, size_t workspaceInBytes, int *info) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) params);
    CusolverFrontend::AddVariableForArguments<int64_t>(m);
    CusolverFrontend::AddVariableForArguments<int64_t>(n);
    CusolverFrontend::AddVariableForArguments<cudaDataType_t>(dataTypeA);
    CusolverFrontend::AddVariableForArguments<int64_t>(lda);
    CusolverFrontend::AddDevicePointerForArguments(ipiv);
    CusolverFrontend::AddVariableForArguments<cudaDataType_t>(computeType);
    CusolverFrontend::AddDevicePointerForArguments(pBuffer);
    CusolverFrontend::AddVariableForArguments<size_t>(workspaceInBytes);
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
    CusolverFrontend::Execute("cusolverDnGetrf");
    if (CusolverFrontend::Success()) {
        ipiv = (int64_t*) CusolverFrontend::GetOutputDevicePointer();
        info = (int*) CusolverFrontend::GetOutputDevicePointer();
        A = CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnSgetrs(cusolverDnHandle_t handle, cublasOperation_t trans, int n, int nrhs, const float *A, int lda, const int *devIpiv, float *B, int ldb, int *devInfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cublasOperation_t>(trans);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nrhs);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(devIpiv);
    CusolverFrontend::AddDevicePointerForArguments(B);
    CusolverFrontend::AddVariableForArguments<int>(ldb);
    CusolverFrontend::AddDevicePointerForArguments(devInfo);
    CusolverFrontend::Execute("cusolverDnSgetrs");
    if (CusolverFrontend::Success()) {
        devInfo = (int*) CusolverFrontend::GetOutputDevicePointer();
        B = (float*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnDgetrs(cusolverDnHandle_t handle, cublasOperation_t trans, int n, int nrhs, const double *A, int lda, const int *devIpiv, double *B, int ldb, int *devInfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cublasOperation_t>(trans);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nrhs);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(devIpiv);
    CusolverFrontend::AddDevicePointerForArguments(B);
    CusolverFrontend::AddVariableForArguments<int>(ldb);
    CusolverFrontend::AddDevicePointerForArguments(devInfo);
    CusolverFrontend::Execute("cusolverDnDgetrs");
    if (CusolverFrontend::Success()) {
        devInfo = (int*) CusolverFrontend::GetOutputDevicePointer();
        B = (double*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnCgetrs(cusolverDnHandle_t handle, cublasOperation_t trans, int n, int nrhs, const cuComplex *A, int lda, const int *devIpiv, cuComplex *B, int ldb, int *devInfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cublasOperation_t>(trans);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nrhs);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(devIpiv);
    CusolverFrontend::AddDevicePointerForArguments(B);
    CusolverFrontend::AddVariableForArguments<int>(ldb);
    CusolverFrontend::AddDevicePointerForArguments(devInfo);
    CusolverFrontend::Execute("cusolverDnCgetrs");
    if (CusolverFrontend::Success()) {
        devInfo = (int*) CusolverFrontend::GetOutputDevicePointer();
        B = (cuComplex*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnZgetrs(cusolverDnHandle_t handle, cublasOperation_t trans, int n, int nrhs, const cuDoubleComplex *A, int lda, const int *devIpiv, cuDoubleComplex *B, int ldb, int *devInfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cublasOperation_t>(trans);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nrhs);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(devIpiv);
    CusolverFrontend::AddDevicePointerForArguments(B);
    CusolverFrontend::AddVariableForArguments<int>(ldb);
    CusolverFrontend::AddDevicePointerForArguments(devInfo);
    CusolverFrontend::Execute("cusolverDnZgetrs");
    if (CusolverFrontend::Success()) {
        devInfo = (int*) CusolverFrontend::GetOutputDevicePointer();
        B = (cuDoubleComplex*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnGetrs(cusolverDnHandle_t handle, cusolverDnParams_t params, cublasOperation_t trans, int64_t n, int64_t nrhs, cudaDataType dataTypeA, const void *A, int64_t lda, const int64_t *ipiv, cudaDataType dataTypeB, void* B, int64_t ldb, int *info) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) params);
    CusolverFrontend::AddVariableForArguments<cublasOperation_t>(trans);
    CusolverFrontend::AddVariableForArguments<int64_t>(n);
    CusolverFrontend::AddVariableForArguments<int64_t>(nrhs);
    CusolverFrontend::AddVariableForArguments<cudaDataType_t>(dataTypeA);
    CusolverFrontend::AddVariableForArguments<int64_t>(lda);
    CusolverFrontend::AddDevicePointerForArguments(ipiv);
    CusolverFrontend::AddVariableForArguments<cudaDataType_t>(dataTypeB);
    CusolverFrontend::AddVariableForArguments<int64_t>(ldb);
    CusolverFrontend::AddDevicePointerForArguments(info);
    switch(dataTypeA){
        case CUDA_R_32F:
            //float
            CusolverFrontend::AddDevicePointerForArguments((float *)A);
            CusolverFrontend::AddDevicePointerForArguments((float *)B);
            break;
        case CUDA_R_64F:
            //double
            CusolverFrontend::AddDevicePointerForArguments((double *)A);
            CusolverFrontend::AddDevicePointerForArguments((double *)B);
            break;
        case CUDA_C_32F:
            //cuComplex
            CusolverFrontend::AddDevicePointerForArguments((cuComplex *)A);
            CusolverFrontend::AddDevicePointerForArguments((cuComplex *)B);
            break;
        case CUDA_C_64F:
            //cuDoubleComplex
            CusolverFrontend::AddDevicePointerForArguments((cuDoubleComplex *)A);
            CusolverFrontend::AddDevicePointerForArguments((cuDoubleComplex *)B);
            break;
        default:
            throw "Type not supported by GVirtus!";
    }
    CusolverFrontend::Execute("cusolverDnGetrs");
    if (CusolverFrontend::Success()) {
        info = (int*) CusolverFrontend::GetOutputDevicePointer();
        B = CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnZZgesv_bufferSize(cusolverDnHandle_t handle, int n, int nrhs, cuDoubleComplex *dA, int ldda, int *   dipiv, cuDoubleComplex *dB, int lddb, cuDoubleComplex *dX, int lddx, void *dwork, size_t *lwork_bytes) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nrhs);
    CusolverFrontend::AddDevicePointerForArguments(dA);
    CusolverFrontend::AddVariableForArguments<int>(ldda);
    CusolverFrontend::AddDevicePointerForArguments(dipiv);
    CusolverFrontend::AddDevicePointerForArguments(dB);
    CusolverFrontend::AddVariableForArguments<int>(lddb);
    CusolverFrontend::AddDevicePointerForArguments(dX);
    CusolverFrontend::AddVariableForArguments<int>(lddx);
    CusolverFrontend::AddDevicePointerForArguments(dwork);
    CusolverFrontend::Execute("cusolverDnZZgesv_bufferSize");
    if (CusolverFrontend::Success()) {
        *lwork_bytes = CusolverFrontend::GetOutputVariable<size_t>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnZCgesv_bufferSize(cusolverDnHandle_t handle, int n, int nrhs, cuDoubleComplex *dA, int ldda, int *   dipiv, cuDoubleComplex *dB, int lddb, cuDoubleComplex *dX, int lddx, void *dwork, size_t *lwork_bytes) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nrhs);
    CusolverFrontend::AddDevicePointerForArguments(dA);
    CusolverFrontend::AddVariableForArguments<int>(ldda);
    CusolverFrontend::AddDevicePointerForArguments(dipiv);
    CusolverFrontend::AddDevicePointerForArguments(dB);
    CusolverFrontend::AddVariableForArguments<int>(lddb);
    CusolverFrontend::AddDevicePointerForArguments(dX);
    CusolverFrontend::AddVariableForArguments<int>(lddx);
    CusolverFrontend::AddDevicePointerForArguments(dwork);
    CusolverFrontend::Execute("cusolverDnZCgesv_bufferSize");
    if (CusolverFrontend::Success()) {
        *lwork_bytes = CusolverFrontend::GetOutputVariable<size_t>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnZKgesv_bufferSize(cusolverDnHandle_t handle, int n, int nrhs, cuDoubleComplex *dA, int ldda, int *   dipiv, cuDoubleComplex *dB, int lddb, cuDoubleComplex *dX, int lddx, void *dwork, size_t *lwork_bytes) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nrhs);
    CusolverFrontend::AddDevicePointerForArguments(dA);
    CusolverFrontend::AddVariableForArguments<int>(ldda);
    CusolverFrontend::AddDevicePointerForArguments(dipiv);
    CusolverFrontend::AddDevicePointerForArguments(dB);
    CusolverFrontend::AddVariableForArguments<int>(lddb);
    CusolverFrontend::AddDevicePointerForArguments(dX);
    CusolverFrontend::AddVariableForArguments<int>(lddx);
    CusolverFrontend::AddDevicePointerForArguments(dwork);
    CusolverFrontend::Execute("cusolverDnZKgesv_bufferSize");
    if (CusolverFrontend::Success()) {
        *lwork_bytes = CusolverFrontend::GetOutputVariable<size_t>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnZEgesv_bufferSize(cusolverDnHandle_t handle, int n, int nrhs, cuDoubleComplex *dA, int ldda, int *   dipiv, cuDoubleComplex *dB, int lddb, cuDoubleComplex *dX, int lddx, void *dwork, size_t *lwork_bytes) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nrhs);
    CusolverFrontend::AddDevicePointerForArguments(dA);
    CusolverFrontend::AddVariableForArguments<int>(ldda);
    CusolverFrontend::AddDevicePointerForArguments(dipiv);
    CusolverFrontend::AddDevicePointerForArguments(dB);
    CusolverFrontend::AddVariableForArguments<int>(lddb);
    CusolverFrontend::AddDevicePointerForArguments(dX);
    CusolverFrontend::AddVariableForArguments<int>(lddx);
    CusolverFrontend::AddDevicePointerForArguments(dwork);
    CusolverFrontend::Execute("cusolverDnZEgesv_bufferSize");
    if (CusolverFrontend::Success()) {
        *lwork_bytes = CusolverFrontend::GetOutputVariable<size_t>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnZYgesv_bufferSize(cusolverDnHandle_t handle, int n, int nrhs, cuDoubleComplex *dA, int ldda, int *   dipiv, cuDoubleComplex *dB, int lddb, cuDoubleComplex *dX, int lddx, void *dwork, size_t *lwork_bytes) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nrhs);
    CusolverFrontend::AddDevicePointerForArguments(dA);
    CusolverFrontend::AddVariableForArguments<int>(ldda);
    CusolverFrontend::AddDevicePointerForArguments(dipiv);
    CusolverFrontend::AddDevicePointerForArguments(dB);
    CusolverFrontend::AddVariableForArguments<int>(lddb);
    CusolverFrontend::AddDevicePointerForArguments(dX);
    CusolverFrontend::AddVariableForArguments<int>(lddx);
    CusolverFrontend::AddDevicePointerForArguments(dwork);
    CusolverFrontend::Execute("cusolverDnZYgesv_bufferSize");
    if (CusolverFrontend::Success()) {
        *lwork_bytes = CusolverFrontend::GetOutputVariable<size_t>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnCCgesv_bufferSize(cusolverDnHandle_t handle, int n, int nrhs, cuComplex *dA, int ldda, int *   dipiv, cuComplex *dB, int lddb, cuComplex *dX, int lddx, void *dwork, size_t *lwork_bytes) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nrhs);
    CusolverFrontend::AddDevicePointerForArguments(dA);
    CusolverFrontend::AddVariableForArguments<int>(ldda);
    CusolverFrontend::AddDevicePointerForArguments(dipiv);
    CusolverFrontend::AddDevicePointerForArguments(dB);
    CusolverFrontend::AddVariableForArguments<int>(lddb);
    CusolverFrontend::AddDevicePointerForArguments(dX);
    CusolverFrontend::AddVariableForArguments<int>(lddx);
    CusolverFrontend::AddDevicePointerForArguments(dwork);
    CusolverFrontend::Execute("cusolverDnCCgesv_bufferSize");
    if (CusolverFrontend::Success()) {
        *lwork_bytes = CusolverFrontend::GetOutputVariable<size_t>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnCKgesv_bufferSize(cusolverDnHandle_t handle, int n, int nrhs, cuComplex *dA, int ldda, int *   dipiv, cuComplex *dB, int lddb, cuComplex *dX, int lddx, void *dwork, size_t *lwork_bytes) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nrhs);
    CusolverFrontend::AddDevicePointerForArguments(dA);
    CusolverFrontend::AddVariableForArguments<int>(ldda);
    CusolverFrontend::AddDevicePointerForArguments(dipiv);
    CusolverFrontend::AddDevicePointerForArguments(dB);
    CusolverFrontend::AddVariableForArguments<int>(lddb);
    CusolverFrontend::AddDevicePointerForArguments(dX);
    CusolverFrontend::AddVariableForArguments<int>(lddx);
    CusolverFrontend::AddDevicePointerForArguments(dwork);
    CusolverFrontend::Execute("cusolverDnCKgesv_bufferSize");
    if (CusolverFrontend::Success()) {
        *lwork_bytes = CusolverFrontend::GetOutputVariable<size_t>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnCEgesv_bufferSize(cusolverDnHandle_t handle, int n, int nrhs, cuComplex *dA, int ldda, int *   dipiv, cuComplex *dB, int lddb, cuComplex *dX, int lddx, void *dwork, size_t *lwork_bytes) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nrhs);
    CusolverFrontend::AddDevicePointerForArguments(dA);
    CusolverFrontend::AddVariableForArguments<int>(ldda);
    CusolverFrontend::AddDevicePointerForArguments(dipiv);
    CusolverFrontend::AddDevicePointerForArguments(dB);
    CusolverFrontend::AddVariableForArguments<int>(lddb);
    CusolverFrontend::AddDevicePointerForArguments(dX);
    CusolverFrontend::AddVariableForArguments<int>(lddx);
    CusolverFrontend::AddDevicePointerForArguments(dwork);
    CusolverFrontend::Execute("cusolverDnCEgesv_bufferSize");
    if (CusolverFrontend::Success()) {
        *lwork_bytes = CusolverFrontend::GetOutputVariable<size_t>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnCYgesv_bufferSize(cusolverDnHandle_t handle, int n, int nrhs, cuComplex *dA, int ldda, int *   dipiv, cuComplex *dB, int lddb, cuComplex *dX, int lddx, void *dwork, size_t *lwork_bytes) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nrhs);
    CusolverFrontend::AddDevicePointerForArguments(dA);
    CusolverFrontend::AddVariableForArguments<int>(ldda);
    CusolverFrontend::AddDevicePointerForArguments(dipiv);
    CusolverFrontend::AddDevicePointerForArguments(dB);
    CusolverFrontend::AddVariableForArguments<int>(lddb);
    CusolverFrontend::AddDevicePointerForArguments(dX);
    CusolverFrontend::AddVariableForArguments<int>(lddx);
    CusolverFrontend::AddDevicePointerForArguments(dwork);
    CusolverFrontend::Execute("cusolverDnCYgesv_bufferSize");
    if (CusolverFrontend::Success()) {
        *lwork_bytes = CusolverFrontend::GetOutputVariable<size_t>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnDDgesv_bufferSize(cusolverDnHandle_t handle, int n, int nrhs, double *dA, int ldda, int *   dipiv, double *dB, int lddb, double *dX, int lddx, void *dwork, size_t *lwork_bytes) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nrhs);
    CusolverFrontend::AddDevicePointerForArguments(dA);
    CusolverFrontend::AddVariableForArguments<int>(ldda);
    CusolverFrontend::AddDevicePointerForArguments(dipiv);
    CusolverFrontend::AddDevicePointerForArguments(dB);
    CusolverFrontend::AddVariableForArguments<int>(lddb);
    CusolverFrontend::AddDevicePointerForArguments(dX);
    CusolverFrontend::AddVariableForArguments<int>(lddx);
    CusolverFrontend::AddDevicePointerForArguments(dwork);
    CusolverFrontend::Execute("cusolverDnDDgesv_bufferSize");
    if (CusolverFrontend::Success()) {
        *lwork_bytes = CusolverFrontend::GetOutputVariable<size_t>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnDSgesv_bufferSize(cusolverDnHandle_t handle, int n, int nrhs, double *dA, int ldda, int *   dipiv, double *dB, int lddb, double *dX, int lddx, void *dwork, size_t *lwork_bytes) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nrhs);
    CusolverFrontend::AddDevicePointerForArguments(dA);
    CusolverFrontend::AddVariableForArguments<int>(ldda);
    CusolverFrontend::AddDevicePointerForArguments(dipiv);
    CusolverFrontend::AddDevicePointerForArguments(dB);
    CusolverFrontend::AddVariableForArguments<int>(lddb);
    CusolverFrontend::AddDevicePointerForArguments(dX);
    CusolverFrontend::AddVariableForArguments<int>(lddx);
    CusolverFrontend::AddDevicePointerForArguments(dwork);
    CusolverFrontend::Execute("cusolverDnDSgesv_bufferSize");
    if (CusolverFrontend::Success()) {
        *lwork_bytes = CusolverFrontend::GetOutputVariable<size_t>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnDHgesv_bufferSize(cusolverDnHandle_t handle, int n, int nrhs, double *dA, int ldda, int *   dipiv, double *dB, int lddb, double *dX, int lddx, void *dwork, size_t *lwork_bytes) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nrhs);
    CusolverFrontend::AddDevicePointerForArguments(dA);
    CusolverFrontend::AddVariableForArguments<int>(ldda);
    CusolverFrontend::AddDevicePointerForArguments(dipiv);
    CusolverFrontend::AddDevicePointerForArguments(dB);
    CusolverFrontend::AddVariableForArguments<int>(lddb);
    CusolverFrontend::AddDevicePointerForArguments(dX);
    CusolverFrontend::AddVariableForArguments<int>(lddx);
    CusolverFrontend::AddDevicePointerForArguments(dwork);
    CusolverFrontend::Execute("cusolverDnDHgesv_bufferSize");
    if (CusolverFrontend::Success()) {
        *lwork_bytes = CusolverFrontend::GetOutputVariable<size_t>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnDBgesv_bufferSize(cusolverDnHandle_t handle, int n, int nrhs, double *dA, int ldda, int *   dipiv, double *dB, int lddb, double *dX, int lddx, void *dwork, size_t *lwork_bytes) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nrhs);
    CusolverFrontend::AddDevicePointerForArguments(dA);
    CusolverFrontend::AddVariableForArguments<int>(ldda);
    CusolverFrontend::AddDevicePointerForArguments(dipiv);
    CusolverFrontend::AddDevicePointerForArguments(dB);
    CusolverFrontend::AddVariableForArguments<int>(lddb);
    CusolverFrontend::AddDevicePointerForArguments(dX);
    CusolverFrontend::AddVariableForArguments<int>(lddx);
    CusolverFrontend::AddDevicePointerForArguments(dwork);
    CusolverFrontend::Execute("cusolverDnDBgesv_bufferSize");
    if (CusolverFrontend::Success()) {
        *lwork_bytes = CusolverFrontend::GetOutputVariable<size_t>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnDXgesv_bufferSize(cusolverDnHandle_t handle, int n, int nrhs, double *dA, int ldda, int *   dipiv, double *dB, int lddb, double *dX, int lddx, void *dwork, size_t *lwork_bytes) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nrhs);
    CusolverFrontend::AddDevicePointerForArguments(dA);
    CusolverFrontend::AddVariableForArguments<int>(ldda);
    CusolverFrontend::AddDevicePointerForArguments(dipiv);
    CusolverFrontend::AddDevicePointerForArguments(dB);
    CusolverFrontend::AddVariableForArguments<int>(lddb);
    CusolverFrontend::AddDevicePointerForArguments(dX);
    CusolverFrontend::AddVariableForArguments<int>(lddx);
    CusolverFrontend::AddDevicePointerForArguments(dwork);
    CusolverFrontend::Execute("cusolverDnDXgesv_bufferSize");
    if (CusolverFrontend::Success()) {
        *lwork_bytes = CusolverFrontend::GetOutputVariable<size_t>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnSSgesv_bufferSize(cusolverDnHandle_t handle, int n, int nrhs, float *dA, int ldda, int *   dipiv, float *dB, int lddb, float *dX, int lddx, void *dwork, size_t *lwork_bytes) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nrhs);
    CusolverFrontend::AddDevicePointerForArguments(dA);
    CusolverFrontend::AddVariableForArguments<int>(ldda);
    CusolverFrontend::AddDevicePointerForArguments(dipiv);
    CusolverFrontend::AddDevicePointerForArguments(dB);
    CusolverFrontend::AddVariableForArguments<int>(lddb);
    CusolverFrontend::AddDevicePointerForArguments(dX);
    CusolverFrontend::AddVariableForArguments<int>(lddx);
    CusolverFrontend::AddDevicePointerForArguments(dwork);
    CusolverFrontend::Execute("cusolverDnSSgesv_bufferSize");
    if (CusolverFrontend::Success()) {
        *lwork_bytes = CusolverFrontend::GetOutputVariable<size_t>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnSHgesv_bufferSize(cusolverDnHandle_t handle, int n, int nrhs, float *dA, int ldda, int *   dipiv, float *dB, int lddb, float *dX, int lddx, void *dwork, size_t *lwork_bytes) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nrhs);
    CusolverFrontend::AddDevicePointerForArguments(dA);
    CusolverFrontend::AddVariableForArguments<int>(ldda);
    CusolverFrontend::AddDevicePointerForArguments(dipiv);
    CusolverFrontend::AddDevicePointerForArguments(dB);
    CusolverFrontend::AddVariableForArguments<int>(lddb);
    CusolverFrontend::AddDevicePointerForArguments(dX);
    CusolverFrontend::AddVariableForArguments<int>(lddx);
    CusolverFrontend::AddDevicePointerForArguments(dwork);
    CusolverFrontend::Execute("cusolverDnSHgesv_bufferSize");
    if (CusolverFrontend::Success()) {
        *lwork_bytes = CusolverFrontend::GetOutputVariable<size_t>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnSBgesv_bufferSize(cusolverDnHandle_t handle, int n, int nrhs, float *dA, int ldda, int *   dipiv, float *dB, int lddb, float *dX, int lddx, void *dwork, size_t *lwork_bytes) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nrhs);
    CusolverFrontend::AddDevicePointerForArguments(dA);
    CusolverFrontend::AddVariableForArguments<int>(ldda);
    CusolverFrontend::AddDevicePointerForArguments(dipiv);
    CusolverFrontend::AddDevicePointerForArguments(dB);
    CusolverFrontend::AddVariableForArguments<int>(lddb);
    CusolverFrontend::AddDevicePointerForArguments(dX);
    CusolverFrontend::AddVariableForArguments<int>(lddx);
    CusolverFrontend::AddDevicePointerForArguments(dwork);
    CusolverFrontend::Execute("cusolverDnSBgesv_bufferSize");
    if (CusolverFrontend::Success()) {
        *lwork_bytes = CusolverFrontend::GetOutputVariable<size_t>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnSXgesv_bufferSize(cusolverDnHandle_t handle, int n, int nrhs, float *dA, int ldda, int *   dipiv, float *dB, int lddb, float *dX, int lddx, void *dwork, size_t *lwork_bytes) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nrhs);
    CusolverFrontend::AddDevicePointerForArguments(dA);
    CusolverFrontend::AddVariableForArguments<int>(ldda);
    CusolverFrontend::AddDevicePointerForArguments(dipiv);
    CusolverFrontend::AddDevicePointerForArguments(dB);
    CusolverFrontend::AddVariableForArguments<int>(lddb);
    CusolverFrontend::AddDevicePointerForArguments(dX);
    CusolverFrontend::AddVariableForArguments<int>(lddx);
    CusolverFrontend::AddDevicePointerForArguments(dwork);
    CusolverFrontend::Execute("cusolverDnSXgesv_bufferSize");
    if (CusolverFrontend::Success()) {
        *lwork_bytes = CusolverFrontend::GetOutputVariable<size_t>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnZZgesv(cusolverDnHandle_t handle, int n, int nrhs, cuDoubleComplex *dA, int ldda, int *dipiv, cuDoubleComplex *dB, int lddb, cuDoubleComplex *dX, int lddx, void *dWorkspace, size_t lwork_bytes, int *niter, int *dinfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nrhs);
    CusolverFrontend::AddDevicePointerForArguments(dA);
    CusolverFrontend::AddVariableForArguments<int>(ldda);
    CusolverFrontend::AddDevicePointerForArguments(dipiv);
    CusolverFrontend::AddDevicePointerForArguments(dB);
    CusolverFrontend::AddVariableForArguments<int>(lddb);
    CusolverFrontend::AddDevicePointerForArguments(dX);
    CusolverFrontend::AddVariableForArguments<int>(lddx);
    CusolverFrontend::AddDevicePointerForArguments(dWorkspace);
    CusolverFrontend::AddVariableForArguments<size_t>(lwork_bytes);
    CusolverFrontend::AddDevicePointerForArguments(dinfo);
    CusolverFrontend::Execute("cusolverDnZZgesv");
    if (CusolverFrontend::Success()) {
        dA = (cuDoubleComplex *) CusolverFrontend::GetOutputDevicePointer();
        dipiv = (int*) CusolverFrontend::GetOutputDevicePointer();
        dX = (cuDoubleComplex *) CusolverFrontend::GetOutputDevicePointer();
        *niter = CusolverFrontend::GetOutputVariable<int>();
        dinfo = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnZCgesv(cusolverDnHandle_t handle, int n, int nrhs, cuDoubleComplex *dA, int ldda, int *dipiv, cuDoubleComplex *dB, int lddb, cuDoubleComplex *dX, int lddx, void *dWorkspace, size_t lwork_bytes, int *niter, int *dinfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nrhs);
    CusolverFrontend::AddDevicePointerForArguments(dA);
    CusolverFrontend::AddVariableForArguments<int>(ldda);
    CusolverFrontend::AddDevicePointerForArguments(dipiv);
    CusolverFrontend::AddDevicePointerForArguments(dB);
    CusolverFrontend::AddVariableForArguments<int>(lddb);
    CusolverFrontend::AddDevicePointerForArguments(dX);
    CusolverFrontend::AddVariableForArguments<int>(lddx);
    CusolverFrontend::AddDevicePointerForArguments(dWorkspace);
    CusolverFrontend::AddVariableForArguments<size_t>(lwork_bytes);
    CusolverFrontend::AddDevicePointerForArguments(dinfo);
    CusolverFrontend::Execute("cusolverDnZCgesv");
    if (CusolverFrontend::Success()) {
        dA = (cuDoubleComplex *) CusolverFrontend::GetOutputDevicePointer();
        dipiv = (int*) CusolverFrontend::GetOutputDevicePointer();
        dX = (cuDoubleComplex *) CusolverFrontend::GetOutputDevicePointer();
        *niter = CusolverFrontend::GetOutputVariable<int>();
        dinfo = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnZKgesv(cusolverDnHandle_t handle, int n, int nrhs, cuDoubleComplex *dA, int ldda, int *dipiv, cuDoubleComplex *dB, int lddb, cuDoubleComplex *dX, int lddx, void *dWorkspace, size_t lwork_bytes, int *niter, int *dinfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nrhs);
    CusolverFrontend::AddDevicePointerForArguments(dA);
    CusolverFrontend::AddVariableForArguments<int>(ldda);
    CusolverFrontend::AddDevicePointerForArguments(dipiv);
    CusolverFrontend::AddDevicePointerForArguments(dB);
    CusolverFrontend::AddVariableForArguments<int>(lddb);
    CusolverFrontend::AddDevicePointerForArguments(dX);
    CusolverFrontend::AddVariableForArguments<int>(lddx);
    CusolverFrontend::AddDevicePointerForArguments(dWorkspace);
    CusolverFrontend::AddVariableForArguments<size_t>(lwork_bytes);
    CusolverFrontend::AddDevicePointerForArguments(dinfo);
    CusolverFrontend::Execute("cusolverDnZKgesv");
    if (CusolverFrontend::Success()) {
        dA = (cuDoubleComplex *) CusolverFrontend::GetOutputDevicePointer();
        dipiv = (int*) CusolverFrontend::GetOutputDevicePointer();
        dX = (cuDoubleComplex *) CusolverFrontend::GetOutputDevicePointer();
        *niter = CusolverFrontend::GetOutputVariable<int>();
        dinfo = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnZEgesv(cusolverDnHandle_t handle, int n, int nrhs, cuDoubleComplex *dA, int ldda, int *dipiv, cuDoubleComplex *dB, int lddb, cuDoubleComplex *dX, int lddx, void *dWorkspace, size_t lwork_bytes, int *niter, int *dinfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nrhs);
    CusolverFrontend::AddDevicePointerForArguments(dA);
    CusolverFrontend::AddVariableForArguments<int>(ldda);
    CusolverFrontend::AddDevicePointerForArguments(dipiv);
    CusolverFrontend::AddDevicePointerForArguments(dB);
    CusolverFrontend::AddVariableForArguments<int>(lddb);
    CusolverFrontend::AddDevicePointerForArguments(dX);
    CusolverFrontend::AddVariableForArguments<int>(lddx);
    CusolverFrontend::AddDevicePointerForArguments(dWorkspace);
    CusolverFrontend::AddVariableForArguments<size_t>(lwork_bytes);
    CusolverFrontend::AddDevicePointerForArguments(dinfo);
    CusolverFrontend::Execute("cusolverDnZEgesv");
    if (CusolverFrontend::Success()) {
        dA = (cuDoubleComplex *) CusolverFrontend::GetOutputDevicePointer();
        dipiv = (int*) CusolverFrontend::GetOutputDevicePointer();
        dX = (cuDoubleComplex *) CusolverFrontend::GetOutputDevicePointer();
        *niter = CusolverFrontend::GetOutputVariable<int>();
        dinfo = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnZYgesv(cusolverDnHandle_t handle, int n, int nrhs, cuDoubleComplex *dA, int ldda, int *dipiv, cuDoubleComplex *dB, int lddb, cuDoubleComplex *dX, int lddx, void *dWorkspace, size_t lwork_bytes, int *niter, int *dinfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nrhs);
    CusolverFrontend::AddDevicePointerForArguments(dA);
    CusolverFrontend::AddVariableForArguments<int>(ldda);
    CusolverFrontend::AddDevicePointerForArguments(dipiv);
    CusolverFrontend::AddDevicePointerForArguments(dB);
    CusolverFrontend::AddVariableForArguments<int>(lddb);
    CusolverFrontend::AddDevicePointerForArguments(dX);
    CusolverFrontend::AddVariableForArguments<int>(lddx);
    CusolverFrontend::AddDevicePointerForArguments(dWorkspace);
    CusolverFrontend::AddVariableForArguments<size_t>(lwork_bytes);
    CusolverFrontend::AddDevicePointerForArguments(dinfo);
    CusolverFrontend::Execute("cusolverDnZYgesv");
    if (CusolverFrontend::Success()) {
        dA = (cuDoubleComplex *) CusolverFrontend::GetOutputDevicePointer();
        dipiv = (int*) CusolverFrontend::GetOutputDevicePointer();
        dX = (cuDoubleComplex *) CusolverFrontend::GetOutputDevicePointer();
        *niter = CusolverFrontend::GetOutputVariable<int>();
        dinfo = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnCCgesv(cusolverDnHandle_t handle, int n, int nrhs, cuComplex *dA, int ldda, int *dipiv, cuComplex *dB, int lddb, cuComplex *dX, int lddx, void *dWorkspace, size_t lwork_bytes, int *niter, int *dinfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nrhs);
    CusolverFrontend::AddDevicePointerForArguments(dA);
    CusolverFrontend::AddVariableForArguments<int>(ldda);
    CusolverFrontend::AddDevicePointerForArguments(dipiv);
    CusolverFrontend::AddDevicePointerForArguments(dB);
    CusolverFrontend::AddVariableForArguments<int>(lddb);
    CusolverFrontend::AddDevicePointerForArguments(dX);
    CusolverFrontend::AddVariableForArguments<int>(lddx);
    CusolverFrontend::AddDevicePointerForArguments(dWorkspace);
    CusolverFrontend::AddVariableForArguments<size_t>(lwork_bytes);
    CusolverFrontend::AddDevicePointerForArguments(dinfo);
    CusolverFrontend::Execute("cusolverDnCCgesv");
    if (CusolverFrontend::Success()) {
        dA = (cuComplex *) CusolverFrontend::GetOutputDevicePointer();
        dipiv = (int*) CusolverFrontend::GetOutputDevicePointer();
        dX = (cuComplex *) CusolverFrontend::GetOutputDevicePointer();
        *niter = CusolverFrontend::GetOutputVariable<int>();
        dinfo = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnCKgesv(cusolverDnHandle_t handle, int n, int nrhs, cuComplex *dA, int ldda, int *dipiv, cuComplex *dB, int lddb, cuComplex *dX, int lddx, void *dWorkspace, size_t lwork_bytes, int *niter, int *dinfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nrhs);
    CusolverFrontend::AddDevicePointerForArguments(dA);
    CusolverFrontend::AddVariableForArguments<int>(ldda);
    CusolverFrontend::AddDevicePointerForArguments(dipiv);
    CusolverFrontend::AddDevicePointerForArguments(dB);
    CusolverFrontend::AddVariableForArguments<int>(lddb);
    CusolverFrontend::AddDevicePointerForArguments(dX);
    CusolverFrontend::AddVariableForArguments<int>(lddx);
    CusolverFrontend::AddDevicePointerForArguments(dWorkspace);
    CusolverFrontend::AddVariableForArguments<size_t>(lwork_bytes);
    CusolverFrontend::AddDevicePointerForArguments(dinfo);
    CusolverFrontend::Execute("cusolverDnCKgesv");
    if (CusolverFrontend::Success()) {
        dA = (cuComplex *) CusolverFrontend::GetOutputDevicePointer();
        dipiv = (int*) CusolverFrontend::GetOutputDevicePointer();
        dX = (cuComplex *) CusolverFrontend::GetOutputDevicePointer();
        *niter = CusolverFrontend::GetOutputVariable<int>();
        dinfo = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnCEgesv(cusolverDnHandle_t handle, int n, int nrhs, cuComplex *dA, int ldda, int *dipiv, cuComplex *dB, int lddb, cuComplex *dX, int lddx, void *dWorkspace, size_t lwork_bytes, int *niter, int *dinfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nrhs);
    CusolverFrontend::AddDevicePointerForArguments(dA);
    CusolverFrontend::AddVariableForArguments<int>(ldda);
    CusolverFrontend::AddDevicePointerForArguments(dipiv);
    CusolverFrontend::AddDevicePointerForArguments(dB);
    CusolverFrontend::AddVariableForArguments<int>(lddb);
    CusolverFrontend::AddDevicePointerForArguments(dX);
    CusolverFrontend::AddVariableForArguments<int>(lddx);
    CusolverFrontend::AddDevicePointerForArguments(dWorkspace);
    CusolverFrontend::AddVariableForArguments<size_t>(lwork_bytes);
    CusolverFrontend::AddDevicePointerForArguments(dinfo);
    CusolverFrontend::Execute("cusolverDnCEgesv");
    if (CusolverFrontend::Success()) {
        dA = (cuComplex *) CusolverFrontend::GetOutputDevicePointer();
        dipiv = (int*) CusolverFrontend::GetOutputDevicePointer();
        dX = (cuComplex *) CusolverFrontend::GetOutputDevicePointer();
        *niter = CusolverFrontend::GetOutputVariable<int>();
        dinfo = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnCYgesv(cusolverDnHandle_t handle, int n, int nrhs, cuComplex *dA, int ldda, int *dipiv, cuComplex *dB, int lddb, cuComplex *dX, int lddx, void *dWorkspace, size_t lwork_bytes, int *niter, int *dinfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nrhs);
    CusolverFrontend::AddDevicePointerForArguments(dA);
    CusolverFrontend::AddVariableForArguments<int>(ldda);
    CusolverFrontend::AddDevicePointerForArguments(dipiv);
    CusolverFrontend::AddDevicePointerForArguments(dB);
    CusolverFrontend::AddVariableForArguments<int>(lddb);
    CusolverFrontend::AddDevicePointerForArguments(dX);
    CusolverFrontend::AddVariableForArguments<int>(lddx);
    CusolverFrontend::AddDevicePointerForArguments(dWorkspace);
    CusolverFrontend::AddVariableForArguments<size_t>(lwork_bytes);
    CusolverFrontend::AddDevicePointerForArguments(dinfo);
    CusolverFrontend::Execute("cusolverDnCYgesv");
    if (CusolverFrontend::Success()) {
        dA = (cuComplex *) CusolverFrontend::GetOutputDevicePointer();
        dipiv = (int*) CusolverFrontend::GetOutputDevicePointer();
        dX = (cuComplex *) CusolverFrontend::GetOutputDevicePointer();
        *niter = CusolverFrontend::GetOutputVariable<int>();
        dinfo = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnDDgesv(cusolverDnHandle_t handle, int n, int nrhs, double *dA, int ldda, int *dipiv, double *dB, int lddb, double *dX, int lddx, void *dWorkspace, size_t lwork_bytes, int *niter, int *dinfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nrhs);
    CusolverFrontend::AddDevicePointerForArguments(dA);
    CusolverFrontend::AddVariableForArguments<int>(ldda);
    CusolverFrontend::AddDevicePointerForArguments(dipiv);
    CusolverFrontend::AddDevicePointerForArguments(dB);
    CusolverFrontend::AddVariableForArguments<int>(lddb);
    CusolverFrontend::AddDevicePointerForArguments(dX);
    CusolverFrontend::AddVariableForArguments<int>(lddx);
    CusolverFrontend::AddDevicePointerForArguments(dWorkspace);
    CusolverFrontend::AddVariableForArguments<size_t>(lwork_bytes);
    CusolverFrontend::AddDevicePointerForArguments(dinfo);
    CusolverFrontend::Execute("cusolverDnDDgesv");
    if (CusolverFrontend::Success()) {
        dA = (double *) CusolverFrontend::GetOutputDevicePointer();
        dipiv = (int*) CusolverFrontend::GetOutputDevicePointer();
        dX = (double *) CusolverFrontend::GetOutputDevicePointer();
        *niter = CusolverFrontend::GetOutputVariable<int>();
        dinfo = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnDSgesv(cusolverDnHandle_t handle, int n, int nrhs, double *dA, int ldda, int *dipiv, double *dB, int lddb, double *dX, int lddx, void *dWorkspace, size_t lwork_bytes, int *niter, int *dinfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nrhs);
    CusolverFrontend::AddDevicePointerForArguments(dA);
    CusolverFrontend::AddVariableForArguments<int>(ldda);
    CusolverFrontend::AddDevicePointerForArguments(dipiv);
    CusolverFrontend::AddDevicePointerForArguments(dB);
    CusolverFrontend::AddVariableForArguments<int>(lddb);
    CusolverFrontend::AddDevicePointerForArguments(dX);
    CusolverFrontend::AddVariableForArguments<int>(lddx);
    CusolverFrontend::AddDevicePointerForArguments(dWorkspace);
    CusolverFrontend::AddVariableForArguments<size_t>(lwork_bytes);
    CusolverFrontend::AddDevicePointerForArguments(dinfo);
    CusolverFrontend::Execute("cusolverDnDSgesv");
    if (CusolverFrontend::Success()) {
        dA = (double *) CusolverFrontend::GetOutputDevicePointer();
        dipiv = (int*) CusolverFrontend::GetOutputDevicePointer();
        dX = (double *) CusolverFrontend::GetOutputDevicePointer();
        *niter = CusolverFrontend::GetOutputVariable<int>();
        dinfo = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnDHgesv(cusolverDnHandle_t handle, int n, int nrhs, double *dA, int ldda, int *dipiv, double *dB, int lddb, double *dX, int lddx, void *dWorkspace, size_t lwork_bytes, int *niter, int *dinfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nrhs);
    CusolverFrontend::AddDevicePointerForArguments(dA);
    CusolverFrontend::AddVariableForArguments<int>(ldda);
    CusolverFrontend::AddDevicePointerForArguments(dipiv);
    CusolverFrontend::AddDevicePointerForArguments(dB);
    CusolverFrontend::AddVariableForArguments<int>(lddb);
    CusolverFrontend::AddDevicePointerForArguments(dX);
    CusolverFrontend::AddVariableForArguments<int>(lddx);
    CusolverFrontend::AddDevicePointerForArguments(dWorkspace);
    CusolverFrontend::AddVariableForArguments<size_t>(lwork_bytes);
    CusolverFrontend::AddDevicePointerForArguments(dinfo);
    CusolverFrontend::Execute("cusolverDnDHgesv");
    if (CusolverFrontend::Success()) {
        dA = (double *) CusolverFrontend::GetOutputDevicePointer();
        dipiv = (int*) CusolverFrontend::GetOutputDevicePointer();
        dX = (double *) CusolverFrontend::GetOutputDevicePointer();
        *niter = CusolverFrontend::GetOutputVariable<int>();
        dinfo = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnDBgesv(cusolverDnHandle_t handle, int n, int nrhs, double *dA, int ldda, int *dipiv, double *dB, int lddb, double *dX, int lddx, void *dWorkspace, size_t lwork_bytes, int *niter, int *dinfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nrhs);
    CusolverFrontend::AddDevicePointerForArguments(dA);
    CusolverFrontend::AddVariableForArguments<int>(ldda);
    CusolverFrontend::AddDevicePointerForArguments(dipiv);
    CusolverFrontend::AddDevicePointerForArguments(dB);
    CusolverFrontend::AddVariableForArguments<int>(lddb);
    CusolverFrontend::AddDevicePointerForArguments(dX);
    CusolverFrontend::AddVariableForArguments<int>(lddx);
    CusolverFrontend::AddDevicePointerForArguments(dWorkspace);
    CusolverFrontend::AddVariableForArguments<size_t>(lwork_bytes);
    CusolverFrontend::AddDevicePointerForArguments(dinfo);
    CusolverFrontend::Execute("cusolverDnDBgesv");
    if (CusolverFrontend::Success()) {
        dA = (double *) CusolverFrontend::GetOutputDevicePointer();
        dipiv = (int*) CusolverFrontend::GetOutputDevicePointer();
        dX = (double *) CusolverFrontend::GetOutputDevicePointer();
        *niter = CusolverFrontend::GetOutputVariable<int>();
        dinfo = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnDXgesv(cusolverDnHandle_t handle, int n, int nrhs, double *dA, int ldda, int *dipiv, double *dB, int lddb, double *dX, int lddx, void *dWorkspace, size_t lwork_bytes, int *niter, int *dinfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nrhs);
    CusolverFrontend::AddDevicePointerForArguments(dA);
    CusolverFrontend::AddVariableForArguments<int>(ldda);
    CusolverFrontend::AddDevicePointerForArguments(dipiv);
    CusolverFrontend::AddDevicePointerForArguments(dB);
    CusolverFrontend::AddVariableForArguments<int>(lddb);
    CusolverFrontend::AddDevicePointerForArguments(dX);
    CusolverFrontend::AddVariableForArguments<int>(lddx);
    CusolverFrontend::AddDevicePointerForArguments(dWorkspace);
    CusolverFrontend::AddVariableForArguments<size_t>(lwork_bytes);
    CusolverFrontend::AddDevicePointerForArguments(dinfo);
    CusolverFrontend::Execute("cusolverDnDXgesv");
    if (CusolverFrontend::Success()) {
        dA = (double *) CusolverFrontend::GetOutputDevicePointer();
        dipiv = (int*) CusolverFrontend::GetOutputDevicePointer();
        dX = (double *) CusolverFrontend::GetOutputDevicePointer();
        *niter = CusolverFrontend::GetOutputVariable<int>();
        dinfo = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnSSgesv(cusolverDnHandle_t handle, int n, int nrhs, float *dA, int ldda, int *dipiv, float *dB, int lddb, float *dX, int lddx, void *dWorkspace, size_t lwork_bytes, int *niter, int *dinfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nrhs);
    CusolverFrontend::AddDevicePointerForArguments(dA);
    CusolverFrontend::AddVariableForArguments<int>(ldda);
    CusolverFrontend::AddDevicePointerForArguments(dipiv);
    CusolverFrontend::AddDevicePointerForArguments(dB);
    CusolverFrontend::AddVariableForArguments<int>(lddb);
    CusolverFrontend::AddDevicePointerForArguments(dX);
    CusolverFrontend::AddVariableForArguments<int>(lddx);
    CusolverFrontend::AddDevicePointerForArguments(dWorkspace);
    CusolverFrontend::AddVariableForArguments<size_t>(lwork_bytes);
    CusolverFrontend::AddDevicePointerForArguments(dinfo);
    CusolverFrontend::Execute("cusolverDnSSgesv");
    if (CusolverFrontend::Success()) {
        dA = (float *) CusolverFrontend::GetOutputDevicePointer();
        dipiv = (int*) CusolverFrontend::GetOutputDevicePointer();
        dX = (float *) CusolverFrontend::GetOutputDevicePointer();
        *niter = CusolverFrontend::GetOutputVariable<int>();
        dinfo = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnSHgesv(cusolverDnHandle_t handle, int n, int nrhs, float *dA, int ldda, int *dipiv, float *dB, int lddb, float *dX, int lddx, void *dWorkspace, size_t lwork_bytes, int *niter, int *dinfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nrhs);
    CusolverFrontend::AddDevicePointerForArguments(dA);
    CusolverFrontend::AddVariableForArguments<int>(ldda);
    CusolverFrontend::AddDevicePointerForArguments(dipiv);
    CusolverFrontend::AddDevicePointerForArguments(dB);
    CusolverFrontend::AddVariableForArguments<int>(lddb);
    CusolverFrontend::AddDevicePointerForArguments(dX);
    CusolverFrontend::AddVariableForArguments<int>(lddx);
    CusolverFrontend::AddDevicePointerForArguments(dWorkspace);
    CusolverFrontend::AddVariableForArguments<size_t>(lwork_bytes);
    CusolverFrontend::AddDevicePointerForArguments(dinfo);
    CusolverFrontend::Execute("cusolverDnSHgesv");
    if (CusolverFrontend::Success()) {
        dA = (float *) CusolverFrontend::GetOutputDevicePointer();
        dipiv = (int*) CusolverFrontend::GetOutputDevicePointer();
        dX = (float *) CusolverFrontend::GetOutputDevicePointer();
        *niter = CusolverFrontend::GetOutputVariable<int>();
        dinfo = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnSBgesv(cusolverDnHandle_t handle, int n, int nrhs, float *dA, int ldda, int *dipiv, float *dB, int lddb, float *dX, int lddx, void *dWorkspace, size_t lwork_bytes, int *niter, int *dinfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nrhs);
    CusolverFrontend::AddDevicePointerForArguments(dA);
    CusolverFrontend::AddVariableForArguments<int>(ldda);
    CusolverFrontend::AddDevicePointerForArguments(dipiv);
    CusolverFrontend::AddDevicePointerForArguments(dB);
    CusolverFrontend::AddVariableForArguments<int>(lddb);
    CusolverFrontend::AddDevicePointerForArguments(dX);
    CusolverFrontend::AddVariableForArguments<int>(lddx);
    CusolverFrontend::AddDevicePointerForArguments(dWorkspace);
    CusolverFrontend::AddVariableForArguments<size_t>(lwork_bytes);
    CusolverFrontend::AddDevicePointerForArguments(dinfo);
    CusolverFrontend::Execute("cusolverDnSBgesv");
    if (CusolverFrontend::Success()) {
        dA = (float *) CusolverFrontend::GetOutputDevicePointer();
        dipiv = (int*) CusolverFrontend::GetOutputDevicePointer();
        dX = (float *) CusolverFrontend::GetOutputDevicePointer();
        *niter = CusolverFrontend::GetOutputVariable<int>();
        dinfo = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnSXgesv(cusolverDnHandle_t handle, int n, int nrhs, float *dA, int ldda, int *dipiv, float *dB, int lddb, float *dX, int lddx, void *dWorkspace, size_t lwork_bytes, int *niter, int *dinfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nrhs);
    CusolverFrontend::AddDevicePointerForArguments(dA);
    CusolverFrontend::AddVariableForArguments<int>(ldda);
    CusolverFrontend::AddDevicePointerForArguments(dipiv);
    CusolverFrontend::AddDevicePointerForArguments(dB);
    CusolverFrontend::AddVariableForArguments<int>(lddb);
    CusolverFrontend::AddDevicePointerForArguments(dX);
    CusolverFrontend::AddVariableForArguments<int>(lddx);
    CusolverFrontend::AddDevicePointerForArguments(dWorkspace);
    CusolverFrontend::AddVariableForArguments<size_t>(lwork_bytes);
    CusolverFrontend::AddDevicePointerForArguments(dinfo);
    CusolverFrontend::Execute("cusolverDnSXgesv");
    if (CusolverFrontend::Success()) {
        dA = (float *) CusolverFrontend::GetOutputDevicePointer();
        dipiv = (int*) CusolverFrontend::GetOutputDevicePointer();
        dX = (float *) CusolverFrontend::GetOutputDevicePointer();
        *niter = CusolverFrontend::GetOutputVariable<int>();
        dinfo = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnIRSXgesv_bufferSize(cusolverDnHandle_t handle, cusolverDnIRSParams_t gesv_irs_params, int n, int nrhs, size_t *lwork_bytes) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) gesv_irs_params);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nrhs);
    CusolverFrontend::Execute("cusolverDnIRSXgesv_bufferSize");
    if (CusolverFrontend::Success()) {
        *lwork_bytes = CusolverFrontend::GetOutputVariable<size_t>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnIRSXgesv(cusolverDnHandle_t handle, cusolverDnIRSParams_t gesv_irs_params, cusolverDnIRSInfos_t gesv_irs_infos, cusolver_int_t n, cusolver_int_t nrhs, void *dA, cusolver_int_t ldda, void *dB, cusolver_int_t lddb, void *dX, cusolver_int_t lddx, void *dWorkspace, size_t lwork_bytes, cusolver_int_t *niter, cusolver_int_t *dinfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) gesv_irs_params);
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) gesv_irs_infos);
    CusolverFrontend::AddVariableForArguments<cusolver_int_t>(n);
    CusolverFrontend::AddVariableForArguments<cusolver_int_t>(nrhs);
    CusolverFrontend::AddDevicePointerForArguments(dA);
    CusolverFrontend::AddVariableForArguments<cusolver_int_t>(ldda);
    CusolverFrontend::AddDevicePointerForArguments(dB);
    CusolverFrontend::AddVariableForArguments<cusolver_int_t>(lddb);
    CusolverFrontend::AddDevicePointerForArguments(dX);
    CusolverFrontend::AddVariableForArguments<cusolver_int_t>(lddx);
    CusolverFrontend::AddDevicePointerForArguments(dWorkspace);
    CusolverFrontend::AddVariableForArguments<size_t>(lwork_bytes);
    CusolverFrontend::AddDevicePointerForArguments(dinfo);
    CusolverFrontend::Execute("cusolverDnIRSXgesv");
    if (CusolverFrontend::Success()) {
        dA = (void *) CusolverFrontend::GetOutputDevicePointer();
        dX = (void *) CusolverFrontend::GetOutputDevicePointer();
        *niter = CusolverFrontend::GetOutputVariable<cusolver_int_t>();
        dinfo = (cusolver_int_t*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnSgeqrf_bufferSize(cusolverDnHandle_t handle, int m, int n, float *A, int lda, int *Lwork) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::Execute("cusolverDnSgeqrf_bufferSize");
    if (CusolverFrontend::Success()) {
        *Lwork = CusolverFrontend::GetOutputVariable<int>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnDgeqrf_bufferSize(cusolverDnHandle_t handle, int m, int n, double *A, int lda, int *Lwork) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::Execute("cusolverDnDgeqrf_bufferSize");
    if (CusolverFrontend::Success()) {
        *Lwork = CusolverFrontend::GetOutputVariable<int>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnCgeqrf_bufferSize(cusolverDnHandle_t handle, int m, int n, cuComplex *A, int lda, int *Lwork) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::Execute("cusolverDnCgeqrf_bufferSize");
    if (CusolverFrontend::Success()) {
        *Lwork = CusolverFrontend::GetOutputVariable<int>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnZgeqrf_bufferSize(cusolverDnHandle_t handle, int m, int n, cuDoubleComplex *A, int lda, int *Lwork) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::Execute("cusolverDnZgeqrf_bufferSize");
    if (CusolverFrontend::Success()) {
        *Lwork = CusolverFrontend::GetOutputVariable<int>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnSgeqrf(cusolverDnHandle_t handle, int m, int n, float *A, int lda, float *TAU, float *Workspace, int Lwork, int *devInfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(TAU);
    CusolverFrontend::AddDevicePointerForArguments(Workspace);
    CusolverFrontend::AddVariableForArguments<int>(Lwork);
    CusolverFrontend::AddDevicePointerForArguments(devInfo);
    CusolverFrontend::Execute("cusolverDnSgeqrf");
    if (CusolverFrontend::Success()) {
        A = (float*) CusolverFrontend::GetOutputDevicePointer();
        TAU = (float*) CusolverFrontend::GetOutputDevicePointer();
        devInfo = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnDgeqrf(cusolverDnHandle_t handle, int m, int n, double *A, int lda, double *TAU, double *Workspace, int Lwork, int *devInfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(TAU);
    CusolverFrontend::AddDevicePointerForArguments(Workspace);
    CusolverFrontend::AddVariableForArguments<int>(Lwork);
    CusolverFrontend::AddDevicePointerForArguments(devInfo);
    CusolverFrontend::Execute("cusolverDnDgeqrf");
    if (CusolverFrontend::Success()) {
        A = (double*) CusolverFrontend::GetOutputDevicePointer();
        TAU = (double*) CusolverFrontend::GetOutputDevicePointer();
        devInfo = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnCgeqrf(cusolverDnHandle_t handle, int m, int n, cuComplex *A, int lda, cuComplex *TAU, cuComplex *Workspace, int Lwork, int *devInfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(TAU);
    CusolverFrontend::AddDevicePointerForArguments(Workspace);
    CusolverFrontend::AddVariableForArguments<int>(Lwork);
    CusolverFrontend::AddDevicePointerForArguments(devInfo);
    CusolverFrontend::Execute("cusolverDnCgeqrf");
    if (CusolverFrontend::Success()) {
        A = (cuComplex*) CusolverFrontend::GetOutputDevicePointer();
        TAU = (cuComplex*) CusolverFrontend::GetOutputDevicePointer();
        devInfo = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnZgeqrf(cusolverDnHandle_t handle, int m, int n, cuDoubleComplex *A, int lda, cuDoubleComplex *TAU, cuDoubleComplex *Workspace, int Lwork, int *devInfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(TAU);
    CusolverFrontend::AddDevicePointerForArguments(Workspace);
    CusolverFrontend::AddVariableForArguments<int>(Lwork);
    CusolverFrontend::AddDevicePointerForArguments(devInfo);
    CusolverFrontend::Execute("cusolverDnZgeqrf");
    if (CusolverFrontend::Success()) {
        A = (cuDoubleComplex*) CusolverFrontend::GetOutputDevicePointer();
        TAU = (cuDoubleComplex*) CusolverFrontend::GetOutputDevicePointer();
        devInfo = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnGeqrf_bufferSize(cusolverDnHandle_t handle, cusolverDnParams_t params, int64_t m, int64_t n, cudaDataType dataTypeA, const void *A, int64_t lda, cudaDataType dataTypeTau, const void *tau, cudaDataType computeType, size_t *workspaceInBytes) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) params);
    CusolverFrontend::AddVariableForArguments<int64_t>(m);
    CusolverFrontend::AddVariableForArguments<int64_t>(n);
    CusolverFrontend::AddVariableForArguments<cudaDataType_t>(dataTypeA);
    CusolverFrontend::AddVariableForArguments<int64_t>(lda);
    CusolverFrontend::AddVariableForArguments<cudaDataType_t>(dataTypeTau);
    CusolverFrontend::AddVariableForArguments<cudaDataType_t>(computeType);
    switch(dataTypeA){
        case CUDA_R_32F:
            //float
            CusolverFrontend::AddDevicePointerForArguments((float *)A);
            CusolverFrontend::AddDevicePointerForArguments((float *)tau);
            break;
        case CUDA_R_64F:
            //double
            CusolverFrontend::AddDevicePointerForArguments((double *)A);
            CusolverFrontend::AddDevicePointerForArguments((double *)tau);
            break;
        case CUDA_C_32F:
            //cuComplex
            CusolverFrontend::AddDevicePointerForArguments((cuComplex *)A);
            CusolverFrontend::AddDevicePointerForArguments((cuComplex *)tau);
            break;
        case CUDA_C_64F:
            //cuDoubleComplex
            CusolverFrontend::AddDevicePointerForArguments((cuDoubleComplex *)A);
            CusolverFrontend::AddDevicePointerForArguments((cuDoubleComplex *)tau);
            break;
        default:
            throw "Type not supported by GVirtus!";
    }
    CusolverFrontend::Execute("cusolverDnGeqrf_bufferSize");
    if (CusolverFrontend::Success()) {
        *workspaceInBytes = CusolverFrontend::GetOutputVariable<size_t>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnGeqrf(cusolverDnHandle_t handle, cusolverDnParams_t params, int64_t m, int64_t n, cudaDataType dataTypeA, void *A, int64_t lda, cudaDataType dataTypeTau, void *tau, cudaDataType computeType, void *pBuffer, size_t workspaceInBytes, int *info) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) params);
    CusolverFrontend::AddVariableForArguments<int64_t>(m);
    CusolverFrontend::AddVariableForArguments<int64_t>(n);
    CusolverFrontend::AddVariableForArguments<cudaDataType_t>(dataTypeA);
    CusolverFrontend::AddVariableForArguments<int64_t>(lda);
    CusolverFrontend::AddVariableForArguments<cudaDataType_t>(dataTypeTau);
    CusolverFrontend::AddVariableForArguments<cudaDataType_t>(computeType);
    CusolverFrontend::AddDevicePointerForArguments(pBuffer);
    CusolverFrontend::AddVariableForArguments<size_t>(workspaceInBytes);
    CusolverFrontend::AddDevicePointerForArguments(info);
    switch(dataTypeA){
        case CUDA_R_32F:
            //float
            CusolverFrontend::AddDevicePointerForArguments((float *)A);
            CusolverFrontend::AddDevicePointerForArguments((float *)tau);
            break;
        case CUDA_R_64F:
            //double
            CusolverFrontend::AddDevicePointerForArguments((double *)A);
            CusolverFrontend::AddDevicePointerForArguments((double *)tau);
            break;
        case CUDA_C_32F:
            //cuComplex
            CusolverFrontend::AddDevicePointerForArguments((cuComplex *)A);
            CusolverFrontend::AddDevicePointerForArguments((cuComplex *)tau);
            break;
        case CUDA_C_64F:
            //cuDoubleComplex
            CusolverFrontend::AddDevicePointerForArguments((cuDoubleComplex *)A);
            CusolverFrontend::AddDevicePointerForArguments((cuDoubleComplex *)tau);
            break;
        default:
            throw "Type not supported by GVirtus!";
    }
    CusolverFrontend::Execute("cusolverDnGeqrf");
    if (CusolverFrontend::Success()) {
        info = (int*) CusolverFrontend::GetOutputDevicePointer();
        tau = CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnZZgels_bufferSize(cusolverDnHandle_t handle, int m, int n, int nrhs, cuDoubleComplex *dA, int ldda, cuDoubleComplex *dB, int lddb, cuDoubleComplex *dX, int lddx, void *dwork, size_t *lwork_bytes) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nrhs);
    CusolverFrontend::AddDevicePointerForArguments(dA);
    CusolverFrontend::AddVariableForArguments<int>(ldda);
    CusolverFrontend::AddDevicePointerForArguments(dB);
    CusolverFrontend::AddVariableForArguments<int>(lddb);
    CusolverFrontend::AddDevicePointerForArguments(dX);
    CusolverFrontend::AddVariableForArguments<int>(lddx);
    CusolverFrontend::AddDevicePointerForArguments(dwork);
    CusolverFrontend::Execute("cusolverDnZZgels_bufferSize");
    if (CusolverFrontend::Success()) {
        *lwork_bytes = CusolverFrontend::GetOutputVariable<size_t>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnZCgels_bufferSize(cusolverDnHandle_t handle, int m, int n, int nrhs, cuDoubleComplex *dA, int ldda, cuDoubleComplex *dB, int lddb, cuDoubleComplex *dX, int lddx, void *dwork, size_t *lwork_bytes) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nrhs);
    CusolverFrontend::AddDevicePointerForArguments(dA);
    CusolverFrontend::AddVariableForArguments<int>(ldda);
    CusolverFrontend::AddDevicePointerForArguments(dB);
    CusolverFrontend::AddVariableForArguments<int>(lddb);
    CusolverFrontend::AddDevicePointerForArguments(dX);
    CusolverFrontend::AddVariableForArguments<int>(lddx);
    CusolverFrontend::AddDevicePointerForArguments(dwork);
    CusolverFrontend::Execute("cusolverDnZCgels_bufferSize");
    if (CusolverFrontend::Success()) {
        *lwork_bytes = CusolverFrontend::GetOutputVariable<size_t>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnZKgels_bufferSize(cusolverDnHandle_t handle, int m, int n, int nrhs, cuDoubleComplex *dA, int ldda, cuDoubleComplex *dB, int lddb, cuDoubleComplex *dX, int lddx, void *dwork, size_t *lwork_bytes) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nrhs);
    CusolverFrontend::AddDevicePointerForArguments(dA);
    CusolverFrontend::AddVariableForArguments<int>(ldda);
    CusolverFrontend::AddDevicePointerForArguments(dB);
    CusolverFrontend::AddVariableForArguments<int>(lddb);
    CusolverFrontend::AddDevicePointerForArguments(dX);
    CusolverFrontend::AddVariableForArguments<int>(lddx);
    CusolverFrontend::AddDevicePointerForArguments(dwork);
    CusolverFrontend::Execute("cusolverDnZKgels_bufferSize");
    if (CusolverFrontend::Success()) {
        *lwork_bytes = CusolverFrontend::GetOutputVariable<size_t>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnZEgels_bufferSize(cusolverDnHandle_t handle, int m, int n, int nrhs, cuDoubleComplex *dA, int ldda, cuDoubleComplex *dB, int lddb, cuDoubleComplex *dX, int lddx, void *dwork, size_t *lwork_bytes) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nrhs);
    CusolverFrontend::AddDevicePointerForArguments(dA);
    CusolverFrontend::AddVariableForArguments<int>(ldda);
    CusolverFrontend::AddDevicePointerForArguments(dB);
    CusolverFrontend::AddVariableForArguments<int>(lddb);
    CusolverFrontend::AddDevicePointerForArguments(dX);
    CusolverFrontend::AddVariableForArguments<int>(lddx);
    CusolverFrontend::AddDevicePointerForArguments(dwork);
    CusolverFrontend::Execute("cusolverDnZEgels_bufferSize");
    if (CusolverFrontend::Success()) {
        *lwork_bytes = CusolverFrontend::GetOutputVariable<size_t>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnZYgels_bufferSize(cusolverDnHandle_t handle, int m, int n, int nrhs, cuDoubleComplex *dA, int ldda, cuDoubleComplex *dB, int lddb, cuDoubleComplex *dX, int lddx, void *dwork, size_t *lwork_bytes) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nrhs);
    CusolverFrontend::AddDevicePointerForArguments(dA);
    CusolverFrontend::AddVariableForArguments<int>(ldda);
    CusolverFrontend::AddDevicePointerForArguments(dB);
    CusolverFrontend::AddVariableForArguments<int>(lddb);
    CusolverFrontend::AddDevicePointerForArguments(dX);
    CusolverFrontend::AddVariableForArguments<int>(lddx);
    CusolverFrontend::AddDevicePointerForArguments(dwork);
    CusolverFrontend::Execute("cusolverDnZYgels_bufferSize");
    if (CusolverFrontend::Success()) {
        *lwork_bytes = CusolverFrontend::GetOutputVariable<size_t>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnCCgels_bufferSize(cusolverDnHandle_t handle, int m, int n, int nrhs, cuComplex *dA, int ldda, cuComplex *dB, int lddb, cuComplex *dX, int lddx, void *dwork, size_t *lwork_bytes) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nrhs);
    CusolverFrontend::AddDevicePointerForArguments(dA);
    CusolverFrontend::AddVariableForArguments<int>(ldda);
    CusolverFrontend::AddDevicePointerForArguments(dB);
    CusolverFrontend::AddVariableForArguments<int>(lddb);
    CusolverFrontend::AddDevicePointerForArguments(dX);
    CusolverFrontend::AddVariableForArguments<int>(lddx);
    CusolverFrontend::AddDevicePointerForArguments(dwork);
    CusolverFrontend::Execute("cusolverDnCCgels_bufferSize");
    if (CusolverFrontend::Success()) {
        *lwork_bytes = CusolverFrontend::GetOutputVariable<size_t>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnCKgels_bufferSize(cusolverDnHandle_t handle, int m, int n, int nrhs, cuComplex *dA, int ldda, cuComplex *dB, int lddb, cuComplex *dX, int lddx, void *dwork, size_t *lwork_bytes) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nrhs);
    CusolverFrontend::AddDevicePointerForArguments(dA);
    CusolverFrontend::AddVariableForArguments<int>(ldda);
    CusolverFrontend::AddDevicePointerForArguments(dB);
    CusolverFrontend::AddVariableForArguments<int>(lddb);
    CusolverFrontend::AddDevicePointerForArguments(dX);
    CusolverFrontend::AddVariableForArguments<int>(lddx);
    CusolverFrontend::AddDevicePointerForArguments(dwork);
    CusolverFrontend::Execute("cusolverDnCKgels_bufferSize");
    if (CusolverFrontend::Success()) {
        *lwork_bytes = CusolverFrontend::GetOutputVariable<size_t>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnCEgels_bufferSize(cusolverDnHandle_t handle, int m, int n, int nrhs, cuComplex *dA, int ldda, cuComplex *dB, int lddb, cuComplex *dX, int lddx, void *dwork, size_t *lwork_bytes) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nrhs);
    CusolverFrontend::AddDevicePointerForArguments(dA);
    CusolverFrontend::AddVariableForArguments<int>(ldda);
    CusolverFrontend::AddDevicePointerForArguments(dB);
    CusolverFrontend::AddVariableForArguments<int>(lddb);
    CusolverFrontend::AddDevicePointerForArguments(dX);
    CusolverFrontend::AddVariableForArguments<int>(lddx);
    CusolverFrontend::AddDevicePointerForArguments(dwork);
    CusolverFrontend::Execute("cusolverDnCEgels_bufferSize");
    if (CusolverFrontend::Success()) {
        *lwork_bytes = CusolverFrontend::GetOutputVariable<size_t>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnCYgels_bufferSize(cusolverDnHandle_t handle, int m, int n, int nrhs, cuComplex *dA, int ldda, cuComplex *dB, int lddb, cuComplex *dX, int lddx, void *dwork, size_t *lwork_bytes) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nrhs);
    CusolverFrontend::AddDevicePointerForArguments(dA);
    CusolverFrontend::AddVariableForArguments<int>(ldda);
    CusolverFrontend::AddDevicePointerForArguments(dB);
    CusolverFrontend::AddVariableForArguments<int>(lddb);
    CusolverFrontend::AddDevicePointerForArguments(dX);
    CusolverFrontend::AddVariableForArguments<int>(lddx);
    CusolverFrontend::AddDevicePointerForArguments(dwork);
    CusolverFrontend::Execute("cusolverDnCYgels_bufferSize");
    if (CusolverFrontend::Success()) {
        *lwork_bytes = CusolverFrontend::GetOutputVariable<size_t>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnDDgels_bufferSize(cusolverDnHandle_t handle, int m, int n, int nrhs, double *dA, int ldda, double *dB, int lddb, double *dX, int lddx, void *dwork, size_t *lwork_bytes) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nrhs);
    CusolverFrontend::AddDevicePointerForArguments(dA);
    CusolverFrontend::AddVariableForArguments<int>(ldda);
    CusolverFrontend::AddDevicePointerForArguments(dB);
    CusolverFrontend::AddVariableForArguments<int>(lddb);
    CusolverFrontend::AddDevicePointerForArguments(dX);
    CusolverFrontend::AddVariableForArguments<int>(lddx);
    CusolverFrontend::AddDevicePointerForArguments(dwork);
    CusolverFrontend::Execute("cusolverDnDDgels_bufferSize");
    if (CusolverFrontend::Success()) {
        *lwork_bytes = CusolverFrontend::GetOutputVariable<size_t>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnDSgels_bufferSize(cusolverDnHandle_t handle, int m, int n, int nrhs, double *dA, int ldda, double *dB, int lddb, double *dX, int lddx, void *dwork, size_t *lwork_bytes) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nrhs);
    CusolverFrontend::AddDevicePointerForArguments(dA);
    CusolverFrontend::AddVariableForArguments<int>(ldda);
    CusolverFrontend::AddDevicePointerForArguments(dB);
    CusolverFrontend::AddVariableForArguments<int>(lddb);
    CusolverFrontend::AddDevicePointerForArguments(dX);
    CusolverFrontend::AddVariableForArguments<int>(lddx);
    CusolverFrontend::AddDevicePointerForArguments(dwork);
    CusolverFrontend::Execute("cusolverDnDSgels_bufferSize");
    if (CusolverFrontend::Success()) {
        *lwork_bytes = CusolverFrontend::GetOutputVariable<size_t>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnDHgels_bufferSize(cusolverDnHandle_t handle, int m, int n, int nrhs, double *dA, int ldda, double *dB, int lddb, double *dX, int lddx, void *dwork, size_t *lwork_bytes) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nrhs);
    CusolverFrontend::AddDevicePointerForArguments(dA);
    CusolverFrontend::AddVariableForArguments<int>(ldda);
    CusolverFrontend::AddDevicePointerForArguments(dB);
    CusolverFrontend::AddVariableForArguments<int>(lddb);
    CusolverFrontend::AddDevicePointerForArguments(dX);
    CusolverFrontend::AddVariableForArguments<int>(lddx);
    CusolverFrontend::AddDevicePointerForArguments(dwork);
    CusolverFrontend::Execute("cusolverDnDHgels_bufferSize");
    if (CusolverFrontend::Success()) {
        *lwork_bytes = CusolverFrontend::GetOutputVariable<size_t>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnDBgels_bufferSize(cusolverDnHandle_t handle, int m, int n, int nrhs, double *dA, int ldda, double *dB, int lddb, double *dX, int lddx, void *dwork, size_t *lwork_bytes) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nrhs);
    CusolverFrontend::AddDevicePointerForArguments(dA);
    CusolverFrontend::AddVariableForArguments<int>(ldda);
    CusolverFrontend::AddDevicePointerForArguments(dB);
    CusolverFrontend::AddVariableForArguments<int>(lddb);
    CusolverFrontend::AddDevicePointerForArguments(dX);
    CusolverFrontend::AddVariableForArguments<int>(lddx);
    CusolverFrontend::AddDevicePointerForArguments(dwork);
    CusolverFrontend::Execute("cusolverDnDBgels_bufferSize");
    if (CusolverFrontend::Success()) {
        *lwork_bytes = CusolverFrontend::GetOutputVariable<size_t>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnDXgels_bufferSize(cusolverDnHandle_t handle, int m, int n, int nrhs, double *dA, int ldda, double *dB, int lddb, double *dX, int lddx, void *dwork, size_t *lwork_bytes) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nrhs);
    CusolverFrontend::AddDevicePointerForArguments(dA);
    CusolverFrontend::AddVariableForArguments<int>(ldda);
    CusolverFrontend::AddDevicePointerForArguments(dB);
    CusolverFrontend::AddVariableForArguments<int>(lddb);
    CusolverFrontend::AddDevicePointerForArguments(dX);
    CusolverFrontend::AddVariableForArguments<int>(lddx);
    CusolverFrontend::AddDevicePointerForArguments(dwork);
    CusolverFrontend::Execute("cusolverDnDXgels_bufferSize");
    if (CusolverFrontend::Success()) {
        *lwork_bytes = CusolverFrontend::GetOutputVariable<size_t>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnSSgels_bufferSize(cusolverDnHandle_t handle, int m, int n, int nrhs, float *dA, int ldda, float *dB, int lddb, float *dX, int lddx, void *dwork, size_t *lwork_bytes) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nrhs);
    CusolverFrontend::AddDevicePointerForArguments(dA);
    CusolverFrontend::AddVariableForArguments<int>(ldda);
    CusolverFrontend::AddDevicePointerForArguments(dB);
    CusolverFrontend::AddVariableForArguments<int>(lddb);
    CusolverFrontend::AddDevicePointerForArguments(dX);
    CusolverFrontend::AddVariableForArguments<int>(lddx);
    CusolverFrontend::AddDevicePointerForArguments(dwork);
    CusolverFrontend::Execute("cusolverDnSSgels_bufferSize");
    if (CusolverFrontend::Success()) {
        *lwork_bytes = CusolverFrontend::GetOutputVariable<size_t>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnSHgels_bufferSize(cusolverDnHandle_t handle, int m, int n, int nrhs, float *dA, int ldda, float *dB, int lddb, float *dX, int lddx, void *dwork, size_t *lwork_bytes) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nrhs);
    CusolverFrontend::AddDevicePointerForArguments(dA);
    CusolverFrontend::AddVariableForArguments<int>(ldda);
    CusolverFrontend::AddDevicePointerForArguments(dB);
    CusolverFrontend::AddVariableForArguments<int>(lddb);
    CusolverFrontend::AddDevicePointerForArguments(dX);
    CusolverFrontend::AddVariableForArguments<int>(lddx);
    CusolverFrontend::AddDevicePointerForArguments(dwork);
    CusolverFrontend::Execute("cusolverDnSHgels_bufferSize");
    if (CusolverFrontend::Success()) {
        *lwork_bytes = CusolverFrontend::GetOutputVariable<size_t>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnSBgels_bufferSize(cusolverDnHandle_t handle, int m, int n, int nrhs, float *dA, int ldda, float *dB, int lddb, float *dX, int lddx, void *dwork, size_t *lwork_bytes) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nrhs);
    CusolverFrontend::AddDevicePointerForArguments(dA);
    CusolverFrontend::AddVariableForArguments<int>(ldda);
    CusolverFrontend::AddDevicePointerForArguments(dB);
    CusolverFrontend::AddVariableForArguments<int>(lddb);
    CusolverFrontend::AddDevicePointerForArguments(dX);
    CusolverFrontend::AddVariableForArguments<int>(lddx);
    CusolverFrontend::AddDevicePointerForArguments(dwork);
    CusolverFrontend::Execute("cusolverDnSBgels_bufferSize");
    if (CusolverFrontend::Success()) {
        *lwork_bytes = CusolverFrontend::GetOutputVariable<size_t>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnSXgels_bufferSize(cusolverDnHandle_t handle, int m, int n, int nrhs, float *dA, int ldda, float *dB, int lddb, float *dX, int lddx, void *dwork, size_t *lwork_bytes) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nrhs);
    CusolverFrontend::AddDevicePointerForArguments(dA);
    CusolverFrontend::AddVariableForArguments<int>(ldda);
    CusolverFrontend::AddDevicePointerForArguments(dB);
    CusolverFrontend::AddVariableForArguments<int>(lddb);
    CusolverFrontend::AddDevicePointerForArguments(dX);
    CusolverFrontend::AddVariableForArguments<int>(lddx);
    CusolverFrontend::AddDevicePointerForArguments(dwork);
    CusolverFrontend::Execute("cusolverDnSXgels_bufferSize");
    if (CusolverFrontend::Success()) {
        *lwork_bytes = CusolverFrontend::GetOutputVariable<size_t>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnZZgels(cusolverDnHandle_t handle, int m, int n, int nrhs, cuDoubleComplex *dA, int ldda, cuDoubleComplex *dB, int lddb, cuDoubleComplex *dX, int lddx, void *dWorkspace, size_t lwork_bytes, int *niter, int *dinfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nrhs);
    CusolverFrontend::AddDevicePointerForArguments(dA);
    CusolverFrontend::AddVariableForArguments<int>(ldda);
    CusolverFrontend::AddDevicePointerForArguments(dB);
    CusolverFrontend::AddVariableForArguments<int>(lddb);
    CusolverFrontend::AddDevicePointerForArguments(dX);
    CusolverFrontend::AddVariableForArguments<int>(lddx);
    CusolverFrontend::AddDevicePointerForArguments(dWorkspace);
    CusolverFrontend::AddVariableForArguments<int>(lwork_bytes);
    CusolverFrontend::AddDevicePointerForArguments(dinfo);
    CusolverFrontend::Execute("cusolverDnZZgels");
    if (CusolverFrontend::Success()) {
        dX = (cuDoubleComplex*) CusolverFrontend::GetOutputDevicePointer();
        *niter = CusolverFrontend::GetOutputVariable<int>();
        dinfo = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnZCgels(cusolverDnHandle_t handle, int m, int n, int nrhs, cuDoubleComplex *dA, int ldda, cuDoubleComplex *dB, int lddb, cuDoubleComplex *dX, int lddx, void *dWorkspace, size_t lwork_bytes, int *niter, int *dinfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nrhs);
    CusolverFrontend::AddDevicePointerForArguments(dA);
    CusolverFrontend::AddVariableForArguments<int>(ldda);
    CusolverFrontend::AddDevicePointerForArguments(dB);
    CusolverFrontend::AddVariableForArguments<int>(lddb);
    CusolverFrontend::AddDevicePointerForArguments(dX);
    CusolverFrontend::AddVariableForArguments<int>(lddx);
    CusolverFrontend::AddDevicePointerForArguments(dWorkspace);
    CusolverFrontend::AddVariableForArguments<int>(lwork_bytes);
    CusolverFrontend::AddDevicePointerForArguments(dinfo);
    CusolverFrontend::Execute("cusolverDnZCgels");
    if (CusolverFrontend::Success()) {
        dX = (cuDoubleComplex*) CusolverFrontend::GetOutputDevicePointer();
        *niter = CusolverFrontend::GetOutputVariable<int>();
        dinfo = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnZKgels(cusolverDnHandle_t handle, int m, int n, int nrhs, cuDoubleComplex *dA, int ldda, cuDoubleComplex *dB, int lddb, cuDoubleComplex *dX, int lddx, void *dWorkspace, size_t lwork_bytes, int *niter, int *dinfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nrhs);
    CusolverFrontend::AddDevicePointerForArguments(dA);
    CusolverFrontend::AddVariableForArguments<int>(ldda);
    CusolverFrontend::AddDevicePointerForArguments(dB);
    CusolverFrontend::AddVariableForArguments<int>(lddb);
    CusolverFrontend::AddDevicePointerForArguments(dX);
    CusolverFrontend::AddVariableForArguments<int>(lddx);
    CusolverFrontend::AddDevicePointerForArguments(dWorkspace);
    CusolverFrontend::AddVariableForArguments<int>(lwork_bytes);
    CusolverFrontend::AddDevicePointerForArguments(dinfo);
    CusolverFrontend::Execute("cusolverDnZKgels");
    if (CusolverFrontend::Success()) {
        dX = (cuDoubleComplex*) CusolverFrontend::GetOutputDevicePointer();
        *niter = CusolverFrontend::GetOutputVariable<int>();
        dinfo = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnZEgels(cusolverDnHandle_t handle, int m, int n, int nrhs, cuDoubleComplex *dA, int ldda, cuDoubleComplex *dB, int lddb, cuDoubleComplex *dX, int lddx, void *dWorkspace, size_t lwork_bytes, int *niter, int *dinfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nrhs);
    CusolverFrontend::AddDevicePointerForArguments(dA);
    CusolverFrontend::AddVariableForArguments<int>(ldda);
    CusolverFrontend::AddDevicePointerForArguments(dB);
    CusolverFrontend::AddVariableForArguments<int>(lddb);
    CusolverFrontend::AddDevicePointerForArguments(dX);
    CusolverFrontend::AddVariableForArguments<int>(lddx);
    CusolverFrontend::AddDevicePointerForArguments(dWorkspace);
    CusolverFrontend::AddVariableForArguments<int>(lwork_bytes);
    CusolverFrontend::AddDevicePointerForArguments(dinfo);
    CusolverFrontend::Execute("cusolverDnZEgels");
    if (CusolverFrontend::Success()) {
        dX = (cuDoubleComplex*) CusolverFrontend::GetOutputDevicePointer();
        *niter = CusolverFrontend::GetOutputVariable<int>();
        dinfo = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnZYgels(cusolverDnHandle_t handle, int m, int n, int nrhs, cuDoubleComplex *dA, int ldda, cuDoubleComplex *dB, int lddb, cuDoubleComplex *dX, int lddx, void *dWorkspace, size_t lwork_bytes, int *niter, int *dinfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nrhs);
    CusolverFrontend::AddDevicePointerForArguments(dA);
    CusolverFrontend::AddVariableForArguments<int>(ldda);
    CusolverFrontend::AddDevicePointerForArguments(dB);
    CusolverFrontend::AddVariableForArguments<int>(lddb);
    CusolverFrontend::AddDevicePointerForArguments(dX);
    CusolverFrontend::AddVariableForArguments<int>(lddx);
    CusolverFrontend::AddDevicePointerForArguments(dWorkspace);
    CusolverFrontend::AddVariableForArguments<int>(lwork_bytes);
    CusolverFrontend::AddDevicePointerForArguments(dinfo);
    CusolverFrontend::Execute("cusolverDnZYgels");
    if (CusolverFrontend::Success()) {
        dX = (cuDoubleComplex*) CusolverFrontend::GetOutputDevicePointer();
        *niter = CusolverFrontend::GetOutputVariable<int>();
        dinfo = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnCCgels(cusolverDnHandle_t handle, int m, int n, int nrhs, cuComplex *dA, int ldda, cuComplex *dB, int lddb, cuComplex *dX, int lddx, void *dWorkspace, size_t lwork_bytes, int *niter, int *dinfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nrhs);
    CusolverFrontend::AddDevicePointerForArguments(dA);
    CusolverFrontend::AddVariableForArguments<int>(ldda);
    CusolverFrontend::AddDevicePointerForArguments(dB);
    CusolverFrontend::AddVariableForArguments<int>(lddb);
    CusolverFrontend::AddDevicePointerForArguments(dX);
    CusolverFrontend::AddVariableForArguments<int>(lddx);
    CusolverFrontend::AddDevicePointerForArguments(dWorkspace);
    CusolverFrontend::AddVariableForArguments<int>(lwork_bytes);
    CusolverFrontend::AddDevicePointerForArguments(dinfo);
    CusolverFrontend::Execute("cusolverDnCCgels");
    if (CusolverFrontend::Success()) {
        dX = (cuComplex*) CusolverFrontend::GetOutputDevicePointer();
        *niter = CusolverFrontend::GetOutputVariable<int>();
        dinfo = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnCKgels(cusolverDnHandle_t handle, int m, int n, int nrhs, cuComplex *dA, int ldda, cuComplex *dB, int lddb, cuComplex *dX, int lddx, void *dWorkspace, size_t lwork_bytes, int *niter, int *dinfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nrhs);
    CusolverFrontend::AddDevicePointerForArguments(dA);
    CusolverFrontend::AddVariableForArguments<int>(ldda);
    CusolverFrontend::AddDevicePointerForArguments(dB);
    CusolverFrontend::AddVariableForArguments<int>(lddb);
    CusolverFrontend::AddDevicePointerForArguments(dX);
    CusolverFrontend::AddVariableForArguments<int>(lddx);
    CusolverFrontend::AddDevicePointerForArguments(dWorkspace);
    CusolverFrontend::AddVariableForArguments<int>(lwork_bytes);
    CusolverFrontend::AddDevicePointerForArguments(dinfo);
    CusolverFrontend::Execute("cusolverDnCKgels");
    if (CusolverFrontend::Success()) {
        dX = (cuComplex*) CusolverFrontend::GetOutputDevicePointer();
        *niter = CusolverFrontend::GetOutputVariable<int>();
        dinfo = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnCEgels(cusolverDnHandle_t handle, int m, int n, int nrhs, cuComplex *dA, int ldda, cuComplex *dB, int lddb, cuComplex *dX, int lddx, void *dWorkspace, size_t lwork_bytes, int *niter, int *dinfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nrhs);
    CusolverFrontend::AddDevicePointerForArguments(dA);
    CusolverFrontend::AddVariableForArguments<int>(ldda);
    CusolverFrontend::AddDevicePointerForArguments(dB);
    CusolverFrontend::AddVariableForArguments<int>(lddb);
    CusolverFrontend::AddDevicePointerForArguments(dX);
    CusolverFrontend::AddVariableForArguments<int>(lddx);
    CusolverFrontend::AddDevicePointerForArguments(dWorkspace);
    CusolverFrontend::AddVariableForArguments<int>(lwork_bytes);
    CusolverFrontend::AddDevicePointerForArguments(dinfo);
    CusolverFrontend::Execute("cusolverDnCEgels");
    if (CusolverFrontend::Success()) {
        dX = (cuComplex*) CusolverFrontend::GetOutputDevicePointer();
        *niter = CusolverFrontend::GetOutputVariable<int>();
        dinfo = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnCYgels(cusolverDnHandle_t handle, int m, int n, int nrhs, cuComplex *dA, int ldda, cuComplex *dB, int lddb, cuComplex *dX, int lddx, void *dWorkspace, size_t lwork_bytes, int *niter, int *dinfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nrhs);
    CusolverFrontend::AddDevicePointerForArguments(dA);
    CusolverFrontend::AddVariableForArguments<int>(ldda);
    CusolverFrontend::AddDevicePointerForArguments(dB);
    CusolverFrontend::AddVariableForArguments<int>(lddb);
    CusolverFrontend::AddDevicePointerForArguments(dX);
    CusolverFrontend::AddVariableForArguments<int>(lddx);
    CusolverFrontend::AddDevicePointerForArguments(dWorkspace);
    CusolverFrontend::AddVariableForArguments<int>(lwork_bytes);
    CusolverFrontend::AddDevicePointerForArguments(dinfo);
    CusolverFrontend::Execute("cusolverDnCYgels");
    if (CusolverFrontend::Success()) {
        dX = (cuComplex*) CusolverFrontend::GetOutputDevicePointer();
        *niter = CusolverFrontend::GetOutputVariable<int>();
        dinfo = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnDDgels(cusolverDnHandle_t handle, int m, int n, int nrhs, double *dA, int ldda, double *dB, int lddb, double *dX, int lddx, void *dWorkspace, size_t lwork_bytes, int *niter, int *dinfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nrhs);
    CusolverFrontend::AddDevicePointerForArguments(dA);
    CusolverFrontend::AddVariableForArguments<int>(ldda);
    CusolverFrontend::AddDevicePointerForArguments(dB);
    CusolverFrontend::AddVariableForArguments<int>(lddb);
    CusolverFrontend::AddDevicePointerForArguments(dX);
    CusolverFrontend::AddVariableForArguments<int>(lddx);
    CusolverFrontend::AddDevicePointerForArguments(dWorkspace);
    CusolverFrontend::AddVariableForArguments<int>(lwork_bytes);
    CusolverFrontend::AddDevicePointerForArguments(dinfo);
    CusolverFrontend::Execute("cusolverDnDDgels");
    if (CusolverFrontend::Success()) {
        dX = (double*) CusolverFrontend::GetOutputDevicePointer();
        *niter = CusolverFrontend::GetOutputVariable<int>();
        dinfo = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnDSgels(cusolverDnHandle_t handle, int m, int n, int nrhs, double *dA, int ldda, double *dB, int lddb, double *dX, int lddx, void *dWorkspace, size_t lwork_bytes, int *niter, int *dinfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nrhs);
    CusolverFrontend::AddDevicePointerForArguments(dA);
    CusolverFrontend::AddVariableForArguments<int>(ldda);
    CusolverFrontend::AddDevicePointerForArguments(dB);
    CusolverFrontend::AddVariableForArguments<int>(lddb);
    CusolverFrontend::AddDevicePointerForArguments(dX);
    CusolverFrontend::AddVariableForArguments<int>(lddx);
    CusolverFrontend::AddDevicePointerForArguments(dWorkspace);
    CusolverFrontend::AddVariableForArguments<int>(lwork_bytes);
    CusolverFrontend::AddDevicePointerForArguments(dinfo);
    CusolverFrontend::Execute("cusolverDnDSgels");
    if (CusolverFrontend::Success()) {
        dX = (double*) CusolverFrontend::GetOutputDevicePointer();
        *niter = CusolverFrontend::GetOutputVariable<int>();
        dinfo = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnDHgels(cusolverDnHandle_t handle, int m, int n, int nrhs, double *dA, int ldda, double *dB, int lddb, double *dX, int lddx, void *dWorkspace, size_t lwork_bytes, int *niter, int *dinfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nrhs);
    CusolverFrontend::AddDevicePointerForArguments(dA);
    CusolverFrontend::AddVariableForArguments<int>(ldda);
    CusolverFrontend::AddDevicePointerForArguments(dB);
    CusolverFrontend::AddVariableForArguments<int>(lddb);
    CusolverFrontend::AddDevicePointerForArguments(dX);
    CusolverFrontend::AddVariableForArguments<int>(lddx);
    CusolverFrontend::AddDevicePointerForArguments(dWorkspace);
    CusolverFrontend::AddVariableForArguments<int>(lwork_bytes);
    CusolverFrontend::AddDevicePointerForArguments(dinfo);
    CusolverFrontend::Execute("cusolverDnDHgels");
    if (CusolverFrontend::Success()) {
        dX = (double*) CusolverFrontend::GetOutputDevicePointer();
        *niter = CusolverFrontend::GetOutputVariable<int>();
        dinfo = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnDBgels(cusolverDnHandle_t handle, int m, int n, int nrhs, double *dA, int ldda, double *dB, int lddb, double *dX, int lddx, void *dWorkspace, size_t lwork_bytes, int *niter, int *dinfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nrhs);
    CusolverFrontend::AddDevicePointerForArguments(dA);
    CusolverFrontend::AddVariableForArguments<int>(ldda);
    CusolverFrontend::AddDevicePointerForArguments(dB);
    CusolverFrontend::AddVariableForArguments<int>(lddb);
    CusolverFrontend::AddDevicePointerForArguments(dX);
    CusolverFrontend::AddVariableForArguments<int>(lddx);
    CusolverFrontend::AddDevicePointerForArguments(dWorkspace);
    CusolverFrontend::AddVariableForArguments<int>(lwork_bytes);
    CusolverFrontend::AddDevicePointerForArguments(dinfo);
    CusolverFrontend::Execute("cusolverDnDBgels");
    if (CusolverFrontend::Success()) {
        dX = (double*) CusolverFrontend::GetOutputDevicePointer();
        *niter = CusolverFrontend::GetOutputVariable<int>();
        dinfo = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnDXgels(cusolverDnHandle_t handle, int m, int n, int nrhs, double *dA, int ldda, double *dB, int lddb, double *dX, int lddx, void *dWorkspace, size_t lwork_bytes, int *niter, int *dinfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nrhs);
    CusolverFrontend::AddDevicePointerForArguments(dA);
    CusolverFrontend::AddVariableForArguments<int>(ldda);
    CusolverFrontend::AddDevicePointerForArguments(dB);
    CusolverFrontend::AddVariableForArguments<int>(lddb);
    CusolverFrontend::AddDevicePointerForArguments(dX);
    CusolverFrontend::AddVariableForArguments<int>(lddx);
    CusolverFrontend::AddDevicePointerForArguments(dWorkspace);
    CusolverFrontend::AddVariableForArguments<int>(lwork_bytes);
    CusolverFrontend::AddDevicePointerForArguments(dinfo);
    CusolverFrontend::Execute("cusolverDnDXgels");
    if (CusolverFrontend::Success()) {
        dX = (double*) CusolverFrontend::GetOutputDevicePointer();
        *niter = CusolverFrontend::GetOutputVariable<int>();
        dinfo = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnSSgels(cusolverDnHandle_t handle, int m, int n, int nrhs, float *dA, int ldda, float *dB, int lddb, float *dX, int lddx, void *dWorkspace, size_t lwork_bytes, int *niter, int *dinfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nrhs);
    CusolverFrontend::AddDevicePointerForArguments(dA);
    CusolverFrontend::AddVariableForArguments<int>(ldda);
    CusolverFrontend::AddDevicePointerForArguments(dB);
    CusolverFrontend::AddVariableForArguments<int>(lddb);
    CusolverFrontend::AddDevicePointerForArguments(dX);
    CusolverFrontend::AddVariableForArguments<int>(lddx);
    CusolverFrontend::AddDevicePointerForArguments(dWorkspace);
    CusolverFrontend::AddVariableForArguments<int>(lwork_bytes);
    CusolverFrontend::AddDevicePointerForArguments(dinfo);
    CusolverFrontend::Execute("cusolverDnSSgels");
    if (CusolverFrontend::Success()) {
        dX = (float*) CusolverFrontend::GetOutputDevicePointer();
        *niter = CusolverFrontend::GetOutputVariable<int>();
        dinfo = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnSHgels(cusolverDnHandle_t handle, int m, int n, int nrhs, float *dA, int ldda, float *dB, int lddb, float *dX, int lddx, void *dWorkspace, size_t lwork_bytes, int *niter, int *dinfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nrhs);
    CusolverFrontend::AddDevicePointerForArguments(dA);
    CusolverFrontend::AddVariableForArguments<int>(ldda);
    CusolverFrontend::AddDevicePointerForArguments(dB);
    CusolverFrontend::AddVariableForArguments<int>(lddb);
    CusolverFrontend::AddDevicePointerForArguments(dX);
    CusolverFrontend::AddVariableForArguments<int>(lddx);
    CusolverFrontend::AddDevicePointerForArguments(dWorkspace);
    CusolverFrontend::AddVariableForArguments<int>(lwork_bytes);
    CusolverFrontend::AddDevicePointerForArguments(dinfo);
    CusolverFrontend::Execute("cusolverDnSHgels");
    if (CusolverFrontend::Success()) {
        dX = (float*) CusolverFrontend::GetOutputDevicePointer();
        *niter = CusolverFrontend::GetOutputVariable<int>();
        dinfo = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnSBgels(cusolverDnHandle_t handle, int m, int n, int nrhs, float *dA, int ldda, float *dB, int lddb, float *dX, int lddx, void *dWorkspace, size_t lwork_bytes, int *niter, int *dinfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nrhs);
    CusolverFrontend::AddDevicePointerForArguments(dA);
    CusolverFrontend::AddVariableForArguments<int>(ldda);
    CusolverFrontend::AddDevicePointerForArguments(dB);
    CusolverFrontend::AddVariableForArguments<int>(lddb);
    CusolverFrontend::AddDevicePointerForArguments(dX);
    CusolverFrontend::AddVariableForArguments<int>(lddx);
    CusolverFrontend::AddDevicePointerForArguments(dWorkspace);
    CusolverFrontend::AddVariableForArguments<int>(lwork_bytes);
    CusolverFrontend::AddDevicePointerForArguments(dinfo);
    CusolverFrontend::Execute("cusolverDnSBgels");
    if (CusolverFrontend::Success()) {
        dX = (float*) CusolverFrontend::GetOutputDevicePointer();
        *niter = CusolverFrontend::GetOutputVariable<int>();
        dinfo = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnSXgels(cusolverDnHandle_t handle, int m, int n, int nrhs, float *dA, int ldda, float *dB, int lddb, float *dX, int lddx, void *dWorkspace, size_t lwork_bytes, int *niter, int *dinfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nrhs);
    CusolverFrontend::AddDevicePointerForArguments(dA);
    CusolverFrontend::AddVariableForArguments<int>(ldda);
    CusolverFrontend::AddDevicePointerForArguments(dB);
    CusolverFrontend::AddVariableForArguments<int>(lddb);
    CusolverFrontend::AddDevicePointerForArguments(dX);
    CusolverFrontend::AddVariableForArguments<int>(lddx);
    CusolverFrontend::AddDevicePointerForArguments(dWorkspace);
    CusolverFrontend::AddVariableForArguments<int>(lwork_bytes);
    CusolverFrontend::AddDevicePointerForArguments(dinfo);
    CusolverFrontend::Execute("cusolverDnSXgels");
    if (CusolverFrontend::Success()) {
        dX = (float*) CusolverFrontend::GetOutputDevicePointer();
        *niter = CusolverFrontend::GetOutputVariable<int>();
        dinfo = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnIRSXgels_bufferSize(cusolverDnHandle_t handle, cusolverDnIRSParams_t gels_irs_params, int m, int n, int nrhs, size_t *lwork_bytes) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) gels_irs_params);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nrhs);
    CusolverFrontend::Execute("cusolverDnIRSXgels_bufferSize");
    if (CusolverFrontend::Success()) {
        *lwork_bytes = CusolverFrontend::GetOutputVariable<size_t>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnIRSXgels(cusolverDnHandle_t handle, cusolverDnIRSParams_t gels_irs_params, cusolverDnIRSInfos_t gels_irs_infos, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, void *dA, cusolver_int_t ldda, void *dB, cusolver_int_t lddb, void *dX, cusolver_int_t lddx, void *dWorkspace, size_t lwork_bytes, cusolver_int_t *niter, cusolver_int_t *dinfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) gels_irs_params);
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) gels_irs_infos);
    CusolverFrontend::AddVariableForArguments<cusolver_int_t>(m);
    CusolverFrontend::AddVariableForArguments<cusolver_int_t>(n);
    CusolverFrontend::AddVariableForArguments<cusolver_int_t>(nrhs);
    CusolverFrontend::AddDevicePointerForArguments(dA);
    CusolverFrontend::AddVariableForArguments<cusolver_int_t>(ldda);
    CusolverFrontend::AddDevicePointerForArguments(dB);
    CusolverFrontend::AddVariableForArguments<cusolver_int_t>(lddb);
    CusolverFrontend::AddDevicePointerForArguments(dX);
    CusolverFrontend::AddVariableForArguments<cusolver_int_t>(lddx);
    CusolverFrontend::AddDevicePointerForArguments(dWorkspace);
    CusolverFrontend::AddVariableForArguments<size_t>(lwork_bytes);
    CusolverFrontend::AddDevicePointerForArguments(dinfo);
    CusolverFrontend::Execute("cusolverDnIRSXgels");
    if (CusolverFrontend::Success()) {
        dA = (void *) CusolverFrontend::GetOutputDevicePointer();
        dX = (void *) CusolverFrontend::GetOutputDevicePointer();
        *niter = CusolverFrontend::GetOutputVariable<cusolver_int_t>();
        dinfo = (cusolver_int_t*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnSormqr_bufferSize(cusolverDnHandle_t handle, cublasSideMode_t side, cublasOperation_t trans, int m, int n, int k, const float *A, int lda, const float *tau, const float *C, int ldc, int *lwork) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cublasSideMode_t>(side);
    CusolverFrontend::AddVariableForArguments<cublasOperation_t>(trans);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(k);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(tau);
    CusolverFrontend::AddDevicePointerForArguments(C);
    CusolverFrontend::AddVariableForArguments<int>(ldc);
    CusolverFrontend::Execute("cusolverDnSormqr_bufferSize");
    if (CusolverFrontend::Success()) {
        *lwork = CusolverFrontend::GetOutputVariable<int>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnDormqr_bufferSize(cusolverDnHandle_t handle, cublasSideMode_t side, cublasOperation_t trans, int m, int n, int k, const double *A, int lda, const double *tau, const double *C, int ldc, int *lwork) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cublasSideMode_t>(side);
    CusolverFrontend::AddVariableForArguments<cublasOperation_t>(trans);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(k);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(tau);
    CusolverFrontend::AddDevicePointerForArguments(C);
    CusolverFrontend::AddVariableForArguments<int>(ldc);
    CusolverFrontend::Execute("cusolverDnDormqr_bufferSize");
    if (CusolverFrontend::Success()) {
        *lwork = CusolverFrontend::GetOutputVariable<int>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnCunmqr_bufferSize(cusolverDnHandle_t handle, cublasSideMode_t side, cublasOperation_t trans, int m, int n, int k, const cuComplex *A, int lda, const cuComplex *tau, const cuComplex *C, int ldc, int *lwork) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cublasSideMode_t>(side);
    CusolverFrontend::AddVariableForArguments<cublasOperation_t>(trans);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(k);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(tau);
    CusolverFrontend::AddDevicePointerForArguments(C);
    CusolverFrontend::AddVariableForArguments<int>(ldc);
    CusolverFrontend::Execute("cusolverDnCunmqr_bufferSize");
    if (CusolverFrontend::Success()) {
        *lwork = CusolverFrontend::GetOutputVariable<int>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnZunmqr_bufferSize(cusolverDnHandle_t handle, cublasSideMode_t side, cublasOperation_t trans, int m, int n, int k, const cuDoubleComplex *A, int lda, const cuDoubleComplex *tau, const cuDoubleComplex *C, int ldc, int *lwork) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cublasSideMode_t>(side);
    CusolverFrontend::AddVariableForArguments<cublasOperation_t>(trans);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(k);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(tau);
    CusolverFrontend::AddDevicePointerForArguments(C);
    CusolverFrontend::AddVariableForArguments<int>(ldc);
    CusolverFrontend::Execute("cusolverDnZunmqr_bufferSize");
    if (CusolverFrontend::Success()) {
        *lwork = CusolverFrontend::GetOutputVariable<int>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnSormqr(cusolverDnHandle_t handle, cublasSideMode_t side, cublasOperation_t trans, int m, int n, int k, const float *A, int lda, const float *tau, float *C, int ldc, float *work, int lwork, int *devInfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cublasSideMode_t>(side);
    CusolverFrontend::AddVariableForArguments<cublasOperation_t>(trans);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(k);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(tau);
    CusolverFrontend::AddDevicePointerForArguments(C);
    CusolverFrontend::AddVariableForArguments<int>(ldc);
    CusolverFrontend::AddDevicePointerForArguments(work);
    CusolverFrontend::AddVariableForArguments<int>(lwork);
    CusolverFrontend::AddDevicePointerForArguments(devInfo);
    CusolverFrontend::Execute("cusolverDnSormqr");
    if (CusolverFrontend::Success()) {
        tau = (float*) CusolverFrontend::GetOutputDevicePointer();
        devInfo = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnDormqr(cusolverDnHandle_t handle, cublasSideMode_t side, cublasOperation_t trans, int m, int n, int k, const double *A, int lda, const double *tau, double *C, int ldc, double *work, int lwork, int *devInfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cublasSideMode_t>(side);
    CusolverFrontend::AddVariableForArguments<cublasOperation_t>(trans);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(k);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(tau);
    CusolverFrontend::AddDevicePointerForArguments(C);
    CusolverFrontend::AddVariableForArguments<int>(ldc);
    CusolverFrontend::AddDevicePointerForArguments(work);
    CusolverFrontend::AddVariableForArguments<int>(lwork);
    CusolverFrontend::AddDevicePointerForArguments(devInfo);
    CusolverFrontend::Execute("cusolverDnDormqr");
    if (CusolverFrontend::Success()) {
        tau = (double*) CusolverFrontend::GetOutputDevicePointer();
        devInfo = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnCunmqr(cusolverDnHandle_t handle, cublasSideMode_t side, cublasOperation_t trans, int m, int n, int k, const cuComplex *A, int lda, const cuComplex *tau, cuComplex *C, int ldc, cuComplex *work, int lwork, int *devInfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cublasSideMode_t>(side);
    CusolverFrontend::AddVariableForArguments<cublasOperation_t>(trans);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(k);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(tau);
    CusolverFrontend::AddDevicePointerForArguments(C);
    CusolverFrontend::AddVariableForArguments<int>(ldc);
    CusolverFrontend::AddDevicePointerForArguments(work);
    CusolverFrontend::AddVariableForArguments<int>(lwork);
    CusolverFrontend::AddDevicePointerForArguments(devInfo);
    CusolverFrontend::Execute("cusolverDnCunmqr");
    if (CusolverFrontend::Success()) {
        tau = (cuComplex*) CusolverFrontend::GetOutputDevicePointer();
        devInfo = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnZunmqr(cusolverDnHandle_t handle, cublasSideMode_t side, cublasOperation_t trans, int m, int n, int k, const cuDoubleComplex *A, int lda, const cuDoubleComplex *tau, cuDoubleComplex *C, int ldc, cuDoubleComplex *work, int lwork, int *devInfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cublasSideMode_t>(side);
    CusolverFrontend::AddVariableForArguments<cublasOperation_t>(trans);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(k);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(tau);
    CusolverFrontend::AddDevicePointerForArguments(C);
    CusolverFrontend::AddVariableForArguments<int>(ldc);
    CusolverFrontend::AddDevicePointerForArguments(work);
    CusolverFrontend::AddVariableForArguments<int>(lwork);
    CusolverFrontend::AddDevicePointerForArguments(devInfo);
    CusolverFrontend::Execute("cusolverDnZunmqr");
    if (CusolverFrontend::Success()) {
        tau = (cuDoubleComplex*) CusolverFrontend::GetOutputDevicePointer();
        devInfo = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnSorgqr_bufferSize(cusolverDnHandle_t handle, int m, int n, int k, const float *A, int lda, const float *tau, int *lwork) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(k);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(tau);
    CusolverFrontend::Execute("cusolverDnSorgqr_bufferSize");
    if (CusolverFrontend::Success()) {
        *lwork = CusolverFrontend::GetOutputVariable<int>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnDorgqr_bufferSize(cusolverDnHandle_t handle, int m, int n, int k, const double *A, int lda, const double *tau, int *lwork) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(k);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(tau);
    CusolverFrontend::Execute("cusolverDnDorgqr_bufferSize");
    if (CusolverFrontend::Success()) {
        *lwork = CusolverFrontend::GetOutputVariable<int>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnCungqr_bufferSize(cusolverDnHandle_t handle, int m, int n, int k, const cuComplex *A, int lda, const cuComplex *tau, int *lwork) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(k);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(tau);
    CusolverFrontend::Execute("cusolverDnCungqr_bufferSize");
    if (CusolverFrontend::Success()) {
        *lwork = CusolverFrontend::GetOutputVariable<int>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnZungqr_bufferSize(cusolverDnHandle_t handle, int m, int n, int k, const cuDoubleComplex *A, int lda, const cuDoubleComplex *tau, int *lwork) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(k);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(tau);
    CusolverFrontend::Execute("cusolverDnZungqr_bufferSize");
    if (CusolverFrontend::Success()) {
        *lwork = CusolverFrontend::GetOutputVariable<int>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnSorgqr(cusolverDnHandle_t handle, int m, int n, int k, float *A, int lda, const float *tau, float *work, int lwork, int *devInfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(k);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(tau);
    CusolverFrontend::AddDevicePointerForArguments(work);
    CusolverFrontend::AddVariableForArguments<int>(lwork);
    CusolverFrontend::AddDevicePointerForArguments(devInfo);
    CusolverFrontend::Execute("cusolverDnSorgqr");
    if (CusolverFrontend::Success()) {
        tau = (float*) CusolverFrontend::GetOutputDevicePointer();
        devInfo = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnDorgqr(cusolverDnHandle_t handle, int m, int n, int k, double *A, int lda, const double *tau, double *work, int lwork, int *devInfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(k);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(tau);
    CusolverFrontend::AddDevicePointerForArguments(work);
    CusolverFrontend::AddVariableForArguments<int>(lwork);
    CusolverFrontend::AddDevicePointerForArguments(devInfo);
    CusolverFrontend::Execute("cusolverDnDorgqr");
    if (CusolverFrontend::Success()) {
        tau = (double*) CusolverFrontend::GetOutputDevicePointer();
        devInfo = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnCungqr(cusolverDnHandle_t handle, int m, int n, int k, cuComplex *A, int lda, const cuComplex *tau, cuComplex *work, int lwork, int *devInfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(k);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(tau);
    CusolverFrontend::AddDevicePointerForArguments(work);
    CusolverFrontend::AddVariableForArguments<int>(lwork);
    CusolverFrontend::AddDevicePointerForArguments(devInfo);
    CusolverFrontend::Execute("cusolverDnCungqr");
    if (CusolverFrontend::Success()) {
        tau = (cuComplex*) CusolverFrontend::GetOutputDevicePointer();
        devInfo = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnZungqr(cusolverDnHandle_t handle, int m, int n, int k, cuDoubleComplex *A, int lda, const cuDoubleComplex *tau, cuDoubleComplex *work, int lwork, int *devInfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(k);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(tau);
    CusolverFrontend::AddDevicePointerForArguments(work);
    CusolverFrontend::AddVariableForArguments<int>(lwork);
    CusolverFrontend::AddDevicePointerForArguments(devInfo);
    CusolverFrontend::Execute("cusolverDnZungqr");
    if (CusolverFrontend::Success()) {
        tau = (cuDoubleComplex*) CusolverFrontend::GetOutputDevicePointer();
        devInfo = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnSsytrf_bufferSize(cusolverDnHandle_t handle, int n, float *A, int lda, int *lwork) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::Execute("cusolverDnSsytrf_bufferSize");
    if (CusolverFrontend::Success()) {
        *lwork = CusolverFrontend::GetOutputVariable<int>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnDsytrf_bufferSize(cusolverDnHandle_t handle, int n, double *A, int lda, int *lwork) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::Execute("cusolverDnDsytrf_bufferSize");
    if (CusolverFrontend::Success()) {
        *lwork = CusolverFrontend::GetOutputVariable<int>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnCsytrf_bufferSize(cusolverDnHandle_t handle, int n, cuComplex *A, int lda, int *lwork) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::Execute("cusolverDnCsytrf_bufferSize");
    if (CusolverFrontend::Success()) {
        *lwork = CusolverFrontend::GetOutputVariable<int>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnZsytrf_bufferSize(cusolverDnHandle_t handle, int n, cuDoubleComplex *A, int lda, int *lwork) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::Execute("cusolverDnZsytrf_bufferSize");
    if (CusolverFrontend::Success()) {
        *lwork = CusolverFrontend::GetOutputVariable<int>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnSsytrf(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, float *A, int lda, int *ipiv, float *work, int lwork, int *devInfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(ipiv);
    CusolverFrontend::AddDevicePointerForArguments(work);
    CusolverFrontend::AddVariableForArguments<int>(lwork);
    CusolverFrontend::AddDevicePointerForArguments(devInfo);
    CusolverFrontend::Execute("cusolverDnSsytrf");
    if (CusolverFrontend::Success()) {
        ipiv = (int*) CusolverFrontend::GetOutputDevicePointer();
        devInfo = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnDsytrf(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, double *A, int lda, int *ipiv, double *work, int lwork, int *devInfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(ipiv);
    CusolverFrontend::AddDevicePointerForArguments(work);
    CusolverFrontend::AddVariableForArguments<int>(lwork);
    CusolverFrontend::AddDevicePointerForArguments(devInfo);
    CusolverFrontend::Execute("cusolverDnDsytrf");
    if (CusolverFrontend::Success()) {
        ipiv = (int*) CusolverFrontend::GetOutputDevicePointer();
        devInfo = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnCsytrf(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuComplex *A, int lda, int *ipiv, cuComplex *work, int lwork, int *devInfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(ipiv);
    CusolverFrontend::AddDevicePointerForArguments(work);
    CusolverFrontend::AddVariableForArguments<int>(lwork);
    CusolverFrontend::AddDevicePointerForArguments(devInfo);
    CusolverFrontend::Execute("cusolverDnCsytrf");
    if (CusolverFrontend::Success()) {
        ipiv = (int*) CusolverFrontend::GetOutputDevicePointer();
        devInfo = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnZsytrf(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuDoubleComplex *A, int lda, int *ipiv, cuDoubleComplex *work, int lwork, int *devInfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(ipiv);
    CusolverFrontend::AddDevicePointerForArguments(work);
    CusolverFrontend::AddVariableForArguments<int>(lwork);
    CusolverFrontend::AddDevicePointerForArguments(devInfo);
    CusolverFrontend::Execute("cusolverDnZsytrf");
    if (CusolverFrontend::Success()) {
        ipiv = (int*) CusolverFrontend::GetOutputDevicePointer();
        devInfo = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnSpotrfBatched(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, float **A, int lda, int *infoArray, int batchSize) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(infoArray);
    CusolverFrontend::AddVariableForArguments<int>(batchSize);
    CusolverFrontend::Execute("cusolverDnSpotrfBatched");
    if (CusolverFrontend::Success()) {
        infoArray = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnDpotrfBatched(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, double **A, int lda, int *infoArray, int batchSize) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(infoArray);
    CusolverFrontend::AddVariableForArguments<int>(batchSize);
    CusolverFrontend::Execute("cusolverDnDpotrfBatched");
    if (CusolverFrontend::Success()) {
        infoArray = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnCpotrfBatched(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuComplex **A, int lda, int *infoArray, int batchSize) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(infoArray);
    CusolverFrontend::AddVariableForArguments<int>(batchSize);
    CusolverFrontend::Execute("cusolverDnCpotrfBatched");
    if (CusolverFrontend::Success()) {
        infoArray = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnZpotrfBatched(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuDoubleComplex **A, int lda, int *infoArray, int batchSize) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(infoArray);
    CusolverFrontend::AddVariableForArguments<int>(batchSize);
    CusolverFrontend::Execute("cusolverDnZpotrfBatched");
    if (CusolverFrontend::Success()) {
        infoArray = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnSpotrsBatched(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, int nrhs, float **A, int lda, float **B, int ldb, int *infoArray, int batchSize) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nrhs);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(B);
    CusolverFrontend::AddVariableForArguments<int>(ldb);
    CusolverFrontend::AddDevicePointerForArguments(infoArray);
    CusolverFrontend::AddVariableForArguments<int>(batchSize);
    CusolverFrontend::Execute("cusolverDnSpotrsBatched");
    if (CusolverFrontend::Success()) {
        infoArray = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnDpotrsBatched(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, int nrhs, double **A, int lda, double **B, int ldb, int *infoArray, int batchSize) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nrhs);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(B);
    CusolverFrontend::AddVariableForArguments<int>(ldb);
    CusolverFrontend::AddDevicePointerForArguments(infoArray);
    CusolverFrontend::AddVariableForArguments<int>(batchSize);
    CusolverFrontend::Execute("cusolverDnDpotrsBatched");
    if (CusolverFrontend::Success()) {
        infoArray = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnCpotrsBatched(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, int nrhs, cuComplex **A, int lda, cuComplex **B, int ldb, int *infoArray, int batchSize) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nrhs);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(B);
    CusolverFrontend::AddVariableForArguments<int>(ldb);
    CusolverFrontend::AddDevicePointerForArguments(infoArray);
    CusolverFrontend::AddVariableForArguments<int>(batchSize);
    CusolverFrontend::Execute("cusolverDnCpotrsBatched");
    if (CusolverFrontend::Success()) {
        infoArray = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnZpotrsBatched(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, int nrhs, cuDoubleComplex **A, int lda, cuDoubleComplex **B, int ldb, int *infoArray, int batchSize) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nrhs);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(B);
    CusolverFrontend::AddVariableForArguments<int>(ldb);
    CusolverFrontend::AddDevicePointerForArguments(infoArray);
    CusolverFrontend::AddVariableForArguments<int>(batchSize);
    CusolverFrontend::Execute("cusolverDnZpotrsBatched");
    if (CusolverFrontend::Success()) {
        infoArray = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}
