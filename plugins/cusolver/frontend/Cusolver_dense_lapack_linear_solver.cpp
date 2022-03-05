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
