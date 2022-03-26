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
 *              Department of Science andTechnology
 */

#include <iostream>
#include <cstdio>
#include <string>

#include "CusolverMgFrontend.h"

using namespace std;

extern "C" cusolverStatus_t CUSOLVERAPI cusolverMgSyevd_bufferSize(cusolverMgHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int N, void *array_d_A[], int IA, int JA, cudaLibMgMatrixDesc_t descrA, void *W, cudaDataType_t dataTypeW, cudaDataType_t computeType, int64_t *lwork) {
    CusolverMgFrontend::Prepare();
    CusolverMgFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverMgFrontend::AddVariableForArguments<cusolverEigMode_t>(jobz);
    CusolverMgFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverMgFrontend::AddVariableForArguments<int>(N);
    CusolverMgFrontend::AddHostPointerForArguments(array_d_A, N*N);
    CusolverMgFrontend::AddVariableForArguments<int>(IA);
    CusolverMgFrontend::AddVariableForArguments<int>(JA);
    CusolverMgFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusolverMgFrontend::AddHostPointerForArguments((size_t*)W, N);
    CusolverMgFrontend::AddVariableForArguments<cudaDataType_t>(dataTypeW);
    CusolverMgFrontend::AddVariableForArguments<cudaDataType_t>(computeType);
    CusolverMgFrontend::Execute("cusolverMgPotrf_bufferSize");
    if (CusolverMgFrontend::Success()) {
        *lwork = CusolverMgFrontend::GetOutputVariable<int64_t>();
    }
    return CusolverMgFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverMgSyevd(cusolverMgHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int N, void *array_d_A[], int IA, int JA, cudaLibMgMatrixDesc_t descrA, void *W, cudaDataType_t dataTypeW, cudaDataType_t computeType, void *array_d_work[], int64_t lwork, int *info) {
    CusolverMgFrontend::Prepare();
    CusolverMgFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverMgFrontend::AddVariableForArguments<cusolverEigMode_t>(jobz);
    CusolverMgFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverMgFrontend::AddVariableForArguments<int>(N);
    CusolverMgFrontend::AddHostPointerForArguments(array_d_A, N*N);
    CusolverMgFrontend::AddVariableForArguments<int>(IA);
    CusolverMgFrontend::AddVariableForArguments<int>(JA);
    CusolverMgFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusolverMgFrontend::AddHostPointerForArguments((size_t*)W, N);
    CusolverMgFrontend::AddVariableForArguments<cudaDataType_t>(dataTypeW);
    CusolverMgFrontend::AddVariableForArguments<cudaDataType_t>(computeType);
    CusolverMgFrontend::AddVariableForArguments<int64_t>(lwork);
    CusolverMgFrontend::AddHostPointerForArguments(array_d_work, sizeof(computeType) * lwork);
    CusolverMgFrontend::Execute("cusolverMgPotrf");
    if (CusolverMgFrontend::Success()) {
        *info = CusolverMgFrontend::GetOutputVariable<int>();
    }
    return CusolverMgFrontend::GetExitCode();
}