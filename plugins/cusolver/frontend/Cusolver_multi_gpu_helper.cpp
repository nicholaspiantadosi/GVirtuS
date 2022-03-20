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

#include "CusolverFrontend.h"

using namespace std;

extern "C" cusolverStatus_t CUSOLVERAPI cusolverMgCreate(cusolverMgHandle_t *handle) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddHostPointerForArguments<cusolverMgHandle_t>(handle);
    CusolverFrontend::Execute("cusolverMgCreate");
    if(CusolverFrontend::Success())
        *handle = CusolverFrontend::GetOutputVariable<cusolverMgHandle_t>();
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverMgDestroy(cusolverMgHandle_t handle) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::Execute("cusolverMgDestroy");
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverMgDeviceSelect(cusolverMgHandle_t handle, int nbDevices, int* deviceId) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(nbDevices);
    CusolverFrontend::AddHostPointerForArguments(deviceId, nbDevices);
    CusolverFrontend::Execute("cusolverMgDeviceSelect");
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverMgCreateDeviceGrid(cudaLibMgGrid_t* grid, int32_t numRowDevices, int32_t numColDevices, const int32_t *deviceId, cusolverMgGridMapping_t mapping) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<int32_t>(numRowDevices);
    CusolverFrontend::AddVariableForArguments<int32_t>(numColDevices);
    CusolverFrontend::AddHostPointerForArguments(deviceId, numColDevices);
    CusolverFrontend::AddVariableForArguments<cusolverMgGridMapping_t>(mapping);
    CusolverFrontend::Execute("cusolverMgCreateDeviceGrid");
    if(CusolverFrontend::Success())
        *grid = CusolverFrontend::GetOutputVariable<cudaLibMgGrid_t>();
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverMgDestroyGrid(cudaLibMgGrid_t grid) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) grid);
    CusolverFrontend::Execute("cusolverMgDestroyGrid");
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverMgCreateMatrixDesc(cudaLibMgMatrixDesc_t * desc, int64_t numRows, int64_t numCols, int64_t rowBlockSize, int64_t colBlockSize, cudaDataType_t dataType, const cudaLibMgGrid_t grid) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<int64_t>(numRows);
    CusolverFrontend::AddVariableForArguments<int64_t>(numCols);
    CusolverFrontend::AddVariableForArguments<int64_t>(rowBlockSize);
    CusolverFrontend::AddVariableForArguments<int64_t>(colBlockSize);
    CusolverFrontend::AddVariableForArguments<cudaDataType_t>(dataType);
    CusolverFrontend::AddVariableForArguments<size_t>((size_t)grid);
    CusolverFrontend::Execute("cusolverMgCreateMatrixDesc");
    if(CusolverFrontend::Success())
        *desc = CusolverFrontend::GetOutputVariable<cudaLibMgMatrixDesc_t>();
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverMgDestroyMatrixDesc(cudaLibMgMatrixDesc_t desc) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) desc);
    CusolverFrontend::Execute("cusolverMgDestroyMatrixDesc");
    return CusolverFrontend::GetExitCode();
}