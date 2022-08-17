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

extern "C" cusolverStatus_t CUSOLVERAPI cusolverMgCreate(cusolverMgHandle_t *handle) {
    CusolverMgFrontend::Prepare();
    CusolverMgFrontend::AddHostPointerForArguments<cusolverMgHandle_t>(handle);
    CusolverMgFrontend::Execute("cusolverMgCreate");
    if(CusolverMgFrontend::Success())
        *handle = CusolverMgFrontend::GetOutputVariable<cusolverMgHandle_t>();
    return CusolverMgFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverMgDestroy(cusolverMgHandle_t handle) {
    CusolverMgFrontend::Prepare();
    CusolverMgFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverMgFrontend::Execute("cusolverMgDestroy");
    return CusolverMgFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverMgDeviceSelect(cusolverMgHandle_t handle, int nbDevices, int* deviceId) {
    CusolverMgFrontend::Prepare();
    CusolverMgFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverMgFrontend::AddVariableForArguments<int>(nbDevices);
    CusolverMgFrontend::AddHostPointerForArguments(deviceId, nbDevices);
    CusolverMgFrontend::Execute("cusolverMgDeviceSelect");
    return CusolverMgFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverMgCreateDeviceGrid(cudaLibMgGrid_t* grid, int32_t numRowDevices, int32_t numColDevices, const int32_t *deviceId, cusolverMgGridMapping_t mapping) {
    CusolverMgFrontend::Prepare();
    CusolverMgFrontend::AddVariableForArguments<int32_t>(numRowDevices);
    CusolverMgFrontend::AddVariableForArguments<int32_t>(numColDevices);
    CusolverMgFrontend::AddHostPointerForArguments(deviceId, numColDevices);
    CusolverMgFrontend::AddVariableForArguments<cusolverMgGridMapping_t>(mapping);
    CusolverMgFrontend::Execute("cusolverMgCreateDeviceGrid");
    if(CusolverMgFrontend::Success())
        *grid = CusolverMgFrontend::GetOutputVariable<cudaLibMgGrid_t>();
    return CusolverMgFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverMgDestroyGrid(cudaLibMgGrid_t grid) {
    CusolverMgFrontend::Prepare();
    CusolverMgFrontend::AddVariableForArguments<size_t>((size_t) grid);
    CusolverMgFrontend::Execute("cusolverMgDestroyGrid");
    return CusolverMgFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverMgCreateMatrixDesc(cudaLibMgMatrixDesc_t * desc, int64_t numRows, int64_t numCols, int64_t rowBlockSize, int64_t colBlockSize, cudaDataType_t dataType, const cudaLibMgGrid_t grid) {
    CusolverMgFrontend::Prepare();
    CusolverMgFrontend::AddVariableForArguments<int64_t>(numRows);
    CusolverMgFrontend::AddVariableForArguments<int64_t>(numCols);
    CusolverMgFrontend::AddVariableForArguments<int64_t>(rowBlockSize);
    CusolverMgFrontend::AddVariableForArguments<int64_t>(colBlockSize);
    CusolverMgFrontend::AddVariableForArguments<cudaDataType_t>(dataType);
    CusolverMgFrontend::AddVariableForArguments<size_t>((size_t)grid);
    CusolverMgFrontend::Execute("cusolverMgCreateMatrixDesc");
    if(CusolverMgFrontend::Success())
        *desc = CusolverMgFrontend::GetOutputVariable<cudaLibMgMatrixDesc_t>();
    return CusolverMgFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverMgDestroyMatrixDesc(cudaLibMgMatrixDesc_t desc) {
    CusolverMgFrontend::Prepare();
    CusolverMgFrontend::AddVariableForArguments<size_t>((size_t) desc);
    CusolverMgFrontend::Execute("cusolverMgDestroyMatrixDesc");
    return CusolverMgFrontend::GetExitCode();
}
