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

extern "C" cusparseStatus_t CUSPARSEAPI cusparseGetVersion(cusparseHandle_t handle, int* version){
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CusparseFrontend::AddHostPointerForArguments<int>(version);
    CusparseFrontend::Execute("cusparseGetVersion");
    if(CusparseFrontend::Success())
        *version = (int) CusparseFrontend::GetOutputVariable<int>();
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseCreate(cusparseHandle_t *handle) {
  CusparseFrontend::Prepare();
  CusparseFrontend::AddHostPointerForArguments<cusparseHandle_t>(handle);
  CusparseFrontend::Execute("cusparseCreate");
  if(CusparseFrontend::Success())
      *handle = CusparseFrontend::GetOutputVariable<cusparseHandle_t>();
  return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseDestroy(cusparseHandle_t handle){
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CusparseFrontend::Execute("cusparseDestroy");
    return CusparseFrontend::GetExitCode();
}

extern "C" const char * CUSPARSEAPI cusparseGetErrorString(cusparseStatus_t status){
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<cusparseStatus_t>(status);
    CusparseFrontend::Execute("cusparseGetErrorString");
    char* error_string = strdup(CusparseFrontend::GetOutputString());
    return error_string;
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseSetStream(cusparseHandle_t handle, cudaStream_t streamId){
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CusparseFrontend::AddVariableForArguments<long long int>((long long int)streamId);
    CusparseFrontend::Execute("cusparseSetStream");
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseGetStream(cusparseHandle_t handle, cudaStream_t *streamId){
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CusparseFrontend::AddHostPointerForArguments<cudaStream_t>(streamId);
    CusparseFrontend::Execute("cusparseGetStream");
    if(CusparseFrontend::Success())
        *streamId = (cudaStream_t) CusparseFrontend::GetOutputVariable<long long int>();
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseGetProperty(libraryPropertyType_t type, int * value){
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<libraryPropertyType_t>(type);
    CusparseFrontend::AddHostPointerForArguments<int>(value);
    CusparseFrontend::Execute("cusparseGetProperty");
    if(CusparseFrontend::Success())
        *value = CusparseFrontend::GetOutputVariable<int>();
    return CusparseFrontend::GetExitCode();
}