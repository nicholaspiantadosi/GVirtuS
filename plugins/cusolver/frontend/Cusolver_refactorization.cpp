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
#include "Utilities.h"

using namespace std;

extern "C" cusolverStatus_t CUSOLVERAPI cusolverRfAccessBundledFactorsDevice(cusolverRfHandle_t handle, int* nnzM, int** Mp, int** Mi, double** Mx) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(*nnzM);
    CusolverFrontend::AddDevicePointerForArguments(*Mp);
    CusolverFrontend::AddDevicePointerForArguments(*Mi);
    CusolverFrontend::AddDevicePointerForArguments(*Mx);
    CusolverFrontend::Execute("cusolverRfAccessBundledFactorsDevice");
    if(CusolverFrontend::Success()) {
        *nnzM = CusolverFrontend::GetOutputVariable<int>();
        *Mp = (int *) CusolverFrontend::GetOutputDevicePointer();
        *Mi = (int *) CusolverFrontend::GetOutputDevicePointer();
        *Mx = (double *) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverRfExtractBundledFactorsHost(cusolverRfHandle_t handle, int* nnzM, int** Mp, int** Mi, double** Mx) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::Execute("cusolverRfExtractBundledFactorsHost");
    if(CusolverFrontend::Success()) {
        *nnzM = CusolverFrontend::GetOutputVariable<int>();
        *Mp = CusolverFrontend::GetOutputHostPointer<int>(*nnzM);
        *Mi = CusolverFrontend::GetOutputHostPointer<int>(*nnzM);
        *Mx = CusolverFrontend::GetOutputHostPointer<double>(*nnzM);
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverRfAnalyze(cusolverRfHandle_t handle) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::Execute("cusolverRfAnalyze");
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverRfSetupDevice(int n, int nnzA, int* csrRowPtrA, int* csrColIndA, double* csrValA, int nnzL, int* csrRowPtrL, int* csrColIndL, double* csrValL, int nnzU, int* csrRowPtrU, int* csrColIndU, double* csrValU, int* P, int* Q, cusolverRfHandle_t handle) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nnzA);
    CusolverFrontend::AddDevicePointerForArguments(csrRowPtrA);
    CusolverFrontend::AddDevicePointerForArguments(csrColIndA);
    CusolverFrontend::AddDevicePointerForArguments(csrValA);
    CusolverFrontend::AddVariableForArguments<int>(nnzL);
    CusolverFrontend::AddDevicePointerForArguments(csrRowPtrL);
    CusolverFrontend::AddDevicePointerForArguments(csrColIndL);
    CusolverFrontend::AddDevicePointerForArguments(csrValL);
    CusolverFrontend::AddVariableForArguments<int>(nnzU);
    CusolverFrontend::AddDevicePointerForArguments(csrRowPtrU);
    CusolverFrontend::AddDevicePointerForArguments(csrColIndU);
    CusolverFrontend::AddDevicePointerForArguments(csrValU);
    CusolverFrontend::AddDevicePointerForArguments(P);
    CusolverFrontend::AddDevicePointerForArguments(Q);
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::Execute("cusolverRfSetupDevice");
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverRfSetupHost(int n, int nnzA, int* csrRowPtrA, int* csrColIndA, double* csrValA, int nnzL, int* csrRowPtrL, int* csrColIndL, double* csrValL, int nnzU, int* csrRowPtrU, int* csrColIndU, double* csrValU, int* P, int* Q, cusolverRfHandle_t handle) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nnzA);
    CusolverFrontend::AddHostPointerForArguments(csrRowPtrA, n + 1);
    CusolverFrontend::AddHostPointerForArguments(csrColIndA, nnzA);
    CusolverFrontend::AddHostPointerForArguments(csrValA, nnzA);
    CusolverFrontend::AddVariableForArguments<int>(nnzL);
    CusolverFrontend::AddHostPointerForArguments(csrRowPtrL, n + 1);
    CusolverFrontend::AddHostPointerForArguments(csrColIndL, nnzL);
    CusolverFrontend::AddHostPointerForArguments(csrValL, nnzL);
    CusolverFrontend::AddVariableForArguments<int>(nnzU);
    CusolverFrontend::AddHostPointerForArguments(csrRowPtrU, n + 1);
    CusolverFrontend::AddHostPointerForArguments(csrColIndU, nnzU);
    CusolverFrontend::AddHostPointerForArguments(csrValU, nnzU);
    CusolverFrontend::AddHostPointerForArguments(P, n);
    CusolverFrontend::AddHostPointerForArguments(Q, n);
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::Execute("cusolverRfSetupHost");
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverRfCreate(cusolverRfHandle_t *handle) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddHostPointerForArguments<cusolverRfHandle_t>(handle);
    CusolverFrontend::Execute("cusolverRfCreate");
    if(CusolverFrontend::Success())
        *handle = CusolverFrontend::GetOutputVariable<cusolverRfHandle_t>();
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverRfDestroy(cusolverRfHandle_t handle) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::Execute("cusolverRfDestroy");
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverRfRefactor(cusolverRfHandle_t handle) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::Execute("cusolverRfRefactor");
    return CusolverFrontend::GetExitCode();
}