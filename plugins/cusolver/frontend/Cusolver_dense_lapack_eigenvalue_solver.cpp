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

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnSgebrd_bufferSize(cusolverDnHandle_t handle, int m, int n, int *Lwork) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::Execute("cusolverDnSgebrd_bufferSize");
    if (CusolverFrontend::Success()) {
        *Lwork = CusolverFrontend::GetOutputVariable<int>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnDgebrd_bufferSize(cusolverDnHandle_t handle, int m, int n, int *Lwork) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::Execute("cusolverDnDgebrd_bufferSize");
    if (CusolverFrontend::Success()) {
        *Lwork = CusolverFrontend::GetOutputVariable<int>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnCgebrd_bufferSize(cusolverDnHandle_t handle, int m, int n, int *Lwork) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::Execute("cusolverDnCgebrd_bufferSize");
    if (CusolverFrontend::Success()) {
        *Lwork = CusolverFrontend::GetOutputVariable<int>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnZgebrd_bufferSize(cusolverDnHandle_t handle, int m, int n, int *Lwork) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::Execute("cusolverDnZgebrd_bufferSize");
    if (CusolverFrontend::Success()) {
        *Lwork = CusolverFrontend::GetOutputVariable<int>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnSgebrd(cusolverDnHandle_t handle, int m, int n, float *A, int lda, float *D, float *E, float *TAUQ, float *TAUP, float *Work, int Lwork, int *devInfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(D);
    CusolverFrontend::AddDevicePointerForArguments(E);
    CusolverFrontend::AddDevicePointerForArguments(TAUQ);
    CusolverFrontend::AddDevicePointerForArguments(TAUP);
    CusolverFrontend::AddDevicePointerForArguments(Work);
    CusolverFrontend::AddVariableForArguments<int>(Lwork);
    CusolverFrontend::AddDevicePointerForArguments(devInfo);
    CusolverFrontend::Execute("cusolverDnSgebrd");
    if (CusolverFrontend::Success()) {
        D = (float*) CusolverFrontend::GetOutputDevicePointer();
        E = (float*) CusolverFrontend::GetOutputDevicePointer();
        TAUQ = (float*) CusolverFrontend::GetOutputDevicePointer();
        TAUP = (float*) CusolverFrontend::GetOutputDevicePointer();
        devInfo = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnDgebrd(cusolverDnHandle_t handle, int m, int n, double *A, int lda, double *D, double *E, double *TAUQ, double *TAUP, double *Work, int Lwork, int *devInfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(D);
    CusolverFrontend::AddDevicePointerForArguments(E);
    CusolverFrontend::AddDevicePointerForArguments(TAUQ);
    CusolverFrontend::AddDevicePointerForArguments(TAUP);
    CusolverFrontend::AddDevicePointerForArguments(Work);
    CusolverFrontend::AddVariableForArguments<int>(Lwork);
    CusolverFrontend::AddDevicePointerForArguments(devInfo);
    CusolverFrontend::Execute("cusolverDnDgebrd");
    if (CusolverFrontend::Success()) {
        D = (double*) CusolverFrontend::GetOutputDevicePointer();
        E = (double*) CusolverFrontend::GetOutputDevicePointer();
        TAUQ = (double*) CusolverFrontend::GetOutputDevicePointer();
        TAUP = (double*) CusolverFrontend::GetOutputDevicePointer();
        devInfo = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnCgebrd(cusolverDnHandle_t handle, int m, int n, cuComplex *A, int lda, float *D, float *E, cuComplex *TAUQ, cuComplex *TAUP, cuComplex *Work, int Lwork, int *devInfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(D);
    CusolverFrontend::AddDevicePointerForArguments(E);
    CusolverFrontend::AddDevicePointerForArguments(TAUQ);
    CusolverFrontend::AddDevicePointerForArguments(TAUP);
    CusolverFrontend::AddDevicePointerForArguments(Work);
    CusolverFrontend::AddVariableForArguments<int>(Lwork);
    CusolverFrontend::AddDevicePointerForArguments(devInfo);
    CusolverFrontend::Execute("cusolverDnCgebrd");
    if (CusolverFrontend::Success()) {
        D = (float*) CusolverFrontend::GetOutputDevicePointer();
        E = (float*) CusolverFrontend::GetOutputDevicePointer();
        TAUQ = (cuComplex*) CusolverFrontend::GetOutputDevicePointer();
        TAUP = (cuComplex*) CusolverFrontend::GetOutputDevicePointer();
        devInfo = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnZgebrd(cusolverDnHandle_t handle, int m, int n, cuDoubleComplex *A, int lda, double *D, double *E, cuDoubleComplex *TAUQ, cuDoubleComplex *TAUP, cuDoubleComplex *Work, int Lwork, int *devInfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(D);
    CusolverFrontend::AddDevicePointerForArguments(E);
    CusolverFrontend::AddDevicePointerForArguments(TAUQ);
    CusolverFrontend::AddDevicePointerForArguments(TAUP);
    CusolverFrontend::AddDevicePointerForArguments(Work);
    CusolverFrontend::AddVariableForArguments<int>(Lwork);
    CusolverFrontend::AddDevicePointerForArguments(devInfo);
    CusolverFrontend::Execute("cusolverDnZgebrd");
    if (CusolverFrontend::Success()) {
        D = (double*) CusolverFrontend::GetOutputDevicePointer();
        E = (double*) CusolverFrontend::GetOutputDevicePointer();
        TAUQ = (cuDoubleComplex*) CusolverFrontend::GetOutputDevicePointer();
        TAUP = (cuDoubleComplex*) CusolverFrontend::GetOutputDevicePointer();
        devInfo = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}