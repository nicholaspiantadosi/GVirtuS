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

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnSorgbr_bufferSize(cusolverDnHandle_t handle, cublasSideMode_t side, int m, int n, int k, const float *A, int lda, const float *tau, int *lwork) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cublasSideMode_t>(side);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(k);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(tau);
    CusolverFrontend::Execute("cusolverDnSorgbr_bufferSize");
    if (CusolverFrontend::Success()) {
        *lwork = CusolverFrontend::GetOutputVariable<int>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnDorgbr_bufferSize(cusolverDnHandle_t handle, cublasSideMode_t side, int m, int n, int k, const double *A, int lda, const double *tau, int *lwork) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cublasSideMode_t>(side);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(k);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(tau);
    CusolverFrontend::Execute("cusolverDnDorgbr_bufferSize");
    if (CusolverFrontend::Success()) {
        *lwork = CusolverFrontend::GetOutputVariable<int>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnCungbr_bufferSize(cusolverDnHandle_t handle, cublasSideMode_t side, int m, int n, int k, const cuComplex *A, int lda, const cuComplex *tau, int *lwork) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cublasSideMode_t>(side);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(k);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(tau);
    CusolverFrontend::Execute("cusolverDnCungbr_bufferSize");
    if (CusolverFrontend::Success()) {
        *lwork = CusolverFrontend::GetOutputVariable<int>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnZungbr_bufferSize(cusolverDnHandle_t handle, cublasSideMode_t side, int m, int n, int k, const cuDoubleComplex *A, int lda, const cuDoubleComplex *tau, int *lwork) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cublasSideMode_t>(side);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(k);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(tau);
    CusolverFrontend::Execute("cusolverDnZungbr_bufferSize");
    if (CusolverFrontend::Success()) {
        *lwork = CusolverFrontend::GetOutputVariable<int>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnSorgbr(cusolverDnHandle_t handle, cublasSideMode_t side, int m, int n, int k, float *A, int lda, const float *tau, float *work, int lwork, int *devInfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cublasSideMode_t>(side);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(k);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(tau);
    CusolverFrontend::AddDevicePointerForArguments(work);
    CusolverFrontend::AddVariableForArguments<int>(lwork);
    CusolverFrontend::AddDevicePointerForArguments(devInfo);
    CusolverFrontend::Execute("cusolverDnSorgbr");
    if (CusolverFrontend::Success()) {
        tau = (float*) CusolverFrontend::GetOutputDevicePointer();
        devInfo = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnDorgbr(cusolverDnHandle_t handle, cublasSideMode_t side, int m, int n, int k, double *A, int lda, const double *tau, double *work, int lwork, int *devInfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cublasSideMode_t>(side);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(k);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(tau);
    CusolverFrontend::AddDevicePointerForArguments(work);
    CusolverFrontend::AddVariableForArguments<int>(lwork);
    CusolverFrontend::AddDevicePointerForArguments(devInfo);
    CusolverFrontend::Execute("cusolverDnDorgbr");
    if (CusolverFrontend::Success()) {
        tau = (double*) CusolverFrontend::GetOutputDevicePointer();
        devInfo = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnCungbr(cusolverDnHandle_t handle, cublasSideMode_t side, int m, int n, int k, cuComplex *A, int lda, const cuComplex *tau, cuComplex *work, int lwork, int *devInfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cublasSideMode_t>(side);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(k);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(tau);
    CusolverFrontend::AddDevicePointerForArguments(work);
    CusolverFrontend::AddVariableForArguments<int>(lwork);
    CusolverFrontend::AddDevicePointerForArguments(devInfo);
    CusolverFrontend::Execute("cusolverDnCungbr");
    if (CusolverFrontend::Success()) {
        tau = (cuComplex*) CusolverFrontend::GetOutputDevicePointer();
        devInfo = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnZungbr(cusolverDnHandle_t handle, cublasSideMode_t side, int m, int n, int k, cuDoubleComplex *A, int lda, const cuDoubleComplex *tau, cuDoubleComplex *work, int lwork, int *devInfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cublasSideMode_t>(side);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(k);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(tau);
    CusolverFrontend::AddDevicePointerForArguments(work);
    CusolverFrontend::AddVariableForArguments<int>(lwork);
    CusolverFrontend::AddDevicePointerForArguments(devInfo);
    CusolverFrontend::Execute("cusolverDnZungbr");
    if (CusolverFrontend::Success()) {
        tau = (cuDoubleComplex*) CusolverFrontend::GetOutputDevicePointer();
        devInfo = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnSsytrd_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, const float *A, int lda, const float *d, const float *e, const float *tau, int *lwork) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(d);
    CusolverFrontend::AddDevicePointerForArguments(e);
    CusolverFrontend::AddDevicePointerForArguments(tau);
    CusolverFrontend::Execute("cusolverDnSsytrd_bufferSize");
    if (CusolverFrontend::Success()) {
        *lwork = CusolverFrontend::GetOutputVariable<int>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnDsytrd_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, const double *A, int lda, const double *d, const double *e, const double *tau, int *lwork) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(d);
    CusolverFrontend::AddDevicePointerForArguments(e);
    CusolverFrontend::AddDevicePointerForArguments(tau);
    CusolverFrontend::Execute("cusolverDnDsytrd_bufferSize");
    if (CusolverFrontend::Success()) {
        *lwork = CusolverFrontend::GetOutputVariable<int>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnChetrd_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex *A, int lda, const float *d, const float *e, const cuComplex *tau, int *lwork) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(d);
    CusolverFrontend::AddDevicePointerForArguments(e);
    CusolverFrontend::AddDevicePointerForArguments(tau);
    CusolverFrontend::Execute("cusolverDnChetrd_bufferSize");
    if (CusolverFrontend::Success()) {
        *lwork = CusolverFrontend::GetOutputVariable<int>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnZhetrd_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex *A, int lda, const double *d, const double *e, const cuDoubleComplex *tau, int *lwork) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(d);
    CusolverFrontend::AddDevicePointerForArguments(e);
    CusolverFrontend::AddDevicePointerForArguments(tau);
    CusolverFrontend::Execute("cusolverDnZhetrd_bufferSize");
    if (CusolverFrontend::Success()) {
        *lwork = CusolverFrontend::GetOutputVariable<int>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnSsytrd(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, float *A, int lda, float *d, float *e, float *tau, float *work, int lwork, int *devInfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(d);
    CusolverFrontend::AddDevicePointerForArguments(e);
    CusolverFrontend::AddDevicePointerForArguments(tau);
    CusolverFrontend::AddDevicePointerForArguments(work);
    CusolverFrontend::AddVariableForArguments<int>(lwork);
    CusolverFrontend::AddDevicePointerForArguments(devInfo);
    CusolverFrontend::Execute("cusolverDnSsytrd");
    if (CusolverFrontend::Success()) {
        d = (float*) CusolverFrontend::GetOutputDevicePointer();
        e = (float*) CusolverFrontend::GetOutputDevicePointer();
        tau = (float*) CusolverFrontend::GetOutputDevicePointer();
        devInfo = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnDsytrd(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, double *A, int lda, double *d, double *e, double *tau, double *work, int lwork, int *devInfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(d);
    CusolverFrontend::AddDevicePointerForArguments(e);
    CusolverFrontend::AddDevicePointerForArguments(tau);
    CusolverFrontend::AddDevicePointerForArguments(work);
    CusolverFrontend::AddVariableForArguments<int>(lwork);
    CusolverFrontend::AddDevicePointerForArguments(devInfo);
    CusolverFrontend::Execute("cusolverDnDsytrd");
    if (CusolverFrontend::Success()) {
        d = (double*) CusolverFrontend::GetOutputDevicePointer();
        e = (double*) CusolverFrontend::GetOutputDevicePointer();
        tau = (double*) CusolverFrontend::GetOutputDevicePointer();
        devInfo = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnChetrd(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuComplex *A, int lda, float *d, float *e, cuComplex *tau, cuComplex *work, int lwork, int *devInfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(d);
    CusolverFrontend::AddDevicePointerForArguments(e);
    CusolverFrontend::AddDevicePointerForArguments(tau);
    CusolverFrontend::AddDevicePointerForArguments(work);
    CusolverFrontend::AddVariableForArguments<int>(lwork);
    CusolverFrontend::AddDevicePointerForArguments(devInfo);
    CusolverFrontend::Execute("cusolverDnChetrd");
    if (CusolverFrontend::Success()) {
        d = (float*) CusolverFrontend::GetOutputDevicePointer();
        e = (float*) CusolverFrontend::GetOutputDevicePointer();
        tau = (cuComplex*) CusolverFrontend::GetOutputDevicePointer();
        devInfo = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnZhetrd(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuDoubleComplex *A, int lda, double *d, double *e, cuDoubleComplex *tau, cuDoubleComplex *work, int lwork, int *devInfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(d);
    CusolverFrontend::AddDevicePointerForArguments(e);
    CusolverFrontend::AddDevicePointerForArguments(tau);
    CusolverFrontend::AddDevicePointerForArguments(work);
    CusolverFrontend::AddVariableForArguments<int>(lwork);
    CusolverFrontend::AddDevicePointerForArguments(devInfo);
    CusolverFrontend::Execute("cusolverDnZhetrd");
    if (CusolverFrontend::Success()) {
        d = (double*) CusolverFrontend::GetOutputDevicePointer();
        e = (double*) CusolverFrontend::GetOutputDevicePointer();
        tau = (cuDoubleComplex*) CusolverFrontend::GetOutputDevicePointer();
        devInfo = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnSormtr_bufferSize(cusolverDnHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, int m, int n, const float *A, int lda, const float *tau, const float *C, int ldc, int *lwork) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cublasSideMode_t>(side);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<cublasOperation_t>(trans);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(tau);
    CusolverFrontend::AddDevicePointerForArguments(C);
    CusolverFrontend::AddVariableForArguments<int>(ldc);
    CusolverFrontend::Execute("cusolverDnSormtr_bufferSize");
    if (CusolverFrontend::Success()) {
        *lwork = CusolverFrontend::GetOutputVariable<int>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnDormtr_bufferSize(cusolverDnHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, int m, int n, const double *A, int lda, const double *tau, const double *C, int ldc, int *lwork) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cublasSideMode_t>(side);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<cublasOperation_t>(trans);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(tau);
    CusolverFrontend::AddDevicePointerForArguments(C);
    CusolverFrontend::AddVariableForArguments<int>(ldc);
    CusolverFrontend::Execute("cusolverDnDormtr_bufferSize");
    if (CusolverFrontend::Success()) {
        *lwork = CusolverFrontend::GetOutputVariable<int>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnCunmtr_bufferSize(cusolverDnHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, int m, int n, const cuComplex *A, int lda, const cuComplex *tau, const cuComplex *C, int ldc, int *lwork) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cublasSideMode_t>(side);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<cublasOperation_t>(trans);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(tau);
    CusolverFrontend::AddDevicePointerForArguments(C);
    CusolverFrontend::AddVariableForArguments<int>(ldc);
    CusolverFrontend::Execute("cusolverDnCunmtr_bufferSize");
    if (CusolverFrontend::Success()) {
        *lwork = CusolverFrontend::GetOutputVariable<int>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnZunmtr_bufferSize(cusolverDnHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, int m, int n, const cuDoubleComplex *A, int lda, const cuDoubleComplex *tau, const cuDoubleComplex *C, int ldc, int *lwork) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cublasSideMode_t>(side);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<cublasOperation_t>(trans);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(tau);
    CusolverFrontend::AddDevicePointerForArguments(C);
    CusolverFrontend::AddVariableForArguments<int>(ldc);
    CusolverFrontend::Execute("cusolverDnZunmtr_bufferSize");
    if (CusolverFrontend::Success()) {
        *lwork = CusolverFrontend::GetOutputVariable<int>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnSormtr(cusolverDnHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, int m, int n, float *A, int lda, float *tau, float *C, int ldc, float *work, int lwork, int *devInfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cublasSideMode_t>(side);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<cublasOperation_t>(trans);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(tau);
    CusolverFrontend::AddDevicePointerForArguments(C);
    CusolverFrontend::AddVariableForArguments<int>(ldc);
    CusolverFrontend::AddDevicePointerForArguments(work);
    CusolverFrontend::AddVariableForArguments<int>(lwork);
    CusolverFrontend::AddDevicePointerForArguments(devInfo);
    CusolverFrontend::Execute("cusolverDnSormtr");
    if (CusolverFrontend::Success()) {
        tau = (float*) CusolverFrontend::GetOutputDevicePointer();
        devInfo = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnDormtr(cusolverDnHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, int m, int n, double *A, int lda, double *tau, double *C, int ldc, double *work, int lwork, int *devInfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cublasSideMode_t>(side);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<cublasOperation_t>(trans);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(tau);
    CusolverFrontend::AddDevicePointerForArguments(C);
    CusolverFrontend::AddVariableForArguments<int>(ldc);
    CusolverFrontend::AddDevicePointerForArguments(work);
    CusolverFrontend::AddVariableForArguments<int>(lwork);
    CusolverFrontend::AddDevicePointerForArguments(devInfo);
    CusolverFrontend::Execute("cusolverDnDormtr");
    if (CusolverFrontend::Success()) {
        tau = (double*) CusolverFrontend::GetOutputDevicePointer();
        devInfo = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnCunmtr(cusolverDnHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, int m, int n, cuComplex *A, int lda, cuComplex *tau, cuComplex *C, int ldc, cuComplex *work, int lwork, int *devInfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cublasSideMode_t>(side);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<cublasOperation_t>(trans);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(tau);
    CusolverFrontend::AddDevicePointerForArguments(C);
    CusolverFrontend::AddVariableForArguments<int>(ldc);
    CusolverFrontend::AddDevicePointerForArguments(work);
    CusolverFrontend::AddVariableForArguments<int>(lwork);
    CusolverFrontend::AddDevicePointerForArguments(devInfo);
    CusolverFrontend::Execute("cusolverDnCunmtr");
    if (CusolverFrontend::Success()) {
        tau = (cuComplex*) CusolverFrontend::GetOutputDevicePointer();
        devInfo = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnZunmtr(cusolverDnHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, int m, int n, cuDoubleComplex *A, int lda, cuDoubleComplex *tau, cuDoubleComplex *C, int ldc, cuDoubleComplex *work, int lwork, int *devInfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cublasSideMode_t>(side);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<cublasOperation_t>(trans);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(tau);
    CusolverFrontend::AddDevicePointerForArguments(C);
    CusolverFrontend::AddVariableForArguments<int>(ldc);
    CusolverFrontend::AddDevicePointerForArguments(work);
    CusolverFrontend::AddVariableForArguments<int>(lwork);
    CusolverFrontend::AddDevicePointerForArguments(devInfo);
    CusolverFrontend::Execute("cusolverDnZunmtr");
    if (CusolverFrontend::Success()) {
        tau = (cuDoubleComplex*) CusolverFrontend::GetOutputDevicePointer();
        devInfo = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnSorgtr_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, const float *A, int lda, const float *tau, int *lwork) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(tau);
    CusolverFrontend::Execute("cusolverDnSorgtr_bufferSize");
    if (CusolverFrontend::Success()) {
        *lwork = CusolverFrontend::GetOutputVariable<int>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnDorgtr_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, const double *A, int lda, const double *tau, int *lwork) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(tau);
    CusolverFrontend::Execute("cusolverDnDorgtr_bufferSize");
    if (CusolverFrontend::Success()) {
        *lwork = CusolverFrontend::GetOutputVariable<int>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnCungtr_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex *A, int lda, const cuComplex *tau, int *lwork) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(tau);
    CusolverFrontend::Execute("cusolverDnCungtr_bufferSize");
    if (CusolverFrontend::Success()) {
        *lwork = CusolverFrontend::GetOutputVariable<int>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnZungtr_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex *A, int lda, const cuDoubleComplex *tau, int *lwork) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(tau);
    CusolverFrontend::Execute("cusolverDnZungtr_bufferSize");
    if (CusolverFrontend::Success()) {
        *lwork = CusolverFrontend::GetOutputVariable<int>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnSorgtr(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, float *A, int lda, const float *tau, float *work, int lwork, int *devInfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(tau);
    CusolverFrontend::AddDevicePointerForArguments(work);
    CusolverFrontend::AddVariableForArguments<int>(lwork);
    CusolverFrontend::AddDevicePointerForArguments(devInfo);
    CusolverFrontend::Execute("cusolverDnSorgtr");
    if (CusolverFrontend::Success()) {
        tau = (float*) CusolverFrontend::GetOutputDevicePointer();
        devInfo = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnDorgtr(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, double *A, int lda, const double *tau, double *work, int lwork, int *devInfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(tau);
    CusolverFrontend::AddDevicePointerForArguments(work);
    CusolverFrontend::AddVariableForArguments<int>(lwork);
    CusolverFrontend::AddDevicePointerForArguments(devInfo);
    CusolverFrontend::Execute("cusolverDnDorgtr");
    if (CusolverFrontend::Success()) {
        tau = (double*) CusolverFrontend::GetOutputDevicePointer();
        devInfo = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnCungtr(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuComplex *A, int lda, const cuComplex *tau, cuComplex *work, int lwork, int *devInfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(tau);
    CusolverFrontend::AddDevicePointerForArguments(work);
    CusolverFrontend::AddVariableForArguments<int>(lwork);
    CusolverFrontend::AddDevicePointerForArguments(devInfo);
    CusolverFrontend::Execute("cusolverDnCungtr");
    if (CusolverFrontend::Success()) {
        tau = (cuComplex*) CusolverFrontend::GetOutputDevicePointer();
        devInfo = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnZungtr(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuDoubleComplex *A, int lda, const cuDoubleComplex *tau, cuDoubleComplex *work, int lwork, int *devInfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(tau);
    CusolverFrontend::AddDevicePointerForArguments(work);
    CusolverFrontend::AddVariableForArguments<int>(lwork);
    CusolverFrontend::AddDevicePointerForArguments(devInfo);
    CusolverFrontend::Execute("cusolverDnZungtr");
    if (CusolverFrontend::Success()) {
        tau = (cuDoubleComplex*) CusolverFrontend::GetOutputDevicePointer();
        devInfo = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnSgesvd_bufferSize(cusolverDnHandle_t handle, int m, int n, int *lwork) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::Execute("cusolverDnSgesvd_bufferSize");
    if (CusolverFrontend::Success()) {
        *lwork = CusolverFrontend::GetOutputVariable<int>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnDgesvd_bufferSize(cusolverDnHandle_t handle, int m, int n, int *lwork) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::Execute("cusolverDnDgesvd_bufferSize");
    if (CusolverFrontend::Success()) {
        *lwork = CusolverFrontend::GetOutputVariable<int>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnCgesvd_bufferSize(cusolverDnHandle_t handle, int m, int n, int *lwork) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::Execute("cusolverDnCgesvd_bufferSize");
    if (CusolverFrontend::Success()) {
        *lwork = CusolverFrontend::GetOutputVariable<int>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnZgesvd_bufferSize(cusolverDnHandle_t handle, int m, int n, int *lwork) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::Execute("cusolverDnZgesvd_bufferSize");
    if (CusolverFrontend::Success()) {
        *lwork = CusolverFrontend::GetOutputVariable<int>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnSgesvd(cusolverDnHandle_t handle, signed char jobu, signed char jobvt, int m, int n, float *A, int lda, float *S, float *U, int ldu, float *VT, int ldvt, float *work, int lwork, float *rwork, int *devInfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<signed char>(jobu);
    CusolverFrontend::AddVariableForArguments<signed char>(jobvt);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(S);
    CusolverFrontend::AddDevicePointerForArguments(U);
    CusolverFrontend::AddVariableForArguments<int>(ldu);
    CusolverFrontend::AddDevicePointerForArguments(VT);
    CusolverFrontend::AddVariableForArguments<int>(ldvt);
    CusolverFrontend::AddDevicePointerForArguments(work);
    CusolverFrontend::AddVariableForArguments<int>(lwork);
    CusolverFrontend::AddDevicePointerForArguments(rwork);
    CusolverFrontend::AddDevicePointerForArguments(devInfo);
    CusolverFrontend::Execute("cusolverDnSgesvd");
    if (CusolverFrontend::Success()) {
        S = (float*) CusolverFrontend::GetOutputDevicePointer();
        U = (float*) CusolverFrontend::GetOutputDevicePointer();
        VT = (float*) CusolverFrontend::GetOutputDevicePointer();
        devInfo = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnDgesvd(cusolverDnHandle_t handle, signed char jobu, signed char jobvt, int m, int n, double *A, int lda, double *S, double *U, int ldu, double *VT, int ldvt, double *work, int lwork, double *rwork, int *devInfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<signed char>(jobu);
    CusolverFrontend::AddVariableForArguments<signed char>(jobvt);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(S);
    CusolverFrontend::AddDevicePointerForArguments(U);
    CusolverFrontend::AddVariableForArguments<int>(ldu);
    CusolverFrontend::AddDevicePointerForArguments(VT);
    CusolverFrontend::AddVariableForArguments<int>(ldvt);
    CusolverFrontend::AddDevicePointerForArguments(work);
    CusolverFrontend::AddVariableForArguments<int>(lwork);
    CusolverFrontend::AddDevicePointerForArguments(rwork);
    CusolverFrontend::AddDevicePointerForArguments(devInfo);
    CusolverFrontend::Execute("cusolverDnDgesvd");
    if (CusolverFrontend::Success()) {
        S = (double*) CusolverFrontend::GetOutputDevicePointer();
        U = (double*) CusolverFrontend::GetOutputDevicePointer();
        VT = (double*) CusolverFrontend::GetOutputDevicePointer();
        devInfo = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnCgesvd(cusolverDnHandle_t handle, signed char jobu, signed char jobvt, int m, int n, cuComplex *A, int lda, float *S, cuComplex *U, int ldu, cuComplex *VT, int ldvt, cuComplex *work, int lwork, float *rwork, int *devInfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<signed char>(jobu);
    CusolverFrontend::AddVariableForArguments<signed char>(jobvt);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(S);
    CusolverFrontend::AddDevicePointerForArguments(U);
    CusolverFrontend::AddVariableForArguments<int>(ldu);
    CusolverFrontend::AddDevicePointerForArguments(VT);
    CusolverFrontend::AddVariableForArguments<int>(ldvt);
    CusolverFrontend::AddDevicePointerForArguments(work);
    CusolverFrontend::AddVariableForArguments<int>(lwork);
    CusolverFrontend::AddDevicePointerForArguments(rwork);
    CusolverFrontend::AddDevicePointerForArguments(devInfo);
    CusolverFrontend::Execute("cusolverDnCgesvd");
    if (CusolverFrontend::Success()) {
        S = (float*) CusolverFrontend::GetOutputDevicePointer();
        U = (cuComplex*) CusolverFrontend::GetOutputDevicePointer();
        VT = (cuComplex*) CusolverFrontend::GetOutputDevicePointer();
        devInfo = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnZgesvd(cusolverDnHandle_t handle, signed char jobu, signed char jobvt, int m, int n, cuDoubleComplex *A, int lda, double *S, cuDoubleComplex *U, int ldu, cuDoubleComplex *VT, int ldvt, cuDoubleComplex *work, int lwork, double *rwork, int *devInfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<signed char>(jobu);
    CusolverFrontend::AddVariableForArguments<signed char>(jobvt);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(S);
    CusolverFrontend::AddDevicePointerForArguments(U);
    CusolverFrontend::AddVariableForArguments<int>(ldu);
    CusolverFrontend::AddDevicePointerForArguments(VT);
    CusolverFrontend::AddVariableForArguments<int>(ldvt);
    CusolverFrontend::AddDevicePointerForArguments(work);
    CusolverFrontend::AddVariableForArguments<int>(lwork);
    CusolverFrontend::AddDevicePointerForArguments(rwork);
    CusolverFrontend::AddDevicePointerForArguments(devInfo);
    CusolverFrontend::Execute("cusolverDnZgesvd");
    if (CusolverFrontend::Success()) {
        S = (double*) CusolverFrontend::GetOutputDevicePointer();
        U = (cuDoubleComplex*) CusolverFrontend::GetOutputDevicePointer();
        VT = (cuDoubleComplex*) CusolverFrontend::GetOutputDevicePointer();
        devInfo = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnGesvd_bufferSize(cusolverDnHandle_t handle, cusolverDnParams_t params, signed char jobu, signed char jobvt, int64_t m, int64_t n, cudaDataType dataTypeA, const void *A, int64_t lda, cudaDataType dataTypeS, const void *S, cudaDataType dataTypeU, const void *U, int64_t ldu, cudaDataType dataTypeVT, const void *VT, int64_t ldvt, cudaDataType computeType, size_t *workspaceInBytes) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) params);
    CusolverFrontend::AddVariableForArguments<signed char>(jobu);
    CusolverFrontend::AddVariableForArguments<signed char>(jobvt);
    CusolverFrontend::AddVariableForArguments<int64_t>(m);
    CusolverFrontend::AddVariableForArguments<int64_t>(n);
    CusolverFrontend::AddVariableForArguments<cudaDataType_t>(dataTypeA);
    CusolverFrontend::AddVariableForArguments<int64_t>(lda);
    CusolverFrontend::AddVariableForArguments<cudaDataType_t>(dataTypeS);
    CusolverFrontend::AddVariableForArguments<cudaDataType_t>(dataTypeU);
    CusolverFrontend::AddVariableForArguments<int64_t>(ldu);
    CusolverFrontend::AddVariableForArguments<cudaDataType_t>(dataTypeVT);
    CusolverFrontend::AddVariableForArguments<int64_t>(ldvt);
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
    switch(dataTypeS){
        case CUDA_R_32F:
            //float
            CusolverFrontend::AddDevicePointerForArguments((float *)S);
            break;
        case CUDA_R_64F:
            //double
            CusolverFrontend::AddDevicePointerForArguments((double *)S);
            break;
        case CUDA_C_32F:
            //cuComplex
            CusolverFrontend::AddDevicePointerForArguments((float *)S);
            break;
        case CUDA_C_64F:
            //cuDoubleComplex
            CusolverFrontend::AddDevicePointerForArguments((double *)S);
            break;
        default:
            throw "Type not supported by GVirtus!";
    }
    switch(dataTypeU){
        case CUDA_R_32F:
            //float
            CusolverFrontend::AddDevicePointerForArguments((float *)U);
            break;
        case CUDA_R_64F:
            //double
            CusolverFrontend::AddDevicePointerForArguments((double *)U);
            break;
        case CUDA_C_32F:
            //cuComplex
            CusolverFrontend::AddDevicePointerForArguments((cuComplex *)U);
            break;
        case CUDA_C_64F:
            //cuDoubleComplex
            CusolverFrontend::AddDevicePointerForArguments((cuDoubleComplex *)U);
            break;
        default:
            throw "Type not supported by GVirtus!";
    }
    switch(dataTypeA){
        case CUDA_R_32F:
            //float
            CusolverFrontend::AddDevicePointerForArguments((float *)VT);
            break;
        case CUDA_R_64F:
            //double
            CusolverFrontend::AddDevicePointerForArguments((double *)VT);
            break;
        case CUDA_C_32F:
            //cuComplex
            CusolverFrontend::AddDevicePointerForArguments((cuComplex *)VT);
            break;
        case CUDA_C_64F:
            //cuDoubleComplex
            CusolverFrontend::AddDevicePointerForArguments((cuDoubleComplex *)VT);
            break;
        default:
            throw "Type not supported by GVirtus!";
    }
    CusolverFrontend::Execute("cusolverDnGesvd_bufferSize");
    if (CusolverFrontend::Success()) {
        *workspaceInBytes = CusolverFrontend::GetOutputVariable<size_t>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnGesvd(cusolverDnHandle_t handle, cusolverDnParams_t params, signed char jobu, signed char jobvt, int64_t m, int64_t n, cudaDataType dataTypeA, void *A, int64_t lda, cudaDataType dataTypeS, void *S, cudaDataType dataTypeU, void *U, int64_t ldu, cudaDataType dataTypeVT, void *VT, int64_t ldvt, cudaDataType computeType, void *pBuffer, size_t workspaceInBytes, int *info) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) params);
    CusolverFrontend::AddVariableForArguments<signed char>(jobu);
    CusolverFrontend::AddVariableForArguments<signed char>(jobvt);
    CusolverFrontend::AddVariableForArguments<int64_t>(m);
    CusolverFrontend::AddVariableForArguments<int64_t>(n);
    CusolverFrontend::AddVariableForArguments<cudaDataType_t>(dataTypeA);
    CusolverFrontend::AddVariableForArguments<int64_t>(lda);
    CusolverFrontend::AddVariableForArguments<cudaDataType_t>(dataTypeS);
    CusolverFrontend::AddVariableForArguments<cudaDataType_t>(dataTypeU);
    CusolverFrontend::AddVariableForArguments<int64_t>(ldu);
    CusolverFrontend::AddVariableForArguments<cudaDataType_t>(dataTypeVT);
    CusolverFrontend::AddVariableForArguments<int64_t>(ldvt);
    CusolverFrontend::AddVariableForArguments<cudaDataType_t>(computeType);
    CusolverFrontend::AddDevicePointerForArguments(pBuffer);
    CusolverFrontend::AddVariableForArguments<size_t>(workspaceInBytes);
    CusolverFrontend::AddDevicePointerForArguments(info);
    switch(dataTypeA){
        case CUDA_R_32F:
            //float
            CusolverFrontend::AddDevicePointerForArguments((float *)A);
            CusolverFrontend::AddDevicePointerForArguments((float *)S);
            CusolverFrontend::AddDevicePointerForArguments((float *)U);
            CusolverFrontend::AddDevicePointerForArguments((float *)VT);
            break;
        case CUDA_R_64F:
            //double
            CusolverFrontend::AddDevicePointerForArguments((double *)A);
            CusolverFrontend::AddDevicePointerForArguments((double *)S);
            CusolverFrontend::AddDevicePointerForArguments((double *)U);
            CusolverFrontend::AddDevicePointerForArguments((double *)VT);
            break;
        case CUDA_C_32F:
            //cuComplex
            CusolverFrontend::AddDevicePointerForArguments((cuComplex *)A);
            CusolverFrontend::AddDevicePointerForArguments((float *)S);
            CusolverFrontend::AddDevicePointerForArguments((cuComplex *)U);
            CusolverFrontend::AddDevicePointerForArguments((cuComplex *)VT);
            break;
        case CUDA_C_64F:
            //cuDoubleComplex
            CusolverFrontend::AddDevicePointerForArguments((cuDoubleComplex *)A);
            CusolverFrontend::AddDevicePointerForArguments((double *)S);
            CusolverFrontend::AddDevicePointerForArguments((cuDoubleComplex *)U);
            CusolverFrontend::AddDevicePointerForArguments((cuDoubleComplex *)VT);
            break;
        default:
            throw "Type not supported by GVirtus!";
    }
    CusolverFrontend::Execute("cusolverDnGesvd");
    if (CusolverFrontend::Success()) {
        S = CusolverFrontend::GetOutputDevicePointer();
        U = CusolverFrontend::GetOutputDevicePointer();
        VT = CusolverFrontend::GetOutputDevicePointer();
        info = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnSgesvdj_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ, int m, int n, const float *A, int lda, const float *S, const float *U, int ldu, const float *V, int ldv, int *lwork, gesvdjInfo_t params) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cusolverEigMode_t>(jobz);
    CusolverFrontend::AddVariableForArguments<int>(econ);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(S);
    CusolverFrontend::AddDevicePointerForArguments(U);
    CusolverFrontend::AddVariableForArguments<int>(ldu);
    CusolverFrontend::AddDevicePointerForArguments(V);
    CusolverFrontend::AddVariableForArguments<int>(ldv);
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) params);
    CusolverFrontend::Execute("cusolverDnSgesvdj_bufferSize");
    if (CusolverFrontend::Success()) {
        *lwork = CusolverFrontend::GetOutputVariable<int>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnDgesvdj_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ, int m, int n, const double *A, int lda, const double *S, const double *U, int ldu, const double *V, int ldv, int *lwork, gesvdjInfo_t params) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cusolverEigMode_t>(jobz);
    CusolverFrontend::AddVariableForArguments<int>(econ);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(S);
    CusolverFrontend::AddDevicePointerForArguments(U);
    CusolverFrontend::AddVariableForArguments<int>(ldu);
    CusolverFrontend::AddDevicePointerForArguments(V);
    CusolverFrontend::AddVariableForArguments<int>(ldv);
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) params);
    CusolverFrontend::Execute("cusolverDnDgesvdj_bufferSize");
    if (CusolverFrontend::Success()) {
        *lwork = CusolverFrontend::GetOutputVariable<int>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnCgesvdj_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ, int m, int n, const cuComplex *A, int lda, const float *S, const cuComplex *U, int ldu, const cuComplex *V, int ldv, int *lwork, gesvdjInfo_t params) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cusolverEigMode_t>(jobz);
    CusolverFrontend::AddVariableForArguments<int>(econ);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(S);
    CusolverFrontend::AddDevicePointerForArguments(U);
    CusolverFrontend::AddVariableForArguments<int>(ldu);
    CusolverFrontend::AddDevicePointerForArguments(V);
    CusolverFrontend::AddVariableForArguments<int>(ldv);
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) params);
    CusolverFrontend::Execute("cusolverDnCgesvdj_bufferSize");
    if (CusolverFrontend::Success()) {
        *lwork = CusolverFrontend::GetOutputVariable<int>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnZgesvdj_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ, int m, int n, const cuDoubleComplex *A, int lda, const double *S, const cuDoubleComplex *U, int ldu, const cuDoubleComplex *V, int ldv, int *lwork, gesvdjInfo_t params) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cusolverEigMode_t>(jobz);
    CusolverFrontend::AddVariableForArguments<int>(econ);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(S);
    CusolverFrontend::AddDevicePointerForArguments(U);
    CusolverFrontend::AddVariableForArguments<int>(ldu);
    CusolverFrontend::AddDevicePointerForArguments(V);
    CusolverFrontend::AddVariableForArguments<int>(ldv);
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) params);
    CusolverFrontend::Execute("cusolverDnZgesvdj_bufferSize");
    if (CusolverFrontend::Success()) {
        *lwork = CusolverFrontend::GetOutputVariable<int>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnSgesvdj(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ, int m, int n, float *A, int lda, float *S, float *U, int ldu, float *V, int ldv, float *work, int lwork, int *info, gesvdjInfo_t params) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cusolverEigMode_t>(jobz);
    CusolverFrontend::AddVariableForArguments<int>(econ);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(S);
    CusolverFrontend::AddDevicePointerForArguments(U);
    CusolverFrontend::AddVariableForArguments<int>(ldu);
    CusolverFrontend::AddDevicePointerForArguments(V);
    CusolverFrontend::AddVariableForArguments<int>(ldv);
    CusolverFrontend::AddDevicePointerForArguments(work);
    CusolverFrontend::AddVariableForArguments<int>(lwork);
    CusolverFrontend::AddDevicePointerForArguments(info);
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) params);
    CusolverFrontend::Execute("cusolverDnSgesvdj");
    if (CusolverFrontend::Success()) {
        S = (float*) CusolverFrontend::GetOutputDevicePointer();
        U = (float*) CusolverFrontend::GetOutputDevicePointer();
        V = (float*) CusolverFrontend::GetOutputDevicePointer();
        info = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnDgesvdj(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ, int m, int n, double *A, int lda, double *S, double *U, int ldu, double *V, int ldv, double *work, int lwork, int *info, gesvdjInfo_t params) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cusolverEigMode_t>(jobz);
    CusolverFrontend::AddVariableForArguments<int>(econ);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(S);
    CusolverFrontend::AddDevicePointerForArguments(U);
    CusolverFrontend::AddVariableForArguments<int>(ldu);
    CusolverFrontend::AddDevicePointerForArguments(V);
    CusolverFrontend::AddVariableForArguments<int>(ldv);
    CusolverFrontend::AddDevicePointerForArguments(work);
    CusolverFrontend::AddVariableForArguments<int>(lwork);
    CusolverFrontend::AddDevicePointerForArguments(info);
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) params);
    CusolverFrontend::Execute("cusolverDnDgesvdj");
    if (CusolverFrontend::Success()) {
        S = (double*) CusolverFrontend::GetOutputDevicePointer();
        U = (double*) CusolverFrontend::GetOutputDevicePointer();
        V = (double*) CusolverFrontend::GetOutputDevicePointer();
        info = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnCgesvdj(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ, int m, int n, cuComplex *A, int lda, float *S, cuComplex *U, int ldu, cuComplex *V, int ldv, cuComplex *work, int lwork, int *info, gesvdjInfo_t params) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cusolverEigMode_t>(jobz);
    CusolverFrontend::AddVariableForArguments<int>(econ);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(S);
    CusolverFrontend::AddDevicePointerForArguments(U);
    CusolverFrontend::AddVariableForArguments<int>(ldu);
    CusolverFrontend::AddDevicePointerForArguments(V);
    CusolverFrontend::AddVariableForArguments<int>(ldv);
    CusolverFrontend::AddDevicePointerForArguments(work);
    CusolverFrontend::AddVariableForArguments<int>(lwork);
    CusolverFrontend::AddDevicePointerForArguments(info);
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) params);
    CusolverFrontend::Execute("cusolverDnCgesvdj");
    if (CusolverFrontend::Success()) {
        S = (float*) CusolverFrontend::GetOutputDevicePointer();
        U = (cuComplex*) CusolverFrontend::GetOutputDevicePointer();
        V = (cuComplex*) CusolverFrontend::GetOutputDevicePointer();
        info = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnZgesvdj(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ, int m, int n, cuDoubleComplex *A, int lda, double *S, cuDoubleComplex *U, int ldu, cuDoubleComplex *V, int ldv, cuDoubleComplex *work, int lwork, int *info, gesvdjInfo_t params) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cusolverEigMode_t>(jobz);
    CusolverFrontend::AddVariableForArguments<int>(econ);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(S);
    CusolverFrontend::AddDevicePointerForArguments(U);
    CusolverFrontend::AddVariableForArguments<int>(ldu);
    CusolverFrontend::AddDevicePointerForArguments(V);
    CusolverFrontend::AddVariableForArguments<int>(ldv);
    CusolverFrontend::AddDevicePointerForArguments(work);
    CusolverFrontend::AddVariableForArguments<int>(lwork);
    CusolverFrontend::AddDevicePointerForArguments(info);
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) params);
    CusolverFrontend::Execute("cusolverDnZgesvdj");
    if (CusolverFrontend::Success()) {
        S = (double*) CusolverFrontend::GetOutputDevicePointer();
        U = (cuDoubleComplex*) CusolverFrontend::GetOutputDevicePointer();
        V = (cuDoubleComplex*) CusolverFrontend::GetOutputDevicePointer();
        info = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnSgesvdjBatched_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int m, int n, const float *A, int lda, const float *S, const float *U, int ldu, const float *V, int ldv, int *lwork, gesvdjInfo_t params, int batchSize) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cusolverEigMode_t>(jobz);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(S);
    CusolverFrontend::AddDevicePointerForArguments(U);
    CusolverFrontend::AddVariableForArguments<int>(ldu);
    CusolverFrontend::AddDevicePointerForArguments(V);
    CusolverFrontend::AddVariableForArguments<int>(ldv);
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) params);
    CusolverFrontend::AddVariableForArguments<int>(batchSize);
    CusolverFrontend::Execute("cusolverDnSgesvdjBatched_bufferSize");
    if (CusolverFrontend::Success()) {
        *lwork = CusolverFrontend::GetOutputVariable<int>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnDgesvdjBatched_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int m, int n, const double *A, int lda, const double *S, const double *U, int ldu, const double *V, int ldv, int *lwork, gesvdjInfo_t params, int batchSize) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cusolverEigMode_t>(jobz);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(S);
    CusolverFrontend::AddDevicePointerForArguments(U);
    CusolverFrontend::AddVariableForArguments<int>(ldu);
    CusolverFrontend::AddDevicePointerForArguments(V);
    CusolverFrontend::AddVariableForArguments<int>(ldv);
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) params);
    CusolverFrontend::AddVariableForArguments<int>(batchSize);
    CusolverFrontend::Execute("cusolverDnDgesvdjBatched_bufferSize");
    if (CusolverFrontend::Success()) {
        *lwork = CusolverFrontend::GetOutputVariable<int>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnCgesvdjBatched_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int m, int n, const cuComplex *A, int lda, const float *S, const cuComplex *U, int ldu, const cuComplex *V, int ldv, int *lwork, gesvdjInfo_t params, int batchSize) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cusolverEigMode_t>(jobz);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(S);
    CusolverFrontend::AddDevicePointerForArguments(U);
    CusolverFrontend::AddVariableForArguments<int>(ldu);
    CusolverFrontend::AddDevicePointerForArguments(V);
    CusolverFrontend::AddVariableForArguments<int>(ldv);
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) params);
    CusolverFrontend::AddVariableForArguments<int>(batchSize);
    CusolverFrontend::Execute("cusolverDnCgesvdjBatched_bufferSize");
    if (CusolverFrontend::Success()) {
        *lwork = CusolverFrontend::GetOutputVariable<int>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnZgesvdjBatched_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int m, int n, const cuDoubleComplex *A, int lda, const double *S, const cuDoubleComplex *U, int ldu, const cuDoubleComplex *V, int ldv, int *lwork, gesvdjInfo_t params, int batchSize) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cusolverEigMode_t>(jobz);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(S);
    CusolverFrontend::AddDevicePointerForArguments(U);
    CusolverFrontend::AddVariableForArguments<int>(ldu);
    CusolverFrontend::AddDevicePointerForArguments(V);
    CusolverFrontend::AddVariableForArguments<int>(ldv);
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) params);
    CusolverFrontend::AddVariableForArguments<int>(batchSize);
    CusolverFrontend::Execute("cusolverDnZgesvdjBatched_bufferSize");
    if (CusolverFrontend::Success()) {
        *lwork = CusolverFrontend::GetOutputVariable<int>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnSgesvdjBatched(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int m, int n, float *A, int lda, float *S, float *U, int ldu, float *V, int ldv, float *work, int lwork, int *info, gesvdjInfo_t params, int batchSize) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cusolverEigMode_t>(jobz);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(S);
    CusolverFrontend::AddDevicePointerForArguments(U);
    CusolverFrontend::AddVariableForArguments<int>(ldu);
    CusolverFrontend::AddDevicePointerForArguments(V);
    CusolverFrontend::AddVariableForArguments<int>(ldv);
    CusolverFrontend::AddDevicePointerForArguments(work);
    CusolverFrontend::AddVariableForArguments<int>(lwork);
    CusolverFrontend::AddDevicePointerForArguments(info);
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) params);
    CusolverFrontend::AddVariableForArguments<int>(batchSize);
    CusolverFrontend::Execute("cusolverDnSgesvdjBatched");
    if (CusolverFrontend::Success()) {
        S = (float*) CusolverFrontend::GetOutputDevicePointer();
        U = (float*) CusolverFrontend::GetOutputDevicePointer();
        V = (float*) CusolverFrontend::GetOutputDevicePointer();
        info = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnDgesvdjBatched(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int m, int n, double *A, int lda, double *S, double *U, int ldu, double *V, int ldv, double *work, int lwork, int *info, gesvdjInfo_t params, int batchSize) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cusolverEigMode_t>(jobz);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(S);
    CusolverFrontend::AddDevicePointerForArguments(U);
    CusolverFrontend::AddVariableForArguments<int>(ldu);
    CusolverFrontend::AddDevicePointerForArguments(V);
    CusolverFrontend::AddVariableForArguments<int>(ldv);
    CusolverFrontend::AddDevicePointerForArguments(work);
    CusolverFrontend::AddVariableForArguments<int>(lwork);
    CusolverFrontend::AddDevicePointerForArguments(info);
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) params);
    CusolverFrontend::AddVariableForArguments<int>(batchSize);
    CusolverFrontend::Execute("cusolverDnDgesvdjBatched");
    if (CusolverFrontend::Success()) {
        S = (double*) CusolverFrontend::GetOutputDevicePointer();
        U = (double*) CusolverFrontend::GetOutputDevicePointer();
        V = (double*) CusolverFrontend::GetOutputDevicePointer();
        info = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnCgesvdjBatched(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int m, int n, cuComplex *A, int lda, float *S, cuComplex *U, int ldu, cuComplex *V, int ldv, cuComplex *work, int lwork, int *info, gesvdjInfo_t params, int batchSize) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cusolverEigMode_t>(jobz);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(S);
    CusolverFrontend::AddDevicePointerForArguments(U);
    CusolverFrontend::AddVariableForArguments<int>(ldu);
    CusolverFrontend::AddDevicePointerForArguments(V);
    CusolverFrontend::AddVariableForArguments<int>(ldv);
    CusolverFrontend::AddDevicePointerForArguments(work);
    CusolverFrontend::AddVariableForArguments<int>(lwork);
    CusolverFrontend::AddDevicePointerForArguments(info);
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) params);
    CusolverFrontend::AddVariableForArguments<int>(batchSize);
    CusolverFrontend::Execute("cusolverDnCgesvdjBatched");
    if (CusolverFrontend::Success()) {
        S = (float*) CusolverFrontend::GetOutputDevicePointer();
        U = (cuComplex*) CusolverFrontend::GetOutputDevicePointer();
        V = (cuComplex*) CusolverFrontend::GetOutputDevicePointer();
        info = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnZgesvdjBatched(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int m, int n, cuDoubleComplex *A, int lda, double *S, cuDoubleComplex *U, int ldu, cuDoubleComplex *V, int ldv, cuDoubleComplex *work, int lwork, int *info, gesvdjInfo_t params, int batchSize) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cusolverEigMode_t>(jobz);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(S);
    CusolverFrontend::AddDevicePointerForArguments(U);
    CusolverFrontend::AddVariableForArguments<int>(ldu);
    CusolverFrontend::AddDevicePointerForArguments(V);
    CusolverFrontend::AddVariableForArguments<int>(ldv);
    CusolverFrontend::AddDevicePointerForArguments(work);
    CusolverFrontend::AddVariableForArguments<int>(lwork);
    CusolverFrontend::AddDevicePointerForArguments(info);
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) params);
    CusolverFrontend::AddVariableForArguments<int>(batchSize);
    CusolverFrontend::Execute("cusolverDnZgesvdjBatched");
    if (CusolverFrontend::Success()) {
        S = (double*) CusolverFrontend::GetOutputDevicePointer();
        U = (cuDoubleComplex*) CusolverFrontend::GetOutputDevicePointer();
        V = (cuDoubleComplex*) CusolverFrontend::GetOutputDevicePointer();
        info = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnSgesvdaStridedBatched_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int rank, int m, int n, const float *A, int lda, long long int strideA, const float *S, long long int strideS, const float *U, int ldu, long long int strideU, const float *V, int ldv, long long int strideV, int *lwork, int batchSize) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cusolverEigMode_t>(jobz);
    CusolverFrontend::AddVariableForArguments<int>(rank);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddVariableForArguments<size_t>(strideA);
    CusolverFrontend::AddDevicePointerForArguments(S);
    CusolverFrontend::AddVariableForArguments<size_t>(strideS);
    CusolverFrontend::AddDevicePointerForArguments(U);
    CusolverFrontend::AddVariableForArguments<int>(ldu);
    CusolverFrontend::AddVariableForArguments<size_t>(strideU);
    CusolverFrontend::AddDevicePointerForArguments(V);
    CusolverFrontend::AddVariableForArguments<int>(ldv);
    CusolverFrontend::AddVariableForArguments<size_t>(strideV);
    CusolverFrontend::AddVariableForArguments<int>(batchSize);
    CusolverFrontend::Execute("cusolverDnSgesvdaStridedBatched_bufferSize");
    if (CusolverFrontend::Success()) {
        *lwork = CusolverFrontend::GetOutputVariable<int>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnDgesvdaStridedBatched_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int rank, int m, int n, const double *A, int lda, long long int strideA, const double *S, long long int strideS, const double *U, int ldu, long long int strideU, const double *V, int ldv, long long int strideV, int *lwork, int batchSize) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cusolverEigMode_t>(jobz);
    CusolverFrontend::AddVariableForArguments<int>(rank);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddVariableForArguments<size_t>(strideA);
    CusolverFrontend::AddDevicePointerForArguments(S);
    CusolverFrontend::AddVariableForArguments<size_t>(strideS);
    CusolverFrontend::AddDevicePointerForArguments(U);
    CusolverFrontend::AddVariableForArguments<int>(ldu);
    CusolverFrontend::AddVariableForArguments<size_t>(strideU);
    CusolverFrontend::AddDevicePointerForArguments(V);
    CusolverFrontend::AddVariableForArguments<int>(ldv);
    CusolverFrontend::AddVariableForArguments<size_t>(strideV);
    CusolverFrontend::AddVariableForArguments<int>(batchSize);
    CusolverFrontend::Execute("cusolverDnDgesvdaStridedBatched_bufferSize");
    if (CusolverFrontend::Success()) {
        *lwork = CusolverFrontend::GetOutputVariable<int>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnCgesvdaStridedBatched_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int rank, int m, int n, const cuComplex *A, int lda, long long int strideA, const float *S, long long int strideS, const cuComplex *U, int ldu, long long int strideU, const cuComplex *V, int ldv, long long int strideV, int *lwork, int batchSize) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cusolverEigMode_t>(jobz);
    CusolverFrontend::AddVariableForArguments<int>(rank);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddVariableForArguments<size_t>(strideA);
    CusolverFrontend::AddDevicePointerForArguments(S);
    CusolverFrontend::AddVariableForArguments<size_t>(strideS);
    CusolverFrontend::AddDevicePointerForArguments(U);
    CusolverFrontend::AddVariableForArguments<int>(ldu);
    CusolverFrontend::AddVariableForArguments<size_t>(strideU);
    CusolverFrontend::AddDevicePointerForArguments(V);
    CusolverFrontend::AddVariableForArguments<int>(ldv);
    CusolverFrontend::AddVariableForArguments<size_t>(strideV);
    CusolverFrontend::AddVariableForArguments<int>(batchSize);
    CusolverFrontend::Execute("cusolverDnCgesvdaStridedBatched_bufferSize");
    if (CusolverFrontend::Success()) {
        *lwork = CusolverFrontend::GetOutputVariable<int>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnZgesvdaStridedBatched_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int rank, int m, int n, const cuDoubleComplex *A, int lda, long long int strideA, const double *S, long long int strideS, const cuDoubleComplex *U, int ldu, long long int strideU, const cuDoubleComplex *V, int ldv, long long int strideV, int *lwork, int batchSize) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cusolverEigMode_t>(jobz);
    CusolverFrontend::AddVariableForArguments<int>(rank);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddVariableForArguments<size_t>(strideA);
    CusolverFrontend::AddDevicePointerForArguments(S);
    CusolverFrontend::AddVariableForArguments<size_t>(strideS);
    CusolverFrontend::AddDevicePointerForArguments(U);
    CusolverFrontend::AddVariableForArguments<int>(ldu);
    CusolverFrontend::AddVariableForArguments<size_t>(strideU);
    CusolverFrontend::AddDevicePointerForArguments(V);
    CusolverFrontend::AddVariableForArguments<int>(ldv);
    CusolverFrontend::AddVariableForArguments<size_t>(strideV);
    CusolverFrontend::AddVariableForArguments<int>(batchSize);
    CusolverFrontend::Execute("cusolverDnZgesvdaStridedBatched_bufferSize");
    if (CusolverFrontend::Success()) {
        *lwork = CusolverFrontend::GetOutputVariable<int>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnSgesvdaStridedBatched(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int rank, int m, int n, const float *A, int lda, long long int strideA, float *S, long long int strideS, float *U, int ldu, long long int strideU, float *V, int ldv, long long int strideV, float *work, int lwork, int *info, double *h_R_nrmF, int batchSize) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cusolverEigMode_t>(jobz);
    CusolverFrontend::AddVariableForArguments<int>(rank);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddVariableForArguments<size_t>(strideA);
    CusolverFrontend::AddDevicePointerForArguments(S);
    CusolverFrontend::AddVariableForArguments<size_t>(strideS);
    CusolverFrontend::AddDevicePointerForArguments(U);
    CusolverFrontend::AddVariableForArguments<int>(ldu);
    CusolverFrontend::AddVariableForArguments<size_t>(strideU);
    CusolverFrontend::AddDevicePointerForArguments(V);
    CusolverFrontend::AddVariableForArguments<int>(ldv);
    CusolverFrontend::AddVariableForArguments<size_t>(strideV);
    CusolverFrontend::AddDevicePointerForArguments(work);
    CusolverFrontend::AddVariableForArguments<int>(lwork);
    CusolverFrontend::AddDevicePointerForArguments(info);
    CusolverFrontend::AddVariableForArguments<int>(batchSize);
    CusolverFrontend::AddHostPointerForArguments(h_R_nrmF);
    CusolverFrontend::Execute("cusolverDnSgesvdaStridedBatched");
    if (CusolverFrontend::Success()) {
        S = (float*) CusolverFrontend::GetOutputDevicePointer();
        U = (float*) CusolverFrontend::GetOutputDevicePointer();
        V = (float*) CusolverFrontend::GetOutputDevicePointer();
        info = (int*) CusolverFrontend::GetOutputDevicePointer();
        h_R_nrmF = CusolverFrontend::GetOutputHostPointer<double>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnDgesvdaStridedBatched(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int rank, int m, int n, const double *A, int lda, long long int strideA, double *S, long long int strideS, double *U, int ldu, long long int strideU, double *V, int ldv, long long int strideV, double *work, int lwork, int *info, double *h_R_nrmF, int batchSize) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cusolverEigMode_t>(jobz);
    CusolverFrontend::AddVariableForArguments<int>(rank);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddVariableForArguments<size_t>(strideA);
    CusolverFrontend::AddDevicePointerForArguments(S);
    CusolverFrontend::AddVariableForArguments<size_t>(strideS);
    CusolverFrontend::AddDevicePointerForArguments(U);
    CusolverFrontend::AddVariableForArguments<int>(ldu);
    CusolverFrontend::AddVariableForArguments<size_t>(strideU);
    CusolverFrontend::AddDevicePointerForArguments(V);
    CusolverFrontend::AddVariableForArguments<int>(ldv);
    CusolverFrontend::AddVariableForArguments<size_t>(strideV);
    CusolverFrontend::AddDevicePointerForArguments(work);
    CusolverFrontend::AddVariableForArguments<int>(lwork);
    CusolverFrontend::AddDevicePointerForArguments(info);
    CusolverFrontend::AddVariableForArguments<int>(batchSize);
    CusolverFrontend::AddHostPointerForArguments(h_R_nrmF);
    CusolverFrontend::Execute("cusolverDnDgesvdaStridedBatched");
    if (CusolverFrontend::Success()) {
        S = (double*) CusolverFrontend::GetOutputDevicePointer();
        U = (double*) CusolverFrontend::GetOutputDevicePointer();
        V = (double*) CusolverFrontend::GetOutputDevicePointer();
        info = (int*) CusolverFrontend::GetOutputDevicePointer();
        h_R_nrmF = CusolverFrontend::GetOutputHostPointer<double>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnCgesvdaStridedBatched(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int rank, int m, int n, const cuComplex *A, int lda, long long int strideA, float *S, long long int strideS, cuComplex *U, int ldu, long long int strideU, cuComplex *V, int ldv, long long int strideV, cuComplex *work, int lwork, int *info, double *h_R_nrmF, int batchSize) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cusolverEigMode_t>(jobz);
    CusolverFrontend::AddVariableForArguments<int>(rank);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddVariableForArguments<size_t>(strideA);
    CusolverFrontend::AddDevicePointerForArguments(S);
    CusolverFrontend::AddVariableForArguments<size_t>(strideS);
    CusolverFrontend::AddDevicePointerForArguments(U);
    CusolverFrontend::AddVariableForArguments<int>(ldu);
    CusolverFrontend::AddVariableForArguments<size_t>(strideU);
    CusolverFrontend::AddDevicePointerForArguments(V);
    CusolverFrontend::AddVariableForArguments<int>(ldv);
    CusolverFrontend::AddVariableForArguments<size_t>(strideV);
    CusolverFrontend::AddDevicePointerForArguments(work);
    CusolverFrontend::AddVariableForArguments<int>(lwork);
    CusolverFrontend::AddDevicePointerForArguments(info);
    CusolverFrontend::AddVariableForArguments<int>(batchSize);
    CusolverFrontend::AddHostPointerForArguments(h_R_nrmF);
    CusolverFrontend::Execute("cusolverDnCgesvdaStridedBatched");
    if (CusolverFrontend::Success()) {
        S = (float*) CusolverFrontend::GetOutputDevicePointer();
        U = (cuComplex*) CusolverFrontend::GetOutputDevicePointer();
        V = (cuComplex*) CusolverFrontend::GetOutputDevicePointer();
        info = (int*) CusolverFrontend::GetOutputDevicePointer();
        h_R_nrmF = CusolverFrontend::GetOutputHostPointer<double>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnZgesvdaStridedBatched(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int rank, int m, int n, const cuDoubleComplex *A, int lda, long long int strideA, double *S, long long int strideS, cuDoubleComplex *U, int ldu, long long int strideU, cuDoubleComplex *V, int ldv, long long int strideV, cuDoubleComplex *work, int lwork, int *info, double *h_R_nrmF, int batchSize) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cusolverEigMode_t>(jobz);
    CusolverFrontend::AddVariableForArguments<int>(rank);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddVariableForArguments<size_t>(strideA);
    CusolverFrontend::AddDevicePointerForArguments(S);
    CusolverFrontend::AddVariableForArguments<size_t>(strideS);
    CusolverFrontend::AddDevicePointerForArguments(U);
    CusolverFrontend::AddVariableForArguments<int>(ldu);
    CusolverFrontend::AddVariableForArguments<size_t>(strideU);
    CusolverFrontend::AddDevicePointerForArguments(V);
    CusolverFrontend::AddVariableForArguments<int>(ldv);
    CusolverFrontend::AddVariableForArguments<size_t>(strideV);
    CusolverFrontend::AddDevicePointerForArguments(work);
    CusolverFrontend::AddVariableForArguments<int>(lwork);
    CusolverFrontend::AddDevicePointerForArguments(info);
    CusolverFrontend::AddVariableForArguments<int>(batchSize);
    CusolverFrontend::AddHostPointerForArguments(h_R_nrmF);
    CusolverFrontend::Execute("cusolverDnZgesvdaStridedBatched");
    if (CusolverFrontend::Success()) {
        S = (double*) CusolverFrontend::GetOutputDevicePointer();
        U = (cuDoubleComplex*) CusolverFrontend::GetOutputDevicePointer();
        V = (cuDoubleComplex*) CusolverFrontend::GetOutputDevicePointer();
        info = (int*) CusolverFrontend::GetOutputDevicePointer();
        h_R_nrmF = CusolverFrontend::GetOutputHostPointer<double>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnSsyevd_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const float *A, int lda, const float *W, int *lwork) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cusolverEigMode_t>(jobz);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(W);
    CusolverFrontend::Execute("cusolverDnSsyevd_bufferSize");
    if (CusolverFrontend::Success()) {
        *lwork = CusolverFrontend::GetOutputVariable<int>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnDsyevd_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const double *A, int lda, const double *W, int *lwork) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cusolverEigMode_t>(jobz);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(W);
    CusolverFrontend::Execute("cusolverDnDsyevd_bufferSize");
    if (CusolverFrontend::Success()) {
        *lwork = CusolverFrontend::GetOutputVariable<int>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnCheevd_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const cuComplex *A, int lda, const float *W, int *lwork) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cusolverEigMode_t>(jobz);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(W);
    CusolverFrontend::Execute("cusolverDnCheevd_bufferSize");
    if (CusolverFrontend::Success()) {
        *lwork = CusolverFrontend::GetOutputVariable<int>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnZheevd_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const cuDoubleComplex *A, int lda, const double *W, int *lwork) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cusolverEigMode_t>(jobz);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(W);
    CusolverFrontend::Execute("cusolverDnZheevd_bufferSize");
    if (CusolverFrontend::Success()) {
        *lwork = CusolverFrontend::GetOutputVariable<int>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnSsyevd(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, float *A, int lda, float *W, float *work, int lwork, int *devInfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cusolverEigMode_t>(jobz);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(W);
    CusolverFrontend::AddDevicePointerForArguments(work);
    CusolverFrontend::AddVariableForArguments<int>(lwork);
    CusolverFrontend::AddDevicePointerForArguments(devInfo);
    CusolverFrontend::Execute("cusolverDnSsyevd");
    if (CusolverFrontend::Success()) {
        W = (float*) CusolverFrontend::GetOutputDevicePointer();
        devInfo = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnDsyevd(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, double *A, int lda, double *W, double *work, int lwork, int *devInfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cusolverEigMode_t>(jobz);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(W);
    CusolverFrontend::AddDevicePointerForArguments(work);
    CusolverFrontend::AddVariableForArguments<int>(lwork);
    CusolverFrontend::AddDevicePointerForArguments(devInfo);
    CusolverFrontend::Execute("cusolverDnDsyevd");
    if (CusolverFrontend::Success()) {
        W = (double*) CusolverFrontend::GetOutputDevicePointer();
        devInfo = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnCheevd(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, cuComplex *A, int lda, float *W, cuComplex *work, int lwork, int *devInfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cusolverEigMode_t>(jobz);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(W);
    CusolverFrontend::AddDevicePointerForArguments(work);
    CusolverFrontend::AddVariableForArguments<int>(lwork);
    CusolverFrontend::AddDevicePointerForArguments(devInfo);
    CusolverFrontend::Execute("cusolverDnCheevd");
    if (CusolverFrontend::Success()) {
        W = (float*) CusolverFrontend::GetOutputDevicePointer();
        devInfo = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnZheevd(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, cuDoubleComplex *A, int lda, double *W, cuDoubleComplex *work, int lwork, int *devInfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cusolverEigMode_t>(jobz);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(W);
    CusolverFrontend::AddDevicePointerForArguments(work);
    CusolverFrontend::AddVariableForArguments<int>(lwork);
    CusolverFrontend::AddDevicePointerForArguments(devInfo);
    CusolverFrontend::Execute("cusolverDnZheevd");
    if (CusolverFrontend::Success()) {
        W = (double*) CusolverFrontend::GetOutputDevicePointer();
        devInfo = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnSyevd_bufferSize(cusolverDnHandle_t handle, cusolverDnParams_t params, cusolverEigMode_t jobz, cublasFillMode_t uplo, int64_t n, cudaDataType dataTypeA, const void *A, int64_t lda, cudaDataType dataTypeW, const void *W, cudaDataType computeType, size_t *workspaceInBytes) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) params);
    CusolverFrontend::AddVariableForArguments<cusolverEigMode_t>(jobz);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<int64_t>(n);
    CusolverFrontend::AddVariableForArguments<cudaDataType_t>(dataTypeA);
    CusolverFrontend::AddVariableForArguments<int64_t>(lda);
    CusolverFrontend::AddVariableForArguments<cudaDataType_t>(dataTypeW);
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
    switch(dataTypeW){
        case CUDA_R_32F:
            //float
            CusolverFrontend::AddDevicePointerForArguments((float *)W);
            break;
        case CUDA_R_64F:
            //double
            CusolverFrontend::AddDevicePointerForArguments((double *)W);
            break;
        case CUDA_C_32F:
            //cuComplex
            CusolverFrontend::AddDevicePointerForArguments((cuComplex *)W);
            break;
        case CUDA_C_64F:
            //cuDoubleComplex
            CusolverFrontend::AddDevicePointerForArguments((cuDoubleComplex *)W);
            break;
        default:
            throw "Type not supported by GVirtus!";
    }
    CusolverFrontend::Execute("cusolverDnSyevd_bufferSize");
    if (CusolverFrontend::Success()) {
        *workspaceInBytes = CusolverFrontend::GetOutputVariable<size_t>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnSyevd(cusolverDnHandle_t handle, cusolverDnParams_t params, cusolverEigMode_t jobz, cublasFillMode_t uplo, int64_t n, cudaDataType dataTypeA, void *A, int64_t lda, cudaDataType dataTypeW, void *W, cudaDataType computeType, void *pBuffer, size_t workspaceInBytes, int *info) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) params);
    CusolverFrontend::AddVariableForArguments<cusolverEigMode_t>(jobz);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<int64_t>(n);
    CusolverFrontend::AddVariableForArguments<cudaDataType_t>(dataTypeA);
    CusolverFrontend::AddVariableForArguments<int64_t>(lda);
    CusolverFrontend::AddVariableForArguments<cudaDataType_t>(dataTypeW);
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
    switch(dataTypeW){
        case CUDA_R_32F:
            //float
            CusolverFrontend::AddDevicePointerForArguments((float *)W);
            break;
        case CUDA_R_64F:
            //double
            CusolverFrontend::AddDevicePointerForArguments((double *)W);
            break;
        case CUDA_C_32F:
            //cuComplex
            CusolverFrontend::AddDevicePointerForArguments((cuComplex *)W);
            break;
        case CUDA_C_64F:
            //cuDoubleComplex
            CusolverFrontend::AddDevicePointerForArguments((cuDoubleComplex *)W);
            break;
        default:
            throw "Type not supported by GVirtus!";
    }
    CusolverFrontend::Execute("cusolverDnSyevd");
    if (CusolverFrontend::Success()) {
        W = CusolverFrontend::GetOutputDevicePointer();
        info = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnSsyevdx_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cusolverEigRange_t range, cublasFillMode_t uplo, int n, const float *A, int lda, float vl, float vu, int il, int iu, int *h_meig, const float *W, int *lwork) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cusolverEigMode_t>(jobz);
    CusolverFrontend::AddVariableForArguments<cusolverEigRange_t>(range);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddVariableForArguments<float>(vl);
    CusolverFrontend::AddVariableForArguments<float>(vu);
    CusolverFrontend::AddVariableForArguments<int>(il);
    CusolverFrontend::AddVariableForArguments<int>(iu);
    CusolverFrontend::AddHostPointerForArguments(h_meig);
    CusolverFrontend::AddDevicePointerForArguments(W);
    CusolverFrontend::Execute("cusolverDnSsyevdx_bufferSize");
    if (CusolverFrontend::Success()) {
        *lwork = CusolverFrontend::GetOutputVariable<int>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnDsyevdx_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cusolverEigRange_t range, cublasFillMode_t uplo, int n, const double *A, int lda, double vl, double vu, int il, int iu, int *h_meig, const double *W, int *lwork) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cusolverEigMode_t>(jobz);
    CusolverFrontend::AddVariableForArguments<cusolverEigRange_t>(range);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddVariableForArguments<double>(vl);
    CusolverFrontend::AddVariableForArguments<double>(vu);
    CusolverFrontend::AddVariableForArguments<int>(il);
    CusolverFrontend::AddVariableForArguments<int>(iu);
    CusolverFrontend::AddHostPointerForArguments(h_meig);
    CusolverFrontend::AddDevicePointerForArguments(W);
    CusolverFrontend::Execute("cusolverDnDsyevdx_bufferSize");
    if (CusolverFrontend::Success()) {
        *lwork = CusolverFrontend::GetOutputVariable<int>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnCheevdx_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cusolverEigRange_t range, cublasFillMode_t uplo, int n, const cuComplex *A, int lda, float vl, float vu, int il, int iu, int *h_meig, const float *W, int *lwork) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cusolverEigMode_t>(jobz);
    CusolverFrontend::AddVariableForArguments<cusolverEigRange_t>(range);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddVariableForArguments<float>(vl);
    CusolverFrontend::AddVariableForArguments<float>(vu);
    CusolverFrontend::AddVariableForArguments<int>(il);
    CusolverFrontend::AddVariableForArguments<int>(iu);
    CusolverFrontend::AddHostPointerForArguments(h_meig);
    CusolverFrontend::AddDevicePointerForArguments(W);
    CusolverFrontend::Execute("cusolverDnCheevdx_bufferSize");
    if (CusolverFrontend::Success()) {
        *lwork = CusolverFrontend::GetOutputVariable<int>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnZheevdx_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cusolverEigRange_t range, cublasFillMode_t uplo, int n, const cuDoubleComplex *A, int lda, double vl, double vu, int il, int iu, int *h_meig, const double *W, int *lwork) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cusolverEigMode_t>(jobz);
    CusolverFrontend::AddVariableForArguments<cusolverEigRange_t>(range);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddVariableForArguments<double>(vl);
    CusolverFrontend::AddVariableForArguments<double>(vu);
    CusolverFrontend::AddVariableForArguments<int>(il);
    CusolverFrontend::AddVariableForArguments<int>(iu);
    CusolverFrontend::AddHostPointerForArguments(h_meig);
    CusolverFrontend::AddDevicePointerForArguments(W);
    CusolverFrontend::Execute("cusolverDnZheevdx_bufferSize");
    if (CusolverFrontend::Success()) {
        *lwork = CusolverFrontend::GetOutputVariable<int>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnSsyevdx(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cusolverEigRange_t range, cublasFillMode_t uplo, int n, float *A, int lda, float vl, float vu, int il, int iu, int *h_meig, float *W, float *work, int lwork, int *devInfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cusolverEigMode_t>(jobz);
    CusolverFrontend::AddVariableForArguments<cusolverEigRange_t>(range);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddVariableForArguments<double>(vl);
    CusolverFrontend::AddVariableForArguments<double>(vu);
    CusolverFrontend::AddVariableForArguments<int>(il);
    CusolverFrontend::AddVariableForArguments<int>(iu);
    CusolverFrontend::AddHostPointerForArguments(h_meig);
    CusolverFrontend::AddDevicePointerForArguments(W);
    CusolverFrontend::AddDevicePointerForArguments(work);
    CusolverFrontend::AddVariableForArguments<int>(lwork);
    CusolverFrontend::AddDevicePointerForArguments(devInfo);
    CusolverFrontend::Execute("cusolverDnSsyevdx");
    if (CusolverFrontend::Success()) {
        h_meig = CusolverFrontend::GetOutputHostPointer<int>();
        W = (float*) CusolverFrontend::GetOutputDevicePointer();
        devInfo = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnDsyevdx(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cusolverEigRange_t range, cublasFillMode_t uplo, int n, double *A, int lda, double vl, double vu, int il, int iu, int *h_meig, double *W, double *work, int lwork, int *devInfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cusolverEigMode_t>(jobz);
    CusolverFrontend::AddVariableForArguments<cusolverEigRange_t>(range);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddVariableForArguments<double>(vl);
    CusolverFrontend::AddVariableForArguments<double>(vu);
    CusolverFrontend::AddVariableForArguments<int>(il);
    CusolverFrontend::AddVariableForArguments<int>(iu);
    CusolverFrontend::AddHostPointerForArguments(h_meig);
    CusolverFrontend::AddDevicePointerForArguments(W);
    CusolverFrontend::AddDevicePointerForArguments(work);
    CusolverFrontend::AddVariableForArguments<int>(lwork);
    CusolverFrontend::AddDevicePointerForArguments(devInfo);
    CusolverFrontend::Execute("cusolverDnDsyevdx");
    if (CusolverFrontend::Success()) {
        h_meig = CusolverFrontend::GetOutputHostPointer<int>();
        W = (double*) CusolverFrontend::GetOutputDevicePointer();
        devInfo = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnCheevdx(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cusolverEigRange_t range, cublasFillMode_t uplo, int n, cuComplex *A, int lda, float vl, float vu, int il, int iu, int *h_meig, float *W, cuComplex *work, int lwork, int *devInfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cusolverEigMode_t>(jobz);
    CusolverFrontend::AddVariableForArguments<cusolverEigRange_t>(range);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddVariableForArguments<double>(vl);
    CusolverFrontend::AddVariableForArguments<double>(vu);
    CusolverFrontend::AddVariableForArguments<int>(il);
    CusolverFrontend::AddVariableForArguments<int>(iu);
    CusolverFrontend::AddHostPointerForArguments(h_meig);
    CusolverFrontend::AddDevicePointerForArguments(W);
    CusolverFrontend::AddDevicePointerForArguments(work);
    CusolverFrontend::AddVariableForArguments<int>(lwork);
    CusolverFrontend::AddDevicePointerForArguments(devInfo);
    CusolverFrontend::Execute("cusolverDnCheevdx");
    if (CusolverFrontend::Success()) {
        h_meig = CusolverFrontend::GetOutputHostPointer<int>();
        W = (float*) CusolverFrontend::GetOutputDevicePointer();
        devInfo = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnZheevdx(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cusolverEigRange_t range, cublasFillMode_t uplo, int n, cuDoubleComplex *A, int lda, double vl, double vu, int il, int iu, int *h_meig, double *W, cuDoubleComplex *work, int lwork, int *devInfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cusolverEigMode_t>(jobz);
    CusolverFrontend::AddVariableForArguments<cusolverEigRange_t>(range);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddVariableForArguments<double>(vl);
    CusolverFrontend::AddVariableForArguments<double>(vu);
    CusolverFrontend::AddVariableForArguments<int>(il);
    CusolverFrontend::AddVariableForArguments<int>(iu);
    CusolverFrontend::AddHostPointerForArguments(h_meig);
    CusolverFrontend::AddDevicePointerForArguments(W);
    CusolverFrontend::AddDevicePointerForArguments(work);
    CusolverFrontend::AddVariableForArguments<int>(lwork);
    CusolverFrontend::AddDevicePointerForArguments(devInfo);
    CusolverFrontend::Execute("cusolverDnZheevdx");
    if (CusolverFrontend::Success()) {
        h_meig = CusolverFrontend::GetOutputHostPointer<int>();
        W = (double*) CusolverFrontend::GetOutputDevicePointer();
        devInfo = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnSsygvd_bufferSize(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const float *A, int lda, const float *B, int ldb, const float *W, int *lwork) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cusolverEigType_t>(itype);
    CusolverFrontend::AddVariableForArguments<cusolverEigMode_t>(jobz);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(B);
    CusolverFrontend::AddVariableForArguments<int>(ldb);
    CusolverFrontend::AddDevicePointerForArguments(W);
    CusolverFrontend::Execute("cusolverDnSsygvd_bufferSize");
    if (CusolverFrontend::Success()) {
        *lwork = CusolverFrontend::GetOutputVariable<int>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnDsygvd_bufferSize(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const double *A, int lda, const double *B, int ldb, const double *W, int *lwork) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cusolverEigType_t>(itype);
    CusolverFrontend::AddVariableForArguments<cusolverEigMode_t>(jobz);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(B);
    CusolverFrontend::AddVariableForArguments<int>(ldb);
    CusolverFrontend::AddDevicePointerForArguments(W);
    CusolverFrontend::Execute("cusolverDnDsygvd_bufferSize");
    if (CusolverFrontend::Success()) {
        *lwork = CusolverFrontend::GetOutputVariable<int>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnChegvd_bufferSize(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const cuComplex *A, int lda, const cuComplex *B, int ldb, const float *W, int *lwork) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cusolverEigType_t>(itype);
    CusolverFrontend::AddVariableForArguments<cusolverEigMode_t>(jobz);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(B);
    CusolverFrontend::AddVariableForArguments<int>(ldb);
    CusolverFrontend::AddDevicePointerForArguments(W);
    CusolverFrontend::Execute("cusolverDnChegvd_bufferSize");
    if (CusolverFrontend::Success()) {
        *lwork = CusolverFrontend::GetOutputVariable<int>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnZhegvd_bufferSize(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb, const double *W, int *lwork) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cusolverEigType_t>(itype);
    CusolverFrontend::AddVariableForArguments<cusolverEigMode_t>(jobz);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(B);
    CusolverFrontend::AddVariableForArguments<int>(ldb);
    CusolverFrontend::AddDevicePointerForArguments(W);
    CusolverFrontend::Execute("cusolverDnZhegvd_bufferSize");
    if (CusolverFrontend::Success()) {
        *lwork = CusolverFrontend::GetOutputVariable<int>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnSsygvd(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, float *A, int lda, float *B, int ldb, float *W, float *work, int lwork, int *devInfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cusolverEigType_t>(itype);
    CusolverFrontend::AddVariableForArguments<cusolverEigMode_t>(jobz);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(B);
    CusolverFrontend::AddVariableForArguments<int>(ldb);
    CusolverFrontend::AddDevicePointerForArguments(W);
    CusolverFrontend::AddDevicePointerForArguments(work);
    CusolverFrontend::AddVariableForArguments<int>(lwork);
    CusolverFrontend::AddDevicePointerForArguments(devInfo);
    CusolverFrontend::Execute("cusolverDnSsygvd");
    if (CusolverFrontend::Success()) {
        W = (float*) CusolverFrontend::GetOutputDevicePointer();
        devInfo = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnDsygvd(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, double *A, int lda, double *B, int ldb, double *W, double *work, int lwork, int *devInfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cusolverEigType_t>(itype);
    CusolverFrontend::AddVariableForArguments<cusolverEigMode_t>(jobz);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(B);
    CusolverFrontend::AddVariableForArguments<int>(ldb);
    CusolverFrontend::AddDevicePointerForArguments(W);
    CusolverFrontend::AddDevicePointerForArguments(work);
    CusolverFrontend::AddVariableForArguments<int>(lwork);
    CusolverFrontend::AddDevicePointerForArguments(devInfo);
    CusolverFrontend::Execute("cusolverDnDsygvd");
    if (CusolverFrontend::Success()) {
        W = (double*) CusolverFrontend::GetOutputDevicePointer();
        devInfo = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnChegvd(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, cuComplex *A, int lda, cuComplex *B, int ldb, float *W, cuComplex *work, int lwork, int *devInfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cusolverEigType_t>(itype);
    CusolverFrontend::AddVariableForArguments<cusolverEigMode_t>(jobz);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(B);
    CusolverFrontend::AddVariableForArguments<int>(ldb);
    CusolverFrontend::AddDevicePointerForArguments(W);
    CusolverFrontend::AddDevicePointerForArguments(work);
    CusolverFrontend::AddVariableForArguments<int>(lwork);
    CusolverFrontend::AddDevicePointerForArguments(devInfo);
    CusolverFrontend::Execute("cusolverDnChegvd");
    if (CusolverFrontend::Success()) {
        W = (float*) CusolverFrontend::GetOutputDevicePointer();
        devInfo = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnZhegvd(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, cuDoubleComplex *A, int lda, cuDoubleComplex *B, int ldb, double *W, cuDoubleComplex *work, int lwork, int *devInfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cusolverEigType_t>(itype);
    CusolverFrontend::AddVariableForArguments<cusolverEigMode_t>(jobz);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(B);
    CusolverFrontend::AddVariableForArguments<int>(ldb);
    CusolverFrontend::AddDevicePointerForArguments(W);
    CusolverFrontend::AddDevicePointerForArguments(work);
    CusolverFrontend::AddVariableForArguments<int>(lwork);
    CusolverFrontend::AddDevicePointerForArguments(devInfo);
    CusolverFrontend::Execute("cusolverDnZhegvd");
    if (CusolverFrontend::Success()) {
        W = (double*) CusolverFrontend::GetOutputDevicePointer();
        devInfo = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnSsygvdx_bufferSize(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cusolverEigRange_t range, cublasFillMode_t uplo, int n, const float *A, int lda, const float *B, int ldb, float vl, float vu, int il, int iu, int *h_meig, const float *W, int *lwork) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cusolverEigType_t>(itype);
    CusolverFrontend::AddVariableForArguments<cusolverEigMode_t>(jobz);
    CusolverFrontend::AddVariableForArguments<cusolverEigRange_t>(range);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(B);
    CusolverFrontend::AddVariableForArguments<int>(ldb);
    CusolverFrontend::AddVariableForArguments<float>(vl);
    CusolverFrontend::AddVariableForArguments<float>(vu);
    CusolverFrontend::AddVariableForArguments<int>(il);
    CusolverFrontend::AddVariableForArguments<int>(iu);
    CusolverFrontend::AddHostPointerForArguments(h_meig);
    CusolverFrontend::AddDevicePointerForArguments(W);
    CusolverFrontend::Execute("cusolverDnSsygvdx_bufferSize");
    if (CusolverFrontend::Success()) {
        *lwork = CusolverFrontend::GetOutputVariable<int>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnDsygvdx_bufferSize(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cusolverEigRange_t range, cublasFillMode_t uplo, int n, const double *A, int lda, const double *B, int ldb, double vl, double vu, int il, int iu, int *h_meig, const double *W, int *lwork) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cusolverEigType_t>(itype);
    CusolverFrontend::AddVariableForArguments<cusolverEigMode_t>(jobz);
    CusolverFrontend::AddVariableForArguments<cusolverEigRange_t>(range);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(B);
    CusolverFrontend::AddVariableForArguments<int>(ldb);
    CusolverFrontend::AddVariableForArguments<double>(vl);
    CusolverFrontend::AddVariableForArguments<double>(vu);
    CusolverFrontend::AddVariableForArguments<int>(il);
    CusolverFrontend::AddVariableForArguments<int>(iu);
    CusolverFrontend::AddHostPointerForArguments(h_meig);
    CusolverFrontend::AddDevicePointerForArguments(W);
    CusolverFrontend::Execute("cusolverDnDsygvdx_bufferSize");
    if (CusolverFrontend::Success()) {
        *lwork = CusolverFrontend::GetOutputVariable<int>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnChegvdx_bufferSize(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cusolverEigRange_t range, cublasFillMode_t uplo, int n, const cuComplex *A, int lda, const cuComplex *B, int ldb, float vl, float vu, int il, int iu, int *h_meig, const float *W, int *lwork) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cusolverEigType_t>(itype);
    CusolverFrontend::AddVariableForArguments<cusolverEigMode_t>(jobz);
    CusolverFrontend::AddVariableForArguments<cusolverEigRange_t>(range);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(B);
    CusolverFrontend::AddVariableForArguments<int>(ldb);
    CusolverFrontend::AddVariableForArguments<float>(vl);
    CusolverFrontend::AddVariableForArguments<float>(vu);
    CusolverFrontend::AddVariableForArguments<int>(il);
    CusolverFrontend::AddVariableForArguments<int>(iu);
    CusolverFrontend::AddHostPointerForArguments(h_meig);
    CusolverFrontend::AddDevicePointerForArguments(W);
    CusolverFrontend::Execute("cusolverDnChegvdx_bufferSize");
    if (CusolverFrontend::Success()) {
        *lwork = CusolverFrontend::GetOutputVariable<int>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnZhegvdx_bufferSize(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cusolverEigRange_t range, cublasFillMode_t uplo, int n, const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb, double vl, double vu, int il, int iu, int *h_meig, const double *W, int *lwork) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cusolverEigType_t>(itype);
    CusolverFrontend::AddVariableForArguments<cusolverEigMode_t>(jobz);
    CusolverFrontend::AddVariableForArguments<cusolverEigRange_t>(range);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(B);
    CusolverFrontend::AddVariableForArguments<int>(ldb);
    CusolverFrontend::AddVariableForArguments<double>(vl);
    CusolverFrontend::AddVariableForArguments<double>(vu);
    CusolverFrontend::AddVariableForArguments<int>(il);
    CusolverFrontend::AddVariableForArguments<int>(iu);
    CusolverFrontend::AddHostPointerForArguments(h_meig);
    CusolverFrontend::AddDevicePointerForArguments(W);
    CusolverFrontend::Execute("cusolverDnZhegvdx_bufferSize");
    if (CusolverFrontend::Success()) {
        *lwork = CusolverFrontend::GetOutputVariable<int>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnSsygvdx(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cusolverEigRange_t range, cublasFillMode_t uplo, int n, float *A, int lda, float *B, int ldb, float vl, float vu, int il, int iu, int *h_meig, float *W, float *work, int lwork, int *devInfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cusolverEigType_t>(itype);
    CusolverFrontend::AddVariableForArguments<cusolverEigMode_t>(jobz);
    CusolverFrontend::AddVariableForArguments<cusolverEigRange_t>(range);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(B);
    CusolverFrontend::AddVariableForArguments<int>(ldb);
    CusolverFrontend::AddVariableForArguments<float>(vl);
    CusolverFrontend::AddVariableForArguments<float>(vu);
    CusolverFrontend::AddVariableForArguments<int>(il);
    CusolverFrontend::AddVariableForArguments<int>(iu);
    CusolverFrontend::AddHostPointerForArguments(h_meig);
    CusolverFrontend::AddDevicePointerForArguments(W);
    CusolverFrontend::AddDevicePointerForArguments(work);
    CusolverFrontend::AddVariableForArguments<int>(lwork);
    CusolverFrontend::AddDevicePointerForArguments(devInfo);
    CusolverFrontend::Execute("cusolverDnSsygvdx");
    if (CusolverFrontend::Success()) {
        h_meig = CusolverFrontend::GetOutputHostPointer<int>();
        W = (float*) CusolverFrontend::GetOutputDevicePointer();
        devInfo = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnDsygvdx(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cusolverEigRange_t range, cublasFillMode_t uplo, int n, double *A, int lda, double *B, int ldb, double vl, double vu, int il, int iu, int *h_meig, double *W, double *work, int lwork, int *devInfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cusolverEigType_t>(itype);
    CusolverFrontend::AddVariableForArguments<cusolverEigMode_t>(jobz);
    CusolverFrontend::AddVariableForArguments<cusolverEigRange_t>(range);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(B);
    CusolverFrontend::AddVariableForArguments<int>(ldb);
    CusolverFrontend::AddVariableForArguments<double>(vl);
    CusolverFrontend::AddVariableForArguments<double>(vu);
    CusolverFrontend::AddVariableForArguments<int>(il);
    CusolverFrontend::AddVariableForArguments<int>(iu);
    CusolverFrontend::AddHostPointerForArguments(h_meig);
    CusolverFrontend::AddDevicePointerForArguments(W);
    CusolverFrontend::AddDevicePointerForArguments(work);
    CusolverFrontend::AddVariableForArguments<int>(lwork);
    CusolverFrontend::AddDevicePointerForArguments(devInfo);
    CusolverFrontend::Execute("cusolverDnDsygvdx");
    if (CusolverFrontend::Success()) {
        h_meig = CusolverFrontend::GetOutputHostPointer<int>();
        W = (double*) CusolverFrontend::GetOutputDevicePointer();
        devInfo = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnChegvdx(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cusolverEigRange_t range, cublasFillMode_t uplo, int n, cuComplex *A, int lda, cuComplex *B, int ldb, float vl, float vu, int il, int iu, int *h_meig, float *W, cuComplex *work, int lwork, int *devInfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cusolverEigType_t>(itype);
    CusolverFrontend::AddVariableForArguments<cusolverEigMode_t>(jobz);
    CusolverFrontend::AddVariableForArguments<cusolverEigRange_t>(range);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(B);
    CusolverFrontend::AddVariableForArguments<int>(ldb);
    CusolverFrontend::AddVariableForArguments<float>(vl);
    CusolverFrontend::AddVariableForArguments<float>(vu);
    CusolverFrontend::AddVariableForArguments<int>(il);
    CusolverFrontend::AddVariableForArguments<int>(iu);
    CusolverFrontend::AddHostPointerForArguments(h_meig);
    CusolverFrontend::AddDevicePointerForArguments(W);
    CusolverFrontend::AddDevicePointerForArguments(work);
    CusolverFrontend::AddVariableForArguments<int>(lwork);
    CusolverFrontend::AddDevicePointerForArguments(devInfo);
    CusolverFrontend::Execute("cusolverDnChegvdx");
    if (CusolverFrontend::Success()) {
        h_meig = CusolverFrontend::GetOutputHostPointer<int>();
        W = (float*) CusolverFrontend::GetOutputDevicePointer();
        devInfo = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnZhegvdx(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cusolverEigRange_t range, cublasFillMode_t uplo, int n, cuDoubleComplex *A, int lda, cuDoubleComplex *B, int ldb, double vl, double vu, int il, int iu, int *h_meig, double *W, cuDoubleComplex *work, int lwork, int *devInfo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cusolverEigType_t>(itype);
    CusolverFrontend::AddVariableForArguments<cusolverEigMode_t>(jobz);
    CusolverFrontend::AddVariableForArguments<cusolverEigRange_t>(range);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(B);
    CusolverFrontend::AddVariableForArguments<int>(ldb);
    CusolverFrontend::AddVariableForArguments<double>(vl);
    CusolverFrontend::AddVariableForArguments<double>(vu);
    CusolverFrontend::AddVariableForArguments<int>(il);
    CusolverFrontend::AddVariableForArguments<int>(iu);
    CusolverFrontend::AddHostPointerForArguments(h_meig);
    CusolverFrontend::AddDevicePointerForArguments(W);
    CusolverFrontend::AddDevicePointerForArguments(work);
    CusolverFrontend::AddVariableForArguments<int>(lwork);
    CusolverFrontend::AddDevicePointerForArguments(devInfo);
    CusolverFrontend::Execute("cusolverDnZhegvdx");
    if (CusolverFrontend::Success()) {
        h_meig = CusolverFrontend::GetOutputHostPointer<int>();
        W = (double*) CusolverFrontend::GetOutputDevicePointer();
        devInfo = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnSsyevj_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const float *A, int lda, const float *W, int *lwork, syevjInfo_t params) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cusolverEigMode_t>(jobz);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(W);
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) params);
    CusolverFrontend::Execute("cusolverDnSsyevj_bufferSize");
    if (CusolverFrontend::Success()) {
        *lwork = CusolverFrontend::GetOutputVariable<int>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnDsyevj_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const double *A, int lda, const double *W, int *lwork, syevjInfo_t params) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cusolverEigMode_t>(jobz);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(W);
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) params);
    CusolverFrontend::Execute("cusolverDnDsyevj_bufferSize");
    if (CusolverFrontend::Success()) {
        *lwork = CusolverFrontend::GetOutputVariable<int>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnCheevj_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const cuComplex *A, int lda, const float *W, int *lwork, syevjInfo_t params) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cusolverEigMode_t>(jobz);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(W);
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) params);
    CusolverFrontend::Execute("cusolverDnCheevj_bufferSize");
    if (CusolverFrontend::Success()) {
        *lwork = CusolverFrontend::GetOutputVariable<int>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnZheevj_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const cuDoubleComplex *A, int lda, const double *W, int *lwork, syevjInfo_t params) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cusolverEigMode_t>(jobz);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(W);
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) params);
    CusolverFrontend::Execute("cusolverDnZheevj_bufferSize");
    if (CusolverFrontend::Success()) {
        *lwork = CusolverFrontend::GetOutputVariable<int>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnSsyevj(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, float *A, int lda, float *W, float *work, int lwork, int *info, syevjInfo_t params) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cusolverEigMode_t>(jobz);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(W);
    CusolverFrontend::AddDevicePointerForArguments(work);
    CusolverFrontend::AddVariableForArguments<int>(lwork);
    CusolverFrontend::AddDevicePointerForArguments(info);
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) params);
    CusolverFrontend::Execute("cusolverDnSsyevj");
    if (CusolverFrontend::Success()) {
        W = (float*) CusolverFrontend::GetOutputDevicePointer();
        info = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnDsyevj(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, double *A, int lda, double *W, double *work, int lwork, int *info, syevjInfo_t params) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cusolverEigMode_t>(jobz);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(W);
    CusolverFrontend::AddDevicePointerForArguments(work);
    CusolverFrontend::AddVariableForArguments<int>(lwork);
    CusolverFrontend::AddDevicePointerForArguments(info);
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) params);
    CusolverFrontend::Execute("cusolverDnDsyevj");
    if (CusolverFrontend::Success()) {
        W = (double*) CusolverFrontend::GetOutputDevicePointer();
        info = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnCheevj(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, cuComplex *A, int lda, float *W, cuComplex *work, int lwork, int *info, syevjInfo_t params) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cusolverEigMode_t>(jobz);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(W);
    CusolverFrontend::AddDevicePointerForArguments(work);
    CusolverFrontend::AddVariableForArguments<int>(lwork);
    CusolverFrontend::AddDevicePointerForArguments(info);
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) params);
    CusolverFrontend::Execute("cusolverDnCheevj");
    if (CusolverFrontend::Success()) {
        W = (float*) CusolverFrontend::GetOutputDevicePointer();
        info = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnZheevj(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, cuDoubleComplex *A, int lda, double *W, cuDoubleComplex *work, int lwork, int *info, syevjInfo_t params) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cusolverEigMode_t>(jobz);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(W);
    CusolverFrontend::AddDevicePointerForArguments(work);
    CusolverFrontend::AddVariableForArguments<int>(lwork);
    CusolverFrontend::AddDevicePointerForArguments(info);
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) params);
    CusolverFrontend::Execute("cusolverDnZheevj");
    if (CusolverFrontend::Success()) {
        W = (double*) CusolverFrontend::GetOutputDevicePointer();
        info = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnSsygvj_bufferSize(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const float *A, int lda, const float *B, int ldb, const float *W, int *lwork, syevjInfo_t params) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cusolverEigType_t>(itype);
    CusolverFrontend::AddVariableForArguments<cusolverEigMode_t>(jobz);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(B);
    CusolverFrontend::AddVariableForArguments<int>(ldb);
    CusolverFrontend::AddDevicePointerForArguments(W);
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) params);
    CusolverFrontend::Execute("cusolverDnSsygvj_bufferSize");
    if (CusolverFrontend::Success()) {
        *lwork = CusolverFrontend::GetOutputVariable<int>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnDsygvj_bufferSize(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const double *A, int lda, const double *B, int ldb, const double *W, int *lwork, syevjInfo_t params) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cusolverEigType_t>(itype);
    CusolverFrontend::AddVariableForArguments<cusolverEigMode_t>(jobz);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(B);
    CusolverFrontend::AddVariableForArguments<int>(ldb);
    CusolverFrontend::AddDevicePointerForArguments(W);
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) params);
    CusolverFrontend::Execute("cusolverDnDsygvj_bufferSize");
    if (CusolverFrontend::Success()) {
        *lwork = CusolverFrontend::GetOutputVariable<int>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnChegvj_bufferSize(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const cuComplex *A, int lda, const cuComplex *B, int ldb, const float *W, int *lwork, syevjInfo_t params) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cusolverEigType_t>(itype);
    CusolverFrontend::AddVariableForArguments<cusolverEigMode_t>(jobz);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(B);
    CusolverFrontend::AddVariableForArguments<int>(ldb);
    CusolverFrontend::AddDevicePointerForArguments(W);
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) params);
    CusolverFrontend::Execute("cusolverDnChegvj_bufferSize");
    if (CusolverFrontend::Success()) {
        *lwork = CusolverFrontend::GetOutputVariable<int>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnZhegvj_bufferSize(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb, const double *W, int *lwork, syevjInfo_t params) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cusolverEigType_t>(itype);
    CusolverFrontend::AddVariableForArguments<cusolverEigMode_t>(jobz);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(B);
    CusolverFrontend::AddVariableForArguments<int>(ldb);
    CusolverFrontend::AddDevicePointerForArguments(W);
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) params);
    CusolverFrontend::Execute("cusolverDnZhegvj_bufferSize");
    if (CusolverFrontend::Success()) {
        *lwork = CusolverFrontend::GetOutputVariable<int>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnSsygvj(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, float *A, int lda, float *B, int ldb, float *W, float *work, int lwork, int *info, syevjInfo_t params) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cusolverEigType_t>(itype);
    CusolverFrontend::AddVariableForArguments<cusolverEigMode_t>(jobz);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(B);
    CusolverFrontend::AddVariableForArguments<int>(ldb);
    CusolverFrontend::AddDevicePointerForArguments(W);
    CusolverFrontend::AddDevicePointerForArguments(work);
    CusolverFrontend::AddVariableForArguments<int>(lwork);
    CusolverFrontend::AddDevicePointerForArguments(info);
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) params);
    CusolverFrontend::Execute("cusolverDnSsygvj");
    if (CusolverFrontend::Success()) {
        W = (float*) CusolverFrontend::GetOutputDevicePointer();
        info = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnDsygvj(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, double *A, int lda, double *B, int ldb, double *W, double *work, int lwork, int *info, syevjInfo_t params) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cusolverEigType_t>(itype);
    CusolverFrontend::AddVariableForArguments<cusolverEigMode_t>(jobz);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(B);
    CusolverFrontend::AddVariableForArguments<int>(ldb);
    CusolverFrontend::AddDevicePointerForArguments(W);
    CusolverFrontend::AddDevicePointerForArguments(work);
    CusolverFrontend::AddVariableForArguments<int>(lwork);
    CusolverFrontend::AddDevicePointerForArguments(info);
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) params);
    CusolverFrontend::Execute("cusolverDnDsygvj");
    if (CusolverFrontend::Success()) {
        W = (double*) CusolverFrontend::GetOutputDevicePointer();
        info = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnChegvj(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, cuComplex *A, int lda, cuComplex *B, int ldb, float *W, cuComplex *work, int lwork, int *info, syevjInfo_t params) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cusolverEigType_t>(itype);
    CusolverFrontend::AddVariableForArguments<cusolverEigMode_t>(jobz);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(B);
    CusolverFrontend::AddVariableForArguments<int>(ldb);
    CusolverFrontend::AddDevicePointerForArguments(W);
    CusolverFrontend::AddDevicePointerForArguments(work);
    CusolverFrontend::AddVariableForArguments<int>(lwork);
    CusolverFrontend::AddDevicePointerForArguments(info);
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) params);
    CusolverFrontend::Execute("cusolverDnChegvj");
    if (CusolverFrontend::Success()) {
        W = (float*) CusolverFrontend::GetOutputDevicePointer();
        info = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnZhegvj(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, cuDoubleComplex *A, int lda, cuDoubleComplex *B, int ldb, double *W, cuDoubleComplex *work, int lwork, int *info, syevjInfo_t params) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cusolverEigType_t>(itype);
    CusolverFrontend::AddVariableForArguments<cusolverEigMode_t>(jobz);
    CusolverFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddDevicePointerForArguments(A);
    CusolverFrontend::AddVariableForArguments<int>(lda);
    CusolverFrontend::AddDevicePointerForArguments(B);
    CusolverFrontend::AddVariableForArguments<int>(ldb);
    CusolverFrontend::AddDevicePointerForArguments(W);
    CusolverFrontend::AddDevicePointerForArguments(work);
    CusolverFrontend::AddVariableForArguments<int>(lwork);
    CusolverFrontend::AddDevicePointerForArguments(info);
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) params);
    CusolverFrontend::Execute("cusolverDnZhegvj");
    if (CusolverFrontend::Success()) {
        W = (double*) CusolverFrontend::GetOutputDevicePointer();
        info = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}