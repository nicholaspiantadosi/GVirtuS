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