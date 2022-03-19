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

extern "C" cusolverStatus_t CUSOLVERAPI cusolverSpXcsrsymrcmHost(cusolverSpHandle_t handle, int n, int nnzA, const cusparseMatDescr_t descrA, const int *csrRowPtrA, const int *csrColIndA, int *p) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nnzA);
    CusolverFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusolverFrontend::AddHostPointerForArguments(csrRowPtrA, n + 1);
    CusolverFrontend::AddHostPointerForArguments(csrColIndA, nnzA);
    CusolverFrontend::AddHostPointerForArguments(p, n);
    CusolverFrontend::Execute("cusolverSpXcsrsymrcmHost");
    if (CusolverFrontend::Success()) {
        int *tmp = CusolverFrontend::GetOutputHostPointer<int>(n);
        for (int i = 0; i < n; i++) {
            *(p+i) = *(tmp+i);
        }
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverSpXcsrsymmdqHost(cusolverSpHandle_t handle, int n, int nnzA, const cusparseMatDescr_t descrA, const int *csrRowPtrA, const int *csrColIndA, int *p) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nnzA);
    CusolverFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusolverFrontend::AddHostPointerForArguments(csrRowPtrA, n + 1);
    CusolverFrontend::AddHostPointerForArguments(csrColIndA, nnzA);
    CusolverFrontend::AddHostPointerForArguments(p, n);
    CusolverFrontend::Execute("cusolverSpXcsrsymmdqHost");
    if (CusolverFrontend::Success()) {
        int *tmp = CusolverFrontend::GetOutputHostPointer<int>(n);
        for (int i = 0; i < n; i++) {
            *(p+i) = *(tmp+i);
        }
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverSpXcsrsymamdHost(cusolverSpHandle_t handle, int n, int nnzA, const cusparseMatDescr_t descrA, const int *csrRowPtrA, const int *csrColIndA, int *p) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nnzA);
    CusolverFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusolverFrontend::AddHostPointerForArguments(csrRowPtrA, n + 1);
    CusolverFrontend::AddHostPointerForArguments(csrColIndA, nnzA);
    CusolverFrontend::AddHostPointerForArguments(p, n);
    CusolverFrontend::Execute("cusolverSpXcsrsymamdHost");
    if (CusolverFrontend::Success()) {
        int *tmp = CusolverFrontend::GetOutputHostPointer<int>(n);
        for (int i = 0; i < n; i++) {
            *(p+i) = *(tmp+i);
        }
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverSpXcsrmetisndHost(cusolverSpHandle_t handle, int n, int nnzA, const cusparseMatDescr_t descrA, const int *csrRowPtrA, const int *csrColIndA, const int64_t *options, int *p) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nnzA);
    CusolverFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusolverFrontend::AddHostPointerForArguments(csrRowPtrA, n + 1);
    CusolverFrontend::AddHostPointerForArguments(csrColIndA, nnzA);
    CusolverFrontend::AddHostPointerForArguments(options, n);
    CusolverFrontend::AddHostPointerForArguments(p, n);
    CusolverFrontend::Execute("cusolverSpXcsrmetisndHost");
    if (CusolverFrontend::Success()) {
        int *tmp = CusolverFrontend::GetOutputHostPointer<int>(n);
        for (int i = 0; i < n; i++) {
            *(p+i) = *(tmp+i);
        }
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverSpScsrzfdHost(cusolverSpHandle_t handle, int n, int nnzA, const cusparseMatDescr_t descrA, const float *csrValA, const int *csrRowPtrA, const int *csrColIndA, int *P, int *numnz) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nnzA);
    CusolverFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusolverFrontend::AddHostPointerForArguments(csrValA, nnzA);
    CusolverFrontend::AddHostPointerForArguments(csrRowPtrA, n + 1);
    CusolverFrontend::AddHostPointerForArguments(csrColIndA, nnzA);
    CusolverFrontend::AddHostPointerForArguments(P, n);
    CusolverFrontend::Execute("cusolverSpScsrzfdHost");
    if (CusolverFrontend::Success()) {
        int *tmp = CusolverFrontend::GetOutputHostPointer<int>(n);
        for (int i = 0; i < n; i++) {
            *(P+i) = *(tmp+i);
        }
        *numnz = CusolverFrontend::GetOutputVariable<int>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverSpDcsrzfdHost(cusolverSpHandle_t handle, int n, int nnzA, const cusparseMatDescr_t descrA, const double *csrValA, const int *csrRowPtrA, const int *csrColIndA, int *P, int *numnz) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nnzA);
    CusolverFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusolverFrontend::AddHostPointerForArguments(csrValA, nnzA);
    CusolverFrontend::AddHostPointerForArguments(csrRowPtrA, n + 1);
    CusolverFrontend::AddHostPointerForArguments(csrColIndA, nnzA);
    CusolverFrontend::AddHostPointerForArguments(P, n);
    CusolverFrontend::Execute("cusolverSpDcsrzfdHost");
    if (CusolverFrontend::Success()) {
        int *tmp = CusolverFrontend::GetOutputHostPointer<int>(n);
        for (int i = 0; i < n; i++) {
            *(P+i) = *(tmp+i);
        }
        *numnz = CusolverFrontend::GetOutputVariable<int>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverSpCcsrzfdHost(cusolverSpHandle_t handle, int n, int nnzA, const cusparseMatDescr_t descrA, const cuComplex *csrValA, const int *csrRowPtrA, const int *csrColIndA, int *P, int *numnz) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nnzA);
    CusolverFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusolverFrontend::AddHostPointerForArguments(csrValA, nnzA);
    CusolverFrontend::AddHostPointerForArguments(csrRowPtrA, n + 1);
    CusolverFrontend::AddHostPointerForArguments(csrColIndA, nnzA);
    CusolverFrontend::AddHostPointerForArguments(P, n);
    CusolverFrontend::Execute("cusolverSpCcsrzfdHost");
    if (CusolverFrontend::Success()) {
        int *tmp = CusolverFrontend::GetOutputHostPointer<int>(n);
        for (int i = 0; i < n; i++) {
            *(P+i) = *(tmp+i);
        }
        *numnz = CusolverFrontend::GetOutputVariable<int>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverSpZcsrzfdHost(cusolverSpHandle_t handle, int n, int nnzA, const cusparseMatDescr_t descrA, const cuDoubleComplex *csrValA, const int *csrRowPtrA, const int *csrColIndA, int *P, int *numnz) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nnzA);
    CusolverFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusolverFrontend::AddHostPointerForArguments(csrValA, nnzA);
    CusolverFrontend::AddHostPointerForArguments(csrRowPtrA, n + 1);
    CusolverFrontend::AddHostPointerForArguments(csrColIndA, nnzA);
    CusolverFrontend::AddHostPointerForArguments(P, n);
    CusolverFrontend::Execute("cusolverSpZcsrzfdHost");
    if (CusolverFrontend::Success()) {
        int *tmp = CusolverFrontend::GetOutputHostPointer<int>(n);
        for (int i = 0; i < n; i++) {
            *(P+i) = *(tmp+i);
        }
        *numnz = CusolverFrontend::GetOutputVariable<int>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverSpXcsrperm_bufferSizeHost(cusolverSpHandle_t handle, int m, int n, int nnzA, const cusparseMatDescr_t descrA, const int *csrRowPtrA, const int *csrColIndA, const int *p, const int *q, size_t *bufferSizeInBytes) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nnzA);
    CusolverFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusolverFrontend::AddHostPointerForArguments(csrRowPtrA, m + 1);
    CusolverFrontend::AddHostPointerForArguments(csrColIndA, nnzA);
    CusolverFrontend::AddHostPointerForArguments(p, m);
    CusolverFrontend::AddHostPointerForArguments(q, n);
    CusolverFrontend::Execute("cusolverSpXcsrperm_bufferSizeHost");
    if (CusolverFrontend::Success()) {
        *bufferSizeInBytes = CusolverFrontend::GetOutputVariable<size_t>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverSpXcsrpermHost(cusolverSpHandle_t handle, int m, int n, int nnzA, const cusparseMatDescr_t descrA, int *csrRowPtrA, int *csrColIndA, const int *p, const int *q, int *map, void *pBuffer) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nnzA);
    CusolverFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusolverFrontend::AddHostPointerForArguments(csrRowPtrA, m + 1);
    CusolverFrontend::AddHostPointerForArguments(csrColIndA, nnzA);
    CusolverFrontend::AddHostPointerForArguments(p, m);
    CusolverFrontend::AddHostPointerForArguments(q, n);
    CusolverFrontend::AddHostPointerForArguments(map, nnzA);
    CusolverFrontend::AddHostPointerForArguments((size_t*)pBuffer);
    CusolverFrontend::Execute("cusolverSpXcsrpermHost");
    if (CusolverFrontend::Success()) {
        int *tmp1 = CusolverFrontend::GetOutputHostPointer<int>(m+1);
        for (int i = 0; i < m+1; i++) {
            *(csrRowPtrA+i) = *(tmp1+i);
        }
        int *tmp2 = CusolverFrontend::GetOutputHostPointer<int>(nnzA);
        int *tmp3 = CusolverFrontend::GetOutputHostPointer<int>(nnzA);
        for (int i = 0; i < nnzA; i++) {
            *(csrColIndA+i) = *(tmp2+i);
            *(map+i) = *(tmp3+i);
        }
    }
    return CusolverFrontend::GetExitCode();
}