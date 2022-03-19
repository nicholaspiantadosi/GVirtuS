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

extern "C" cusolverStatus_t CUSOLVERAPI cusolverSpScsrlsvluHost(cusolverSpHandle_t handle, int n, int nnzA, const cusparseMatDescr_t descrA, const float *csrValA, const int *csrRowPtrA, const int *csrColIndA, const float *b, float tol, int reorder, float *x, int *singularity) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nnzA);
    CusolverFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusolverFrontend::AddHostPointerForArguments(csrValA, nnzA);
    CusolverFrontend::AddHostPointerForArguments(csrRowPtrA, n + 1);
    CusolverFrontend::AddHostPointerForArguments(csrColIndA, nnzA);
    CusolverFrontend::AddHostPointerForArguments(b, n);
    CusolverFrontend::AddVariableForArguments<float>(tol);
    CusolverFrontend::AddVariableForArguments<int>(reorder);
    CusolverFrontend::AddHostPointerForArguments(x, n);
    CusolverFrontend::Execute("cusolverSpScsrlsvluHost");
    if (CusolverFrontend::Success()) {
        //x = CusolverFrontend::GetOutputHostPointer<float>(n);
        //*x = *(CusolverFrontend::GetOutputHostPointer<float>(n));
        float *tmp = CusolverFrontend::GetOutputHostPointer<float>(n);
        for (int i = 0; i < n; i++) {
            *(x+i) = *(tmp+i);
        }
        *singularity = CusolverFrontend::GetOutputVariable<int>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverSpDcsrlsvluHost(cusolverSpHandle_t handle, int n, int nnzA, const cusparseMatDescr_t descrA, const double *csrValA, const int *csrRowPtrA, const int *csrColIndA, const double *b, double tol, int reorder, double *x, int *singularity) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nnzA);
    CusolverFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusolverFrontend::AddHostPointerForArguments(csrValA, nnzA);
    CusolverFrontend::AddHostPointerForArguments(csrRowPtrA, n + 1);
    CusolverFrontend::AddHostPointerForArguments(csrColIndA, nnzA);
    CusolverFrontend::AddHostPointerForArguments(b, n);
    CusolverFrontend::AddVariableForArguments<double>(tol);
    CusolverFrontend::AddVariableForArguments<int>(reorder);
    CusolverFrontend::AddHostPointerForArguments(x, n);
    CusolverFrontend::Execute("cusolverSpDcsrlsvluHost");
    if (CusolverFrontend::Success()) {
        double *tmp = CusolverFrontend::GetOutputHostPointer<double>(n);
        for (int i = 0; i < n; i++) {
            *(x+i) = *(tmp+i);
        }
        *singularity = CusolverFrontend::GetOutputVariable<int>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverSpCcsrlsvluHost(cusolverSpHandle_t handle, int n, int nnzA, const cusparseMatDescr_t descrA, const cuComplex *csrValA, const int *csrRowPtrA, const int *csrColIndA, const cuComplex *b, float tol, int reorder, cuComplex *x, int *singularity) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nnzA);
    CusolverFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusolverFrontend::AddHostPointerForArguments(csrValA, nnzA);
    CusolverFrontend::AddHostPointerForArguments(csrRowPtrA, n + 1);
    CusolverFrontend::AddHostPointerForArguments(csrColIndA, nnzA);
    CusolverFrontend::AddHostPointerForArguments(b, n);
    CusolverFrontend::AddVariableForArguments<float>(tol);
    CusolverFrontend::AddVariableForArguments<int>(reorder);
    CusolverFrontend::AddHostPointerForArguments(x, n);
    CusolverFrontend::Execute("cusolverSpCcsrlsvluHost");
    if (CusolverFrontend::Success()) {
        cuComplex *tmp = CusolverFrontend::GetOutputHostPointer<cuComplex>(n);
        for (int i = 0; i < n; i++) {
            *(x+i) = *(tmp+i);
        }
        *singularity = CusolverFrontend::GetOutputVariable<int>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverSpZcsrlsvluHost(cusolverSpHandle_t handle, int n, int nnzA, const cusparseMatDescr_t descrA, const cuDoubleComplex *csrValA, const int *csrRowPtrA, const int *csrColIndA, const cuDoubleComplex *b, double tol, int reorder, cuDoubleComplex *x, int *singularity) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nnzA);
    CusolverFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusolverFrontend::AddHostPointerForArguments(csrValA, nnzA);
    CusolverFrontend::AddHostPointerForArguments(csrRowPtrA, n + 1);
    CusolverFrontend::AddHostPointerForArguments(csrColIndA, nnzA);
    CusolverFrontend::AddHostPointerForArguments(b, n);
    CusolverFrontend::AddVariableForArguments<double>(tol);
    CusolverFrontend::AddVariableForArguments<int>(reorder);
    CusolverFrontend::AddHostPointerForArguments(x, n);
    CusolverFrontend::Execute("cusolverSpZcsrlsvluHost");
    if (CusolverFrontend::Success()) {
        cuDoubleComplex *tmp = CusolverFrontend::GetOutputHostPointer<cuDoubleComplex>(n);
        for (int i = 0; i < n; i++) {
            *(x+i) = *(tmp+i);
        }
        *singularity = CusolverFrontend::GetOutputVariable<int>();
    }
    return CusolverFrontend::GetExitCode();
}