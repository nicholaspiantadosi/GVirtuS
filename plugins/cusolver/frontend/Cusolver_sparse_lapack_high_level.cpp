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

extern "C" cusolverStatus_t CUSOLVERAPI cusolverSpScsrlsvqrHost(cusolverSpHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, const float *csrValA, const int *csrRowPtrA, const int *csrColIndA, const float *b, float tol, int reorder, float *x, int *singularity) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(nnz);
    CusolverFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusolverFrontend::AddHostPointerForArguments(csrValA, nnz);
    CusolverFrontend::AddHostPointerForArguments(csrRowPtrA, m + 1);
    CusolverFrontend::AddHostPointerForArguments(csrColIndA, nnz);
    CusolverFrontend::AddHostPointerForArguments(b, m);
    CusolverFrontend::AddVariableForArguments<float>(tol);
    CusolverFrontend::AddVariableForArguments<int>(reorder);
    CusolverFrontend::AddHostPointerForArguments(x, m);
    CusolverFrontend::Execute("cusolverSpScsrlsvqrHost");
    if (CusolverFrontend::Success()) {
        float *tmp = CusolverFrontend::GetOutputHostPointer<float>(m);
        for (int i = 0; i < m; i++) {
            *(x+i) = *(tmp+i);
        }
        *singularity = CusolverFrontend::GetOutputVariable<int>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverSpDcsrlsvqrHost(cusolverSpHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, const double *csrValA, const int *csrRowPtrA, const int *csrColIndA, const double *b, double tol, int reorder, double *x, int *singularity) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(nnz);
    CusolverFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusolverFrontend::AddHostPointerForArguments(csrValA, nnz);
    CusolverFrontend::AddHostPointerForArguments(csrRowPtrA, m + 1);
    CusolverFrontend::AddHostPointerForArguments(csrColIndA, nnz);
    CusolverFrontend::AddHostPointerForArguments(b, m);
    CusolverFrontend::AddVariableForArguments<double>(tol);
    CusolverFrontend::AddVariableForArguments<int>(reorder);
    CusolverFrontend::AddHostPointerForArguments(x, m);
    CusolverFrontend::Execute("cusolverSpDcsrlsvqrHost");
    if (CusolverFrontend::Success()) {
        double *tmp = CusolverFrontend::GetOutputHostPointer<double>(m);
        for (int i = 0; i < m; i++) {
            *(x+i) = *(tmp+i);
        }
        *singularity = CusolverFrontend::GetOutputVariable<int>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverSpCcsrlsvqrHost(cusolverSpHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, const cuComplex *csrValA, const int *csrRowPtrA, const int *csrColIndA, const cuComplex *b, float tol, int reorder, cuComplex *x, int *singularity) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(nnz);
    CusolverFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusolverFrontend::AddHostPointerForArguments(csrValA, nnz);
    CusolverFrontend::AddHostPointerForArguments(csrRowPtrA, m + 1);
    CusolverFrontend::AddHostPointerForArguments(csrColIndA, nnz);
    CusolverFrontend::AddHostPointerForArguments(b, m);
    CusolverFrontend::AddVariableForArguments<float>(tol);
    CusolverFrontend::AddVariableForArguments<int>(reorder);
    CusolverFrontend::AddHostPointerForArguments(x, m);
    CusolverFrontend::Execute("cusolverSpCcsrlsvqrHost");
    if (CusolverFrontend::Success()) {
        cuComplex *tmp = CusolverFrontend::GetOutputHostPointer<cuComplex>(m);
        for (int i = 0; i < m; i++) {
            *(x+i) = *(tmp+i);
        }
        *singularity = CusolverFrontend::GetOutputVariable<int>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverSpZcsrlsvqrHost(cusolverSpHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, const cuDoubleComplex *csrValA, const int *csrRowPtrA, const int *csrColIndA, const cuDoubleComplex *b, double tol, int reorder, cuDoubleComplex *x, int *singularity) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(nnz);
    CusolverFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusolverFrontend::AddHostPointerForArguments(csrValA, nnz);
    CusolverFrontend::AddHostPointerForArguments(csrRowPtrA, m + 1);
    CusolverFrontend::AddHostPointerForArguments(csrColIndA, nnz);
    CusolverFrontend::AddHostPointerForArguments(b, m);
    CusolverFrontend::AddVariableForArguments<double>(tol);
    CusolverFrontend::AddVariableForArguments<int>(reorder);
    CusolverFrontend::AddHostPointerForArguments(x, m);
    CusolverFrontend::Execute("cusolverSpZcsrlsvqrHost");
    if (CusolverFrontend::Success()) {
        cuDoubleComplex *tmp = CusolverFrontend::GetOutputHostPointer<cuDoubleComplex>(m);
        for (int i = 0; i < m; i++) {
            *(x+i) = *(tmp+i);
        }
        *singularity = CusolverFrontend::GetOutputVariable<int>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverSpScsrlsvcholHost(cusolverSpHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, const float *csrValA, const int *csrRowPtrA, const int *csrColIndA, const float *b, float tol, int reorder, float *x, int *singularity) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(nnz);
    CusolverFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusolverFrontend::AddHostPointerForArguments(csrValA, nnz);
    CusolverFrontend::AddHostPointerForArguments(csrRowPtrA, m + 1);
    CusolverFrontend::AddHostPointerForArguments(csrColIndA, nnz);
    CusolverFrontend::AddHostPointerForArguments(b, m);
    CusolverFrontend::AddVariableForArguments<float>(tol);
    CusolverFrontend::AddVariableForArguments<int>(reorder);
    CusolverFrontend::AddHostPointerForArguments(x, m);
    CusolverFrontend::Execute("cusolverSpScsrlsvcholHost");
    if (CusolverFrontend::Success()) {
        float *tmp = CusolverFrontend::GetOutputHostPointer<float>(m);
        for (int i = 0; i < m; i++) {
            *(x+i) = *(tmp+i);
        }
        *singularity = CusolverFrontend::GetOutputVariable<int>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverSpDcsrlsvcholHost(cusolverSpHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, const double *csrValA, const int *csrRowPtrA, const int *csrColIndA, const double *b, double tol, int reorder, double *x, int *singularity) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(nnz);
    CusolverFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusolverFrontend::AddHostPointerForArguments(csrValA, nnz);
    CusolverFrontend::AddHostPointerForArguments(csrRowPtrA, m + 1);
    CusolverFrontend::AddHostPointerForArguments(csrColIndA, nnz);
    CusolverFrontend::AddHostPointerForArguments(b, m);
    CusolverFrontend::AddVariableForArguments<double>(tol);
    CusolverFrontend::AddVariableForArguments<int>(reorder);
    CusolverFrontend::AddHostPointerForArguments(x, m);
    CusolverFrontend::Execute("cusolverSpDcsrlsvcholHost");
    if (CusolverFrontend::Success()) {
        double *tmp = CusolverFrontend::GetOutputHostPointer<double>(m);
        for (int i = 0; i < m; i++) {
            *(x+i) = *(tmp+i);
        }
        *singularity = CusolverFrontend::GetOutputVariable<int>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverSpCcsrlsvcholHost(cusolverSpHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, const cuComplex *csrValA, const int *csrRowPtrA, const int *csrColIndA, const cuComplex *b, float tol, int reorder, cuComplex *x, int *singularity) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(nnz);
    CusolverFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusolverFrontend::AddHostPointerForArguments(csrValA, nnz);
    CusolverFrontend::AddHostPointerForArguments(csrRowPtrA, m + 1);
    CusolverFrontend::AddHostPointerForArguments(csrColIndA, nnz);
    CusolverFrontend::AddHostPointerForArguments(b, m);
    CusolverFrontend::AddVariableForArguments<float>(tol);
    CusolverFrontend::AddVariableForArguments<int>(reorder);
    CusolverFrontend::AddHostPointerForArguments(x, m);
    CusolverFrontend::Execute("cusolverSpCcsrlsvcholHost");
    if (CusolverFrontend::Success()) {
        cuComplex *tmp = CusolverFrontend::GetOutputHostPointer<cuComplex>(m);
        for (int i = 0; i < m; i++) {
            *(x+i) = *(tmp+i);
        }
        *singularity = CusolverFrontend::GetOutputVariable<int>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverSpZcsrlsvcholHost(cusolverSpHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, const cuDoubleComplex *csrValA, const int *csrRowPtrA, const int *csrColIndA, const cuDoubleComplex *b, double tol, int reorder, cuDoubleComplex *x, int *singularity) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(nnz);
    CusolverFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusolverFrontend::AddHostPointerForArguments(csrValA, nnz);
    CusolverFrontend::AddHostPointerForArguments(csrRowPtrA, m + 1);
    CusolverFrontend::AddHostPointerForArguments(csrColIndA, nnz);
    CusolverFrontend::AddHostPointerForArguments(b, m);
    CusolverFrontend::AddVariableForArguments<double>(tol);
    CusolverFrontend::AddVariableForArguments<int>(reorder);
    CusolverFrontend::AddHostPointerForArguments(x, m);
    CusolverFrontend::Execute("cusolverSpZcsrlsvcholHost");
    if (CusolverFrontend::Success()) {
        cuDoubleComplex *tmp = CusolverFrontend::GetOutputHostPointer<cuDoubleComplex>(m);
        for (int i = 0; i < m; i++) {
            *(x+i) = *(tmp+i);
        }
        *singularity = CusolverFrontend::GetOutputVariable<int>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverSpScsrlsqvqrHost(cusolverSpHandle_t handle, int m, int n, int nnz, const cusparseMatDescr_t descrA, const float *csrValA, const int *csrRowPtrA, const int *csrColIndA, const float *b, float tol, int *rankA, float *x, int *p, float *min_norm) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nnz);
    CusolverFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusolverFrontend::AddHostPointerForArguments(csrValA, nnz);
    CusolverFrontend::AddHostPointerForArguments(csrRowPtrA, m + 1);
    CusolverFrontend::AddHostPointerForArguments(csrColIndA, nnz);
    CusolverFrontend::AddHostPointerForArguments(b, m);
    CusolverFrontend::AddVariableForArguments<float>(tol);
    CusolverFrontend::AddHostPointerForArguments(x, n);
    CusolverFrontend::AddHostPointerForArguments(p, n);
    CusolverFrontend::Execute("cusolverSpScsrlsqvqrHost");
    if (CusolverFrontend::Success()) {
        *rankA = CusolverFrontend::GetOutputVariable<int>();
        float *tmp = CusolverFrontend::GetOutputHostPointer<float>(n);
        for (int i = 0; i < n; i++) {
            *(x+i) = *(tmp+i);
        }
        int *tmpP = CusolverFrontend::GetOutputHostPointer<int>(n);
        for (int i = 0; i < n; i++) {
            *(p+i) = *(tmpP+i);
        }
        *min_norm = CusolverFrontend::GetOutputVariable<float>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverSpDcsrlsqvqrHost(cusolverSpHandle_t handle, int m, int n, int nnz, const cusparseMatDescr_t descrA, const double *csrValA, const int *csrRowPtrA, const int *csrColIndA, const double *b, double tol, int *rankA, double *x, int *p, double *min_norm) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nnz);
    CusolverFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusolverFrontend::AddHostPointerForArguments(csrValA, nnz);
    CusolverFrontend::AddHostPointerForArguments(csrRowPtrA, m + 1);
    CusolverFrontend::AddHostPointerForArguments(csrColIndA, nnz);
    CusolverFrontend::AddHostPointerForArguments(b, m);
    CusolverFrontend::AddVariableForArguments<double>(tol);
    CusolverFrontend::AddHostPointerForArguments(x, n);
    CusolverFrontend::AddHostPointerForArguments(p, n);
    CusolverFrontend::Execute("cusolverSpDcsrlsqvqrHost");
    if (CusolverFrontend::Success()) {
        *rankA = CusolverFrontend::GetOutputVariable<int>();
        double *tmp = CusolverFrontend::GetOutputHostPointer<double>(n);
        for (int i = 0; i < n; i++) {
            *(x+i) = *(tmp+i);
        }
        int *tmpP = CusolverFrontend::GetOutputHostPointer<int>(n);
        for (int i = 0; i < n; i++) {
            *(p+i) = *(tmpP+i);
        }
        *min_norm = CusolverFrontend::GetOutputVariable<double>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverSpCcsrlsqvqrHost(cusolverSpHandle_t handle, int m, int n, int nnz, const cusparseMatDescr_t descrA, const cuComplex *csrValA, const int *csrRowPtrA, const int *csrColIndA, const cuComplex *b, float tol, int *rankA, cuComplex *x, int *p, float *min_norm) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nnz);
    CusolverFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusolverFrontend::AddHostPointerForArguments(csrValA, nnz);
    CusolverFrontend::AddHostPointerForArguments(csrRowPtrA, m + 1);
    CusolverFrontend::AddHostPointerForArguments(csrColIndA, nnz);
    CusolverFrontend::AddHostPointerForArguments(b, m);
    CusolverFrontend::AddVariableForArguments<float>(tol);
    CusolverFrontend::AddHostPointerForArguments(x, n);
    CusolverFrontend::AddHostPointerForArguments(p, n);
    CusolverFrontend::Execute("cusolverSpCcsrlsqvqrHost");
    if (CusolverFrontend::Success()) {
        *rankA = CusolverFrontend::GetOutputVariable<int>();
        cuComplex *tmp = CusolverFrontend::GetOutputHostPointer<cuComplex>(n);
        for (int i = 0; i < n; i++) {
            *(x+i) = *(tmp+i);
        }
        int *tmpP = CusolverFrontend::GetOutputHostPointer<int>(n);
        for (int i = 0; i < n; i++) {
            *(p+i) = *(tmpP+i);
        }
        *min_norm = CusolverFrontend::GetOutputVariable<float>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverSpZcsrlsqvqrHost(cusolverSpHandle_t handle, int m, int n, int nnz, const cusparseMatDescr_t descrA, const cuDoubleComplex *csrValA, const int *csrRowPtrA, const int *csrColIndA, const cuDoubleComplex *b, double tol, int *rankA, cuDoubleComplex *x, int *p, double *min_norm) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nnz);
    CusolverFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusolverFrontend::AddHostPointerForArguments(csrValA, nnz);
    CusolverFrontend::AddHostPointerForArguments(csrRowPtrA, m + 1);
    CusolverFrontend::AddHostPointerForArguments(csrColIndA, nnz);
    CusolverFrontend::AddHostPointerForArguments(b, m);
    CusolverFrontend::AddVariableForArguments<double>(tol);
    CusolverFrontend::AddHostPointerForArguments(x, n);
    CusolverFrontend::AddHostPointerForArguments(p, n);
    CusolverFrontend::Execute("cusolverSpZcsrlsqvqrHost");
    if (CusolverFrontend::Success()) {
        *rankA = CusolverFrontend::GetOutputVariable<int>();
        cuDoubleComplex *tmp = CusolverFrontend::GetOutputHostPointer<cuDoubleComplex>(n);
        for (int i = 0; i < n; i++) {
            *(x+i) = *(tmp+i);
        }
        int *tmpP = CusolverFrontend::GetOutputHostPointer<int>(n);
        for (int i = 0; i < n; i++) {
            *(p+i) = *(tmpP+i);
        }
        *min_norm = CusolverFrontend::GetOutputVariable<double>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverSpScsreigvsiHost(cusolverSpHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, const float *csrValA, const int *csrRowPtrA, const int *csrColIndA, float mu0, const float *x0, int maxite, float tol, float *mu, float *x) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(nnz);
    CusolverFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusolverFrontend::AddHostPointerForArguments(csrValA, nnz);
    CusolverFrontend::AddHostPointerForArguments(csrRowPtrA, m + 1);
    CusolverFrontend::AddHostPointerForArguments(csrColIndA, nnz);
    CusolverFrontend::AddVariableForArguments<float>(mu0);
    CusolverFrontend::AddHostPointerForArguments(x0, m);
    CusolverFrontend::AddVariableForArguments<int>(maxite);
    CusolverFrontend::AddVariableForArguments<float>(tol);
    CusolverFrontend::AddHostPointerForArguments(x, m);
    CusolverFrontend::Execute("cusolverSpScsreigvsiHost");
    if (CusolverFrontend::Success()) {
        *mu = CusolverFrontend::GetOutputVariable<float>();
        float *tmp = CusolverFrontend::GetOutputHostPointer<float>(m);
        for (int i = 0; i < m; i++) {
            *(x+i) = *(tmp+i);
        }
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverSpDcsreigvsiHost(cusolverSpHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, const double *csrValA, const int *csrRowPtrA, const int *csrColIndA, double mu0, const double *x0, int maxite, double tol, double *mu, double *x) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(nnz);
    CusolverFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusolverFrontend::AddHostPointerForArguments(csrValA, nnz);
    CusolverFrontend::AddHostPointerForArguments(csrRowPtrA, m + 1);
    CusolverFrontend::AddHostPointerForArguments(csrColIndA, nnz);
    CusolverFrontend::AddVariableForArguments<double>(mu0);
    CusolverFrontend::AddHostPointerForArguments(x0, m);
    CusolverFrontend::AddVariableForArguments<int>(maxite);
    CusolverFrontend::AddVariableForArguments<double>(tol);
    CusolverFrontend::AddHostPointerForArguments(x, m);
    CusolverFrontend::Execute("cusolverSpDcsreigvsiHost");
    if (CusolverFrontend::Success()) {
        *mu = CusolverFrontend::GetOutputVariable<double>();
        double *tmp = CusolverFrontend::GetOutputHostPointer<double>(m);
        for (int i = 0; i < m; i++) {
            *(x+i) = *(tmp+i);
        }
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverSpCcsreigvsiHost(cusolverSpHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, const cuComplex *csrValA, const int *csrRowPtrA, const int *csrColIndA, cuComplex mu0, const cuComplex *x0, int maxite, float tol, cuComplex *mu, cuComplex *x) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(nnz);
    CusolverFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusolverFrontend::AddHostPointerForArguments(csrValA, nnz);
    CusolverFrontend::AddHostPointerForArguments(csrRowPtrA, m + 1);
    CusolverFrontend::AddHostPointerForArguments(csrColIndA, nnz);
    CusolverFrontend::AddVariableForArguments<cuComplex>(mu0);
    CusolverFrontend::AddHostPointerForArguments(x0, m);
    CusolverFrontend::AddVariableForArguments<int>(maxite);
    CusolverFrontend::AddVariableForArguments<float>(tol);
    CusolverFrontend::AddHostPointerForArguments(x, m);
    CusolverFrontend::Execute("cusolverSpCcsreigvsiHost");
    if (CusolverFrontend::Success()) {
        *mu = CusolverFrontend::GetOutputVariable<cuComplex>();
        cuComplex *tmp = CusolverFrontend::GetOutputHostPointer<cuComplex>(m);
        for (int i = 0; i < m; i++) {
            *(x+i) = *(tmp+i);
        }
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverSpZcsreigvsiHost(cusolverSpHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, const cuDoubleComplex *csrValA, const int *csrRowPtrA, const int *csrColIndA, cuDoubleComplex mu0, const cuDoubleComplex *x0, int maxite, double tol, cuDoubleComplex *mu, cuDoubleComplex *x) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<int>(m);
    CusolverFrontend::AddVariableForArguments<int>(nnz);
    CusolverFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusolverFrontend::AddHostPointerForArguments(csrValA, nnz);
    CusolverFrontend::AddHostPointerForArguments(csrRowPtrA, m + 1);
    CusolverFrontend::AddHostPointerForArguments(csrColIndA, nnz);
    CusolverFrontend::AddVariableForArguments<cuDoubleComplex>(mu0);
    CusolverFrontend::AddHostPointerForArguments(x0, m);
    CusolverFrontend::AddVariableForArguments<int>(maxite);
    CusolverFrontend::AddVariableForArguments<double>(tol);
    CusolverFrontend::AddHostPointerForArguments(x, m);
    CusolverFrontend::Execute("cusolverSpZcsreigvsiHost");
    if (CusolverFrontend::Success()) {
        *mu = CusolverFrontend::GetOutputVariable<cuDoubleComplex>();
        cuDoubleComplex *tmp = CusolverFrontend::GetOutputHostPointer<cuDoubleComplex>(m);
        for (int i = 0; i < m; i++) {
            *(x+i) = *(tmp+i);
        }
    }
    return CusolverFrontend::GetExitCode();
}