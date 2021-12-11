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

extern "C" cusparseStatus_t CUSPARSEAPI cusparseSaxpyi(cusparseHandle_t handle, int nnz, const float* alpha, const float* xVal, const int* xInd, float* y, cusparseIndexBase_t idxBase) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CusparseFrontend::AddVariableForArguments<int>(nnz);
    CusparseFrontend::AddHostPointerForArguments(const_cast<float *>(alpha));
    CusparseFrontend::AddDevicePointerForArguments(xVal);
    CusparseFrontend::AddDevicePointerForArguments(xInd);
    CusparseFrontend::AddDevicePointerForArguments(y);
    CusparseFrontend::AddVariableForArguments<cusparseIndexBase_t>(idxBase);
    CusparseFrontend::Execute("cusparseSaxpyi");
    if (CusparseFrontend::Success()) {
        y = (float *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseDaxpyi(cusparseHandle_t handle, int nnz, const double* alpha, const double* xVal, const int* xInd, double* y, cusparseIndexBase_t idxBase) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CusparseFrontend::AddVariableForArguments<int>(nnz);
    CusparseFrontend::AddHostPointerForArguments(const_cast<double *>(alpha));
    CusparseFrontend::AddDevicePointerForArguments(xVal);
    CusparseFrontend::AddDevicePointerForArguments(xInd);
    CusparseFrontend::AddDevicePointerForArguments(y);
    CusparseFrontend::AddVariableForArguments<cusparseIndexBase_t>(idxBase);
    CusparseFrontend::Execute("cusparseDaxpyi");
    if (CusparseFrontend::Success()) {
        y = (double *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseCaxpyi(cusparseHandle_t handle, int nnz, const cuComplex* alpha, const cuComplex* xVal, const int* xInd, cuComplex* y, cusparseIndexBase_t idxBase) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CusparseFrontend::AddVariableForArguments<int>(nnz);
    CusparseFrontend::AddHostPointerForArguments(const_cast<cuComplex *>(alpha));
    CusparseFrontend::AddDevicePointerForArguments(xVal);
    CusparseFrontend::AddDevicePointerForArguments(xInd);
    CusparseFrontend::AddDevicePointerForArguments(y);
    CusparseFrontend::AddVariableForArguments<cusparseIndexBase_t>(idxBase);
    CusparseFrontend::Execute("cusparseCaxpyi");
    if (CusparseFrontend::Success()) {
        y = (cuComplex *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseZaxpyi(cusparseHandle_t handle, int nnz, const cuDoubleComplex* alpha, const cuDoubleComplex* xVal, const int* xInd, cuDoubleComplex* y, cusparseIndexBase_t idxBase) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CusparseFrontend::AddVariableForArguments<int>(nnz);
    CusparseFrontend::AddHostPointerForArguments(const_cast<cuDoubleComplex *>(alpha));
    CusparseFrontend::AddDevicePointerForArguments(xVal);
    CusparseFrontend::AddDevicePointerForArguments(xInd);
    CusparseFrontend::AddDevicePointerForArguments(y);
    CusparseFrontend::AddVariableForArguments<cusparseIndexBase_t>(idxBase);
    CusparseFrontend::Execute("cusparseZaxpyi");
    if (CusparseFrontend::Success()) {
        y = (cuDoubleComplex *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseSgthr(cusparseHandle_t handle, int nnz, const float* y, float* xVal, const int* xInd, cusparseIndexBase_t idxBase) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CusparseFrontend::AddVariableForArguments<int>(nnz);
    CusparseFrontend::AddDevicePointerForArguments(y);
    CusparseFrontend::AddDevicePointerForArguments(xVal);
    CusparseFrontend::AddDevicePointerForArguments(xInd);
    CusparseFrontend::AddVariableForArguments<cusparseIndexBase_t>(idxBase);
    CusparseFrontend::Execute("cusparseSgthr");
    if (CusparseFrontend::Success()) {
        xVal = (float *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseDgthr(cusparseHandle_t handle, int nnz, const double* y, double* xVal, const int* xInd, cusparseIndexBase_t idxBase) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CusparseFrontend::AddVariableForArguments<int>(nnz);
    CusparseFrontend::AddDevicePointerForArguments(y);
    CusparseFrontend::AddDevicePointerForArguments(xVal);
    CusparseFrontend::AddDevicePointerForArguments(xInd);
    CusparseFrontend::AddVariableForArguments<cusparseIndexBase_t>(idxBase);
    CusparseFrontend::Execute("cusparseDgthr");
    if (CusparseFrontend::Success()) {
        xVal = (double *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseCgthr(cusparseHandle_t handle, int nnz, const cuComplex* y, cuComplex* xVal, const int* xInd, cusparseIndexBase_t idxBase) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CusparseFrontend::AddVariableForArguments<int>(nnz);
    CusparseFrontend::AddDevicePointerForArguments(y);
    CusparseFrontend::AddDevicePointerForArguments(xVal);
    CusparseFrontend::AddDevicePointerForArguments(xInd);
    CusparseFrontend::AddVariableForArguments<cusparseIndexBase_t>(idxBase);
    CusparseFrontend::Execute("cusparseCgthr");
    if (CusparseFrontend::Success()) {
        xVal = (cuComplex *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseZgthr(cusparseHandle_t handle, int nnz, const cuDoubleComplex* y, cuDoubleComplex* xVal, const int* xInd, cusparseIndexBase_t idxBase) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CusparseFrontend::AddVariableForArguments<int>(nnz);
    CusparseFrontend::AddDevicePointerForArguments(y);
    CusparseFrontend::AddDevicePointerForArguments(xVal);
    CusparseFrontend::AddDevicePointerForArguments(xInd);
    CusparseFrontend::AddVariableForArguments<cusparseIndexBase_t>(idxBase);
    CusparseFrontend::Execute("cusparseZgthr");
    if (CusparseFrontend::Success()) {
        xVal = (cuDoubleComplex *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseSgthrz(cusparseHandle_t handle, int nnz, float* y, float* xVal, const int* xInd, cusparseIndexBase_t idxBase) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CusparseFrontend::AddVariableForArguments<int>(nnz);
    CusparseFrontend::AddDevicePointerForArguments(y);
    CusparseFrontend::AddDevicePointerForArguments(xVal);
    CusparseFrontend::AddDevicePointerForArguments(xInd);
    CusparseFrontend::AddVariableForArguments<cusparseIndexBase_t>(idxBase);
    CusparseFrontend::Execute("cusparseSgthrz");
    if (CusparseFrontend::Success()) {
        xVal = (float *)CusparseFrontend::GetOutputDevicePointer();
        y = (float *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseDgthrz(cusparseHandle_t handle, int nnz, double* y, double* xVal, const int* xInd, cusparseIndexBase_t idxBase) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CusparseFrontend::AddVariableForArguments<int>(nnz);
    CusparseFrontend::AddDevicePointerForArguments(y);
    CusparseFrontend::AddDevicePointerForArguments(xVal);
    CusparseFrontend::AddDevicePointerForArguments(xInd);
    CusparseFrontend::AddVariableForArguments<cusparseIndexBase_t>(idxBase);
    CusparseFrontend::Execute("cusparseDgthrz");
    if (CusparseFrontend::Success()) {
        xVal = (double *)CusparseFrontend::GetOutputDevicePointer();
        y = (double *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseCgthrz(cusparseHandle_t handle, int nnz, cuComplex* y, cuComplex* xVal, const int* xInd, cusparseIndexBase_t idxBase) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CusparseFrontend::AddVariableForArguments<int>(nnz);
    CusparseFrontend::AddDevicePointerForArguments(y);
    CusparseFrontend::AddDevicePointerForArguments(xVal);
    CusparseFrontend::AddDevicePointerForArguments(xInd);
    CusparseFrontend::AddVariableForArguments<cusparseIndexBase_t>(idxBase);
    CusparseFrontend::Execute("cusparseCgthrz");
    if (CusparseFrontend::Success()) {
        xVal = (cuComplex *)CusparseFrontend::GetOutputDevicePointer();
        y = (cuComplex *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseZgthrz(cusparseHandle_t handle, int nnz, cuDoubleComplex* y, cuDoubleComplex* xVal, const int* xInd, cusparseIndexBase_t idxBase) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CusparseFrontend::AddVariableForArguments<int>(nnz);
    CusparseFrontend::AddDevicePointerForArguments(y);
    CusparseFrontend::AddDevicePointerForArguments(xVal);
    CusparseFrontend::AddDevicePointerForArguments(xInd);
    CusparseFrontend::AddVariableForArguments<cusparseIndexBase_t>(idxBase);
    CusparseFrontend::Execute("cusparseZgthrz");
    if (CusparseFrontend::Success()) {
        xVal = (cuDoubleComplex *)CusparseFrontend::GetOutputDevicePointer();
        y = (cuDoubleComplex *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseSroti(cusparseHandle_t handle, int nnz, float* xVal, const int* xInd, float* y, const float* c, const float* s, cusparseIndexBase_t idxBase) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CusparseFrontend::AddVariableForArguments<int>(nnz);
    CusparseFrontend::AddDevicePointerForArguments(xVal);
    CusparseFrontend::AddDevicePointerForArguments(xInd);
    CusparseFrontend::AddDevicePointerForArguments(y);
    CusparseFrontend::AddHostPointerForArguments(c);
    CusparseFrontend::AddHostPointerForArguments(s);
    CusparseFrontend::AddVariableForArguments<cusparseIndexBase_t>(idxBase);
    CusparseFrontend::Execute("cusparseSroti");
    if (CusparseFrontend::Success()) {
        y = (float *)CusparseFrontend::GetOutputDevicePointer();
        xVal = (float *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseDroti(cusparseHandle_t handle, int nnz, double* xVal, const int* xInd, double* y, const double* c, const double* s, cusparseIndexBase_t idxBase) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CusparseFrontend::AddVariableForArguments<int>(nnz);
    CusparseFrontend::AddDevicePointerForArguments(xVal);
    CusparseFrontend::AddDevicePointerForArguments(xInd);
    CusparseFrontend::AddDevicePointerForArguments(y);
    CusparseFrontend::AddHostPointerForArguments(c);
    CusparseFrontend::AddHostPointerForArguments(s);
    CusparseFrontend::AddVariableForArguments<cusparseIndexBase_t>(idxBase);
    CusparseFrontend::Execute("cusparseDroti");
    if (CusparseFrontend::Success()) {
        y = (double *)CusparseFrontend::GetOutputDevicePointer();
        xVal = (double *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseSsctr(cusparseHandle_t handle, int nnz, const float* xVal, const int* xInd, float* y, cusparseIndexBase_t idxBase) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CusparseFrontend::AddVariableForArguments<int>(nnz);
    CusparseFrontend::AddDevicePointerForArguments(xVal);
    CusparseFrontend::AddDevicePointerForArguments(xInd);
    CusparseFrontend::AddDevicePointerForArguments(y);
    CusparseFrontend::AddVariableForArguments<cusparseIndexBase_t>(idxBase);
    CusparseFrontend::Execute("cusparseSsctr");
    if (CusparseFrontend::Success()) {
        y = (float *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseDsctr(cusparseHandle_t handle, int nnz, const double* xVal, const int* xInd, double* y, cusparseIndexBase_t idxBase) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CusparseFrontend::AddVariableForArguments<int>(nnz);
    CusparseFrontend::AddDevicePointerForArguments(xVal);
    CusparseFrontend::AddDevicePointerForArguments(xInd);
    CusparseFrontend::AddDevicePointerForArguments(y);
    CusparseFrontend::AddVariableForArguments<cusparseIndexBase_t>(idxBase);
    CusparseFrontend::Execute("cusparseDsctr");
    if (CusparseFrontend::Success()) {
        y = (double *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseCsctr(cusparseHandle_t handle, int nnz, const cuComplex* xVal, const int* xInd, cuComplex* y, cusparseIndexBase_t idxBase) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CusparseFrontend::AddVariableForArguments<int>(nnz);
    CusparseFrontend::AddDevicePointerForArguments(xVal);
    CusparseFrontend::AddDevicePointerForArguments(xInd);
    CusparseFrontend::AddDevicePointerForArguments(y);
    CusparseFrontend::AddVariableForArguments<cusparseIndexBase_t>(idxBase);
    CusparseFrontend::Execute("cusparseCsctr");
    if (CusparseFrontend::Success()) {
        y = (cuComplex *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseZsctr(cusparseHandle_t handle, int nnz, const cuDoubleComplex* xVal, const int* xInd, cuDoubleComplex* y, cusparseIndexBase_t idxBase) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CusparseFrontend::AddVariableForArguments<int>(nnz);
    CusparseFrontend::AddDevicePointerForArguments(xVal);
    CusparseFrontend::AddDevicePointerForArguments(xInd);
    CusparseFrontend::AddDevicePointerForArguments(y);
    CusparseFrontend::AddVariableForArguments<cusparseIndexBase_t>(idxBase);
    CusparseFrontend::Execute("cusparseZsctr");
    if (CusparseFrontend::Success()) {
        y = (cuDoubleComplex *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}