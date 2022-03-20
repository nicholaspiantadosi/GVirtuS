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

extern "C" cusolverStatus_t CUSOLVERAPI cusolverRfExtractSplitFactorsHost(cusolverRfHandle_t handle, int* nnzL, int** Lp, int** Li, double** Lx, int* nnzU, int** Up, int** Ui, double** Ux) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::Execute("cusolverRfExtractSplitFactorsHost");
    if(CusolverFrontend::Success()) {
        *nnzL = CusolverFrontend::GetOutputVariable<int>();
        *Lp = CusolverFrontend::GetOutputHostPointer<int>(*nnzL);
        *Li = CusolverFrontend::GetOutputHostPointer<int>(*nnzL);
        *Lx = CusolverFrontend::GetOutputHostPointer<double>(*nnzL);
        *nnzU = CusolverFrontend::GetOutputVariable<int>();
        *Up = CusolverFrontend::GetOutputHostPointer<int>(*nnzU);
        *Ui = CusolverFrontend::GetOutputHostPointer<int>(*nnzU);
        *Ux = CusolverFrontend::GetOutputHostPointer<double>(*nnzU);
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverRfDestroy(cusolverRfHandle_t handle) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::Execute("cusolverRfDestroy");
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverRfGetMatrixFormat(cusolverRfHandle_t handle, cusolverRfMatrixFormat_t *format, cusolverRfUnitDiagonal_t *diag) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::Execute("cusolverRfGetMatrixFormat");
    if(CusolverFrontend::Success()) {
        *format = CusolverFrontend::GetOutputVariable<cusolverRfMatrixFormat_t>();
        *diag = CusolverFrontend::GetOutputVariable<cusolverRfUnitDiagonal_t>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverRfGetNumericProperties(cusolverRfHandle_t handle, double *zero, double *boost) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::Execute("cusolverRfGetNumericProperties");
    if(CusolverFrontend::Success()) {
        *zero = CusolverFrontend::GetOutputVariable<double>();
        *boost = CusolverFrontend::GetOutputVariable<double>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverRfGetNumericBoostReport(cusolverRfHandle_t handle, cusolverRfNumericBoostReport_t *report) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::Execute("cusolverRfGetNumericBoostReport");
    if(CusolverFrontend::Success()) {
        *report = CusolverFrontend::GetOutputVariable<cusolverRfNumericBoostReport_t>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverRfGetResetValuesFastMode(cusolverRfHandle_t handle, cusolverRfResetValuesFastMode_t *fastMode) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::Execute("cusolverRfGetResetValuesFastMode");
    if(CusolverFrontend::Success()) {
        *fastMode = CusolverFrontend::GetOutputVariable<cusolverRfResetValuesFastMode_t>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverRfGetAlgs(cusolverRfHandle_t handle, cusolverRfFactorization_t* fact_alg, cusolverRfTriangularSolve_t* solve_alg) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::Execute("cusolverRfGetAlgs");
    if(CusolverFrontend::Success()) {
        *fact_alg = CusolverFrontend::GetOutputVariable<cusolverRfFactorization_t>();
        *solve_alg = CusolverFrontend::GetOutputVariable<cusolverRfTriangularSolve_t>();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverRfRefactor(cusolverRfHandle_t handle) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::Execute("cusolverRfRefactor");
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverRfResetValues(int n, int nnzA, int* csrRowPtrA, int* csrColIndA, double* csrValA, int* P, int* Q, cusolverRfHandle_t handle) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nnzA);
    CusolverFrontend::AddDevicePointerForArguments(csrRowPtrA);
    CusolverFrontend::AddDevicePointerForArguments(csrColIndA);
    CusolverFrontend::AddDevicePointerForArguments(csrValA);
    CusolverFrontend::AddDevicePointerForArguments(P);
    CusolverFrontend::AddDevicePointerForArguments(Q);
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::Execute("cusolverRfResetValues");
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverRfSetMatrixFormat(cusolverRfHandle_t handle, cusolverRfMatrixFormat_t format, cusolverRfUnitDiagonal_t diag) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cusolverRfMatrixFormat_t>(format);
    CusolverFrontend::AddVariableForArguments<cusolverRfUnitDiagonal_t>(diag);
    CusolverFrontend::Execute("cusolverRfSetMatrixFormat");
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverRfSetNumericProperties(cusolverRfHandle_t handle, double zero, double boost) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<double>(zero);
    CusolverFrontend::AddVariableForArguments<double>(boost);
    CusolverFrontend::Execute("cusolverRfSetNumericProperties");
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverRfSetResetValuesFastMode(cusolverRfHandle_t handle, cusolverRfResetValuesFastMode_t fastMode) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cusolverRfResetValuesFastMode_t>(fastMode);
    CusolverFrontend::Execute("cusolverRfSetResetValuesFastMode");
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverRfSetAlgs(cusolverRfHandle_t handle, cusolverRfFactorization_t fact_alg, cusolverRfTriangularSolve_t alg) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<cusolverRfFactorization_t>(fact_alg);
    CusolverFrontend::AddVariableForArguments<cusolverRfTriangularSolve_t>(alg);
    CusolverFrontend::Execute("cusolverRfSetAlgs");
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverRfSolve(cusolverRfHandle_t handle, int *P, int *Q, int nrhs, double *Temp, int ldt, double *XF, int ldxf) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddDevicePointerForArguments(P);
    CusolverFrontend::AddDevicePointerForArguments(Q);
    CusolverFrontend::AddVariableForArguments<int>(nrhs);
    CusolverFrontend::AddDevicePointerForArguments(Temp);
    CusolverFrontend::AddVariableForArguments<int>(ldt);
    CusolverFrontend::AddDevicePointerForArguments(XF);
    CusolverFrontend::AddVariableForArguments<int>(ldxf);
    CusolverFrontend::Execute("cusolverRfSolve");
    if (CusolverFrontend::Success()) {
        XF = (double*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverRfBatchSetupHost(int batchSize, int n, int nnzA, int* csrRowPtrA, int* csrColIndA, double** csrValA, int nnzL, int* csrRowPtrL, int* csrColIndL, double* csrValL, int nnzU, int* csrRowPtrU, int* csrColIndU, double* csrValU, int* P, int* Q, cusolverRfHandle_t handle) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<int>(batchSize);
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nnzA);
    CusolverFrontend::AddHostPointerForArguments(csrRowPtrA, n + 1);
    CusolverFrontend::AddHostPointerForArguments(csrColIndA, nnzA);
    CusolverFrontend::AddHostPointerForArguments<double*>(csrValA, nnzA);
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
    CusolverFrontend::Execute("cusolverRfBatchSetupHost");
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverRfBatchAnalyze(cusolverRfHandle_t handle) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::Execute("cusolverRfBatchAnalyze");
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverRfBatchResetValues(int batchSize, int n, int nnzA, int* csrRowPtrA, int* csrColIndA, double** csrValA, int* P, int* Q, cusolverRfHandle_t handle) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<int>(n);
    CusolverFrontend::AddVariableForArguments<int>(nnzA);
    CusolverFrontend::AddDevicePointerForArguments(csrRowPtrA);
    CusolverFrontend::AddDevicePointerForArguments(csrColIndA);
    CusolverFrontend::AddDevicePointerForArguments(csrValA);
    CusolverFrontend::AddDevicePointerForArguments(P);
    CusolverFrontend::AddDevicePointerForArguments(Q);
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::Execute("cusolverRfBatchResetValues");
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverRfBatchRefactor(cusolverRfHandle_t handle) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::Execute("cusolverRfBatchRefactor");
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverRfBatchSolve(cusolverRfHandle_t handle, int *P, int *Q, int nrhs, double *Temp, int ldt, double **XF, int ldxf) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddDevicePointerForArguments(P);
    CusolverFrontend::AddDevicePointerForArguments(Q);
    CusolverFrontend::AddVariableForArguments<int>(nrhs);
    CusolverFrontend::AddDevicePointerForArguments(Temp);
    CusolverFrontend::AddVariableForArguments<int>(ldt);
    CusolverFrontend::AddDevicePointerForArguments(XF);
    CusolverFrontend::AddVariableForArguments<int>(ldxf);
    CusolverFrontend::Execute("cusolverRfBatchSolve");
    if (CusolverFrontend::Success()) {
        XF = (double**) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverRfBatchZeroPivot(cusolverRfHandle_t handle, int *position) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::Execute("cusolverRfBatchZeroPivot");
    if (CusolverFrontend::Success()) {
        position = (int*) CusolverFrontend::GetOutputDevicePointer();
    }
    return CusolverFrontend::GetExitCode();
}