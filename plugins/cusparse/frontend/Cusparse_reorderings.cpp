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
 * Department of Science and Technologies
 */

#include "Cusparse.h"

using namespace std;

extern "C" cusparseStatus_t CUSPARSEAPI cusparseScsrcolor(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, const float* csrValA, const int* csrRowPtrA, const int* csrColIndA, const float* fractionToColor, int* ncolors, int* coloring, int* reordering, cusparseColorInfo_t info) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddVariableForArguments<int>(nnz);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusparseFrontend::AddDevicePointerForArguments(csrValA);
    CusparseFrontend::AddDevicePointerForArguments(csrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(csrColIndA);
    CusparseFrontend::AddHostPointerForArguments((fractionToColor));
    CusparseFrontend::AddHostPointerForArguments(ncolors);
    CusparseFrontend::AddDevicePointerForArguments(coloring);
    CusparseFrontend::AddDevicePointerForArguments(reordering);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)info);
    CusparseFrontend::Execute("cusparseScsrcolor");
    if (CusparseFrontend::Success()) {
        ncolors = (int*)CusparseFrontend::GetOutputDevicePointer();
        coloring = (int*)CusparseFrontend::GetOutputDevicePointer();
        reordering = (int*)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseDcsrcolor(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, const double* csrValA, const int* csrRowPtrA, const int* csrColIndA, const double* fractionToColor, int* ncolors, int* coloring, int* reordering, cusparseColorInfo_t info) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddVariableForArguments<int>(nnz);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusparseFrontend::AddDevicePointerForArguments(csrValA);
    CusparseFrontend::AddDevicePointerForArguments(csrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(csrColIndA);
    CusparseFrontend::AddHostPointerForArguments((fractionToColor));
    CusparseFrontend::AddHostPointerForArguments(ncolors);
    CusparseFrontend::AddDevicePointerForArguments(coloring);
    CusparseFrontend::AddDevicePointerForArguments(reordering);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)info);
    CusparseFrontend::Execute("cusparseDcsrcolor");
    if (CusparseFrontend::Success()) {
        ncolors = (int*)CusparseFrontend::GetOutputDevicePointer();
        coloring = (int*)CusparseFrontend::GetOutputDevicePointer();
        reordering = (int*)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseCcsrcolor(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, const cuComplex* csrValA, const int* csrRowPtrA, const int* csrColIndA, const float* fractionToColor, int* ncolors, int* coloring, int* reordering, cusparseColorInfo_t info) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddVariableForArguments<int>(nnz);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusparseFrontend::AddDevicePointerForArguments(csrValA);
    CusparseFrontend::AddDevicePointerForArguments(csrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(csrColIndA);
    CusparseFrontend::AddHostPointerForArguments((fractionToColor));
    CusparseFrontend::AddHostPointerForArguments(ncolors);
    CusparseFrontend::AddDevicePointerForArguments(coloring);
    CusparseFrontend::AddDevicePointerForArguments(reordering);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)info);
    CusparseFrontend::Execute("cusparseCcsrcolor");
    if (CusparseFrontend::Success()) {
        ncolors = (int*)CusparseFrontend::GetOutputDevicePointer();
        coloring = (int*)CusparseFrontend::GetOutputDevicePointer();
        reordering = (int*)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseZcsrcolor(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, const cuDoubleComplex* csrValA, const int* csrRowPtrA, const int* csrColIndA, const double* fractionToColor, int* ncolors, int* coloring, int* reordering, cusparseColorInfo_t info) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddVariableForArguments<int>(nnz);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrA);
    CusparseFrontend::AddDevicePointerForArguments(csrValA);
    CusparseFrontend::AddDevicePointerForArguments(csrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(csrColIndA);
    CusparseFrontend::AddHostPointerForArguments((fractionToColor));
    CusparseFrontend::AddHostPointerForArguments(ncolors);
    CusparseFrontend::AddDevicePointerForArguments(coloring);
    CusparseFrontend::AddDevicePointerForArguments(reordering);
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)info);
    CusparseFrontend::Execute("cusparseZcsrcolor");
    if (CusparseFrontend::Success()) {
        ncolors = (int*)CusparseFrontend::GetOutputDevicePointer();
        coloring = (int*)CusparseFrontend::GetOutputDevicePointer();
        reordering = (int*)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}