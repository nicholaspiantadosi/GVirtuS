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

extern "C" cusparseStatus_t CUSPARSEAPI cusparseCreateColorInfo(cusparseColorInfo_t * info){
    CusparseFrontend::Prepare();
    CusparseFrontend::AddHostPointerForArguments<cusparseColorInfo_t>(info);
    CusparseFrontend::Execute("cusparseCreateColorInfo");
    if(CusparseFrontend::Success())
        *info = CusparseFrontend::GetOutputVariable<cusparseColorInfo_t>();
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseCreateMatDescr(cusparseMatDescr_t * descrA){
    CusparseFrontend::Prepare();
    CusparseFrontend::AddHostPointerForArguments<cusparseMatDescr_t>(descrA);
    CusparseFrontend::Execute("cusparseCreateMatDescr");
    if(CusparseFrontend::Success())
        *descrA = CusparseFrontend::GetOutputVariable<cusparseMatDescr_t>();
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseDestroyColorInfo(cusparseColorInfo_t info){
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<long long int>((long long int) info);
    CusparseFrontend::Execute("cusparseDestroyColorInfo");
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseDestroyMatDescr(cusparseMatDescr_t descrA){
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<long long int>((long long int) descrA);
    CusparseFrontend::Execute("cusparseDestroyMatDescr");
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseDiagType_t CUSPARSEAPI cusparseGetMatDiagType(cusparseMatDescr_t descrA){
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<long long int>((long long int) descrA);
    CusparseFrontend::Execute("cusparseGetMatDiagType");
    cusparseDiagType_t diagType;
    if(CusparseFrontend::Success())
        diagType = CusparseFrontend::GetOutputVariable<cusparseDiagType_t>();
    return diagType;
}

extern "C" cusparseFillMode_t CUSPARSEAPI cusparseGetMatFillMode(cusparseMatDescr_t descrA){
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<long long int>((long long int) descrA);
    CusparseFrontend::Execute("cusparseGetMatFillMode");
    cusparseFillMode_t fillMode;
    if(CusparseFrontend::Success())
        fillMode = CusparseFrontend::GetOutputVariable<cusparseFillMode_t>();
    return fillMode;
}

extern "C" cusparseIndexBase_t CUSPARSEAPI cusparseGetMatIndexBase(cusparseMatDescr_t descrA){
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<long long int>((long long int) descrA);
    CusparseFrontend::Execute("cusparseGetMatIndexBase");
    cusparseIndexBase_t indexBase;
    if(CusparseFrontend::Success())
        indexBase = CusparseFrontend::GetOutputVariable<cusparseIndexBase_t>();
    return indexBase;
}

extern "C" cusparseMatrixType_t CUSPARSEAPI cusparseGetMatType(cusparseMatDescr_t descrA){
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<long long int>((long long int) descrA);
    CusparseFrontend::Execute("cusparseGetMatType");
    cusparseMatrixType_t matrixType;
    if(CusparseFrontend::Success())
        matrixType = CusparseFrontend::GetOutputVariable<cusparseMatrixType_t>();
    return matrixType;
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseSetMatDiagType(cusparseMatDescr_t descrA, cusparseDiagType_t diagType){
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<long long int>((long long int) descrA);
    CusparseFrontend::AddVariableForArguments<cusparseDiagType_t>(diagType);
    CusparseFrontend::Execute("cusparseSetMatDiagType");
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseSetMatFillMode(cusparseMatDescr_t descrA, cusparseFillMode_t fillMode){
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<long long int>((long long int) descrA);
    CusparseFrontend::AddVariableForArguments<cusparseFillMode_t>(fillMode);
    CusparseFrontend::Execute("cusparseSetMatFillMode");
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseSetMatIndexBase(cusparseMatDescr_t descrA, cusparseIndexBase_t indexBase){
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<long long int>((long long int) descrA);
    CusparseFrontend::AddVariableForArguments<cusparseIndexBase_t>(indexBase);
    CusparseFrontend::Execute("cusparseSetMatIndexBase");
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseSetMatType(cusparseMatDescr_t descrA, cusparseMatrixType_t matrixType){
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<long long int>((long long int) descrA);
    CusparseFrontend::AddVariableForArguments<cusparseMatrixType_t>(matrixType);
    CusparseFrontend::Execute("cusparseSetMatType");
    return CusparseFrontend::GetExitCode();
}