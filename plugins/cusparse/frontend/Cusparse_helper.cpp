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

extern "C" cusparseStatus_t CUSPARSEAPI cusparseCreateCsrsv2Info(csrsv2Info_t * info){
    CusparseFrontend::Prepare();
    CusparseFrontend::AddHostPointerForArguments<csrsv2Info_t>(info);
    CusparseFrontend::Execute("cusparseCreateCsrsv2Info");
    if(CusparseFrontend::Success())
        *info = CusparseFrontend::GetOutputVariable<csrsv2Info_t>();
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseDestroyCsrsv2Info(csrsv2Info_t info){
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<long long int>((long long int) info);
    CusparseFrontend::Execute("cusparseDestroyCsrsv2Info");
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseCreateCsrsm2Info(csrsm2Info_t * info){
    CusparseFrontend::Prepare();
    CusparseFrontend::AddHostPointerForArguments<csrsm2Info_t>(info);
    CusparseFrontend::Execute("cusparseCreateCsrsm2Info");
    if(CusparseFrontend::Success())
        *info = CusparseFrontend::GetOutputVariable<csrsm2Info_t>();
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseDestroyCsrsm2Info(csrsm2Info_t info){
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<long long int>((long long int) info);
    CusparseFrontend::Execute("cusparseDestroyCsrsm2Info");
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseCreateCsric02Info(csric02Info_t * info){
    CusparseFrontend::Prepare();
    CusparseFrontend::AddHostPointerForArguments<csric02Info_t>(info);
    CusparseFrontend::Execute("cusparseCreateCsric02Info");
    if(CusparseFrontend::Success())
        *info = CusparseFrontend::GetOutputVariable<csric02Info_t>();
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseDestroyCsric02Info(csric02Info_t info){
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<long long int>((long long int) info);
    CusparseFrontend::Execute("cusparseDestroyCsric02Info");
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseCreateCsrilu02Info(csrilu02Info_t * info){
    CusparseFrontend::Prepare();
    CusparseFrontend::AddHostPointerForArguments<csrilu02Info_t>(info);
    CusparseFrontend::Execute("cusparseCreateCsrilu02Info");
    if(CusparseFrontend::Success())
        *info = CusparseFrontend::GetOutputVariable<csrilu02Info_t>();
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseDestroyCsrilu02Info(csrilu02Info_t info){
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<long long int>((long long int) info);
    CusparseFrontend::Execute("cusparseDestroyCsrilu02Info");
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseCreateBsrsv2Info(bsrsv2Info_t * info){
    CusparseFrontend::Prepare();
    CusparseFrontend::AddHostPointerForArguments<bsrsv2Info_t>(info);
    CusparseFrontend::Execute("cusparseCreateBsrsv2Info");
    if(CusparseFrontend::Success())
        *info = CusparseFrontend::GetOutputVariable<bsrsv2Info_t>();
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseDestroyBsrsv2Info(bsrsv2Info_t info){
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<long long int>((long long int) info);
    CusparseFrontend::Execute("cusparseDestroyBsrsv2Info");
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseCreateBsrsm2Info(bsrsm2Info_t * info){
    CusparseFrontend::Prepare();
    CusparseFrontend::AddHostPointerForArguments<bsrsm2Info_t>(info);
    CusparseFrontend::Execute("cusparseCreateBsrsm2Info");
    if(CusparseFrontend::Success())
        *info = CusparseFrontend::GetOutputVariable<bsrsm2Info_t>();
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseDestroyBsrsm2Info(bsrsm2Info_t info){
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<long long int>((long long int) info);
    CusparseFrontend::Execute("cusparseDestroyBsrsm2Info");
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseCreateBsric02Info(bsric02Info_t * info){
    CusparseFrontend::Prepare();
    CusparseFrontend::AddHostPointerForArguments<bsric02Info_t>(info);
    CusparseFrontend::Execute("cusparseCreateBsric02Info");
    if(CusparseFrontend::Success())
        *info = CusparseFrontend::GetOutputVariable<bsric02Info_t>();
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseDestroyBsric02Info(bsric02Info_t info){
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<long long int>((long long int) info);
    CusparseFrontend::Execute("cusparseDestroyBsric02Info");
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseCreateBsrilu02Info(bsrilu02Info_t * info){
    CusparseFrontend::Prepare();
    CusparseFrontend::AddHostPointerForArguments<bsrilu02Info_t>(info);
    CusparseFrontend::Execute("cusparseCreateBsrilu02Info");
    if(CusparseFrontend::Success())
        *info = CusparseFrontend::GetOutputVariable<bsrilu02Info_t>();
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseDestroyBsrilu02Info(bsrilu02Info_t info){
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<long long int>((long long int) info);
    CusparseFrontend::Execute("cusparseDestroyBsrilu02Info");
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseCreateCsrgemm2Info(csrgemm2Info_t * info){
    CusparseFrontend::Prepare();
    CusparseFrontend::AddHostPointerForArguments<csrgemm2Info_t>(info);
    CusparseFrontend::Execute("cusparseCreateCsrgemm2Info");
    if(CusparseFrontend::Success())
        *info = CusparseFrontend::GetOutputVariable<csrgemm2Info_t>();
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseDestroyCsrgemm2Info(csrgemm2Info_t info){
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<long long int>((long long int) info);
    CusparseFrontend::Execute("cusparseDestroyCsrgemm2Info");
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseCreatePruneInfo(pruneInfo_t * info){
    CusparseFrontend::Prepare();
    CusparseFrontend::AddHostPointerForArguments<pruneInfo_t>(info);
    CusparseFrontend::Execute("cusparseCreatePruneInfo");
    if(CusparseFrontend::Success())
        *info = CusparseFrontend::GetOutputVariable<pruneInfo_t>();
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseDestroyPruneInfo(pruneInfo_t info){
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<long long int>((long long int) info);
    CusparseFrontend::Execute("cusparseDestroyPruneInfo");
    return CusparseFrontend::GetExitCode();
}
