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

#include "CusparseHandler.h"

using namespace log4cplus;

using gvirtus::communicators::Buffer;
using gvirtus::communicators::Result;

CUSPARSE_ROUTINE_HANDLER(CreateColorInfo){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("CreateColorInfo"));
    cusparseColorInfo_t info;
    cusparseStatus_t cs = cusparseCreateColorInfo(&info);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->Add<cusparseColorInfo_t>(info);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(cs);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseCreateColorInfo Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(CreateMatDescr){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("CreateMatDescr"));
    cusparseMatDescr_t descrA;
    cusparseStatus_t cs = cusparseCreateMatDescr(&descrA);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->Add<cusparseMatDescr_t>(descrA);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(cs);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseCreateMatDescr Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(DestroyColorInfo){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DestroyColorInfo"));
    cusparseColorInfo_t info = (cusparseColorInfo_t)in->Get<cusparseColorInfo_t>();
    cusparseStatus_t cs = cusparseDestroyColorInfo(info);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    LOG4CPLUS_DEBUG(logger,"cusparseDestroyColorInfo Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(DestroyMatDescr){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DestroyMatDescr"));
    cusparseMatDescr_t descrA = (cusparseMatDescr_t)in->Get<cusparseMatDescr_t>();
    cusparseStatus_t cs = cusparseDestroyMatDescr(descrA);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    LOG4CPLUS_DEBUG(logger,"cusparseDestroyMatDescr Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(GetMatDiagType){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetMatDiagType"));
    cusparseMatDescr_t descrA = (cusparseMatDescr_t)in->Get<cusparseMatDescr_t>();
    cusparseDiagType_t diagType = cusparseGetMatDiagType(descrA);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->Add<cusparseDiagType_t>(diagType);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseGetMatDiagType Executed");
    return std::make_shared<Result>(CUSPARSE_STATUS_SUCCESS, out);
}

CUSPARSE_ROUTINE_HANDLER(GetMatFillMode){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetMatFillMode"));
    cusparseMatDescr_t descrA = (cusparseMatDescr_t)in->Get<cusparseMatDescr_t>();
    cusparseFillMode_t fillMode = cusparseGetMatFillMode(descrA);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->Add<cusparseFillMode_t>(fillMode);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseGetMatFillMode Executed");
    return std::make_shared<Result>(CUSPARSE_STATUS_SUCCESS, out);
}

CUSPARSE_ROUTINE_HANDLER(GetMatIndexBase){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetMatIndexBase"));
    cusparseMatDescr_t descrA = (cusparseMatDescr_t)in->Get<cusparseMatDescr_t>();
    cusparseIndexBase_t indexBase = cusparseGetMatIndexBase(descrA);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->Add<cusparseIndexBase_t>(indexBase);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseGetMatIndexBase Executed");
    return std::make_shared<Result>(CUSPARSE_STATUS_SUCCESS, out);
}

CUSPARSE_ROUTINE_HANDLER(GetMatType){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetMatType"));
    cusparseMatDescr_t descrA = (cusparseMatDescr_t)in->Get<cusparseMatDescr_t>();
    cusparseMatrixType_t matrixType = cusparseGetMatType(descrA);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->Add<cusparseMatrixType_t>(matrixType);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseGetMatType Executed");
    return std::make_shared<Result>(CUSPARSE_STATUS_SUCCESS, out);
}

CUSPARSE_ROUTINE_HANDLER(SetMatDiagType){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetMatDiagType"));
    cusparseMatDescr_t descrA = (cusparseMatDescr_t)in->Get<cusparseMatDescr_t>();
    cusparseDiagType_t diagType = (cusparseDiagType_t)in->Get<cusparseDiagType_t>();
    cusparseStatus_t cs = cusparseSetMatDiagType(descrA, diagType);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    LOG4CPLUS_DEBUG(logger,"cusparseSetMatDiagType Executed");
    return std::make_shared<Result>(cs, out);
}

CUSPARSE_ROUTINE_HANDLER(SetMatFillMode){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetMatFillMode"));
    cusparseMatDescr_t descrA = (cusparseMatDescr_t)in->Get<cusparseMatDescr_t>();
    cusparseFillMode_t fillMode = (cusparseFillMode_t)in->Get<cusparseFillMode_t>();
    cusparseStatus_t cs = cusparseSetMatFillMode(descrA, fillMode);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    LOG4CPLUS_DEBUG(logger,"cusparseSetMatFillMode Executed");
    return std::make_shared<Result>(cs, out);
}

CUSPARSE_ROUTINE_HANDLER(SetMatIndexBase){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetMatIndexBase"));
    cusparseMatDescr_t descrA = (cusparseMatDescr_t)in->Get<cusparseMatDescr_t>();
    cusparseIndexBase_t indexBase = (cusparseIndexBase_t)in->Get<cusparseIndexBase_t>();
    cusparseStatus_t cs = cusparseSetMatIndexBase(descrA, indexBase);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    LOG4CPLUS_DEBUG(logger,"cusparseSetMatIndexBase Executed");
    return std::make_shared<Result>(cs, out);
}

CUSPARSE_ROUTINE_HANDLER(SetMatType){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetMatType"));
    cusparseMatDescr_t descrA = (cusparseMatDescr_t)in->Get<cusparseMatDescr_t>();
    cusparseMatrixType_t matrixType = (cusparseMatrixType_t)in->Get<cusparseMatrixType_t>();
    cusparseStatus_t cs = cusparseSetMatType(descrA, matrixType);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    LOG4CPLUS_DEBUG(logger,"cusparseSetMatType Executed");
    return std::make_shared<Result>(cs, out);
}

CUSPARSE_ROUTINE_HANDLER(CreateCsrsv2Info){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("CreateCsrsv2Info"));
    csrsv2Info_t info;
    cusparseStatus_t cs = cusparseCreateCsrsv2Info(&info);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->Add<csrsv2Info_t>(info);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(cs);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseCreateCsrsv2Info Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(DestroyCsrsv2Info){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DestroyCsrsv2Info"));
    csrsv2Info_t info = (csrsv2Info_t)in->Get<csrsv2Info_t>();
    cusparseStatus_t cs = cusparseDestroyCsrsv2Info(info);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    LOG4CPLUS_DEBUG(logger,"cusparseDestroyCsrsv2Info Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(CreateCsrsm2Info){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("CreateCsrsm2Info"));
    csrsm2Info_t info;
    cusparseStatus_t cs = cusparseCreateCsrsm2Info(&info);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->Add<csrsm2Info_t>(info);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(cs);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseCreateCsrsm2Info Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(DestroyCsrsm2Info){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DestroyCsrsm2Info"));
    csrsm2Info_t info = (csrsm2Info_t)in->Get<csrsm2Info_t>();
    cusparseStatus_t cs = cusparseDestroyCsrsm2Info(info);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    LOG4CPLUS_DEBUG(logger,"cusparseDestroyCsrsm2Info Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(CreateCsric02Info){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("CreateCsric02Info"));
    csric02Info_t info;
    cusparseStatus_t cs = cusparseCreateCsric02Info(&info);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->Add<csric02Info_t>(info);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(cs);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseCreateCsric02Info Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(DestroyCsric02Info){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DestroyCsric02Info"));
    csric02Info_t info = (csric02Info_t)in->Get<csric02Info_t>();
    cusparseStatus_t cs = cusparseDestroyCsric02Info(info);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    LOG4CPLUS_DEBUG(logger,"cusparseDestroyCsric02Info Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(CreateCsrilu02Info){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("CreateCsrilu02Info"));
    csrilu02Info_t info;
    cusparseStatus_t cs = cusparseCreateCsrilu02Info(&info);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->Add<csrilu02Info_t>(info);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(cs);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseCreateCsrilu02Info Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(DestroyCsrilu02Info){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DestroyCsrilu02Info"));
    csrilu02Info_t info = (csrilu02Info_t)in->Get<csrilu02Info_t>();
    cusparseStatus_t cs = cusparseDestroyCsrilu02Info(info);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    LOG4CPLUS_DEBUG(logger,"cusparseDestroyCsrilu02Info Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(CreateBsrsv2Info){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("CreateBsrsv2Info"));
    bsrsv2Info_t info;
    cusparseStatus_t cs = cusparseCreateBsrsv2Info(&info);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->Add<bsrsv2Info_t>(info);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(cs);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseCreateBsrsv2Info Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(DestroyBsrsv2Info){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DestroyBsrsv2Info"));
    bsrsv2Info_t info = (bsrsv2Info_t)in->Get<bsrsv2Info_t>();
    cusparseStatus_t cs = cusparseDestroyBsrsv2Info(info);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    LOG4CPLUS_DEBUG(logger,"cusparseDestroyBsrsv2Info Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(CreateBsrsm2Info){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("CreateBsrsm2Info"));
    bsrsm2Info_t info;
    cusparseStatus_t cs = cusparseCreateBsrsm2Info(&info);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->Add<bsrsm2Info_t>(info);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(cs);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseCreateBsrsm2Info Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(DestroyBsrsm2Info){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DestroyBsrsm2Info"));
    bsrsm2Info_t info = (bsrsm2Info_t)in->Get<bsrsm2Info_t>();
    cusparseStatus_t cs = cusparseDestroyBsrsm2Info(info);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    LOG4CPLUS_DEBUG(logger,"cusparseDestroyBsrsm2Info Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(CreateBsric02Info){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("CreateBsric02Info"));
    bsric02Info_t info;
    cusparseStatus_t cs = cusparseCreateBsric02Info(&info);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->Add<bsric02Info_t>(info);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(cs);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseCreateBsric02Info Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(DestroyBsric02Info){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DestroyBsric02Info"));
    bsric02Info_t info = (bsric02Info_t)in->Get<bsric02Info_t>();
    cusparseStatus_t cs = cusparseDestroyBsric02Info(info);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    LOG4CPLUS_DEBUG(logger,"cusparseDestroyBsric02Info Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(CreateBsrilu02Info){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("CreateBsrilu02Info"));
    bsrilu02Info_t info;
    cusparseStatus_t cs = cusparseCreateBsrilu02Info(&info);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->Add<bsrilu02Info_t>(info);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(cs);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseCreateBsrilu02Info Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(DestroyBsrilu02Info){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DestroyBsrilu02Info"));
    bsrilu02Info_t info = (bsrilu02Info_t)in->Get<bsrilu02Info_t>();
    cusparseStatus_t cs = cusparseDestroyBsrilu02Info(info);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    LOG4CPLUS_DEBUG(logger,"cusparseDestroyBsrilu02Info Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(CreateCsrgemm2Info){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("CreateCsrgemm2Info"));
    csrgemm2Info_t info;
    cusparseStatus_t cs = cusparseCreateCsrgemm2Info(&info);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->Add<csrgemm2Info_t>(info);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(cs);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseCreateCsrgemm2Info Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(DestroyCsrgemm2Info){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DestroyCsrgemm2Info"));
    csrgemm2Info_t info = (csrgemm2Info_t)in->Get<csrgemm2Info_t>();
    cusparseStatus_t cs = cusparseDestroyCsrgemm2Info(info);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    LOG4CPLUS_DEBUG(logger,"cusparseDestroyCsrgemm2Info Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(CreatePruneInfo){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("CreatePruneInfo"));
    pruneInfo_t info;
    cusparseStatus_t cs = cusparseCreatePruneInfo(&info);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->Add<pruneInfo_t>(info);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(cs);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseCreatePruneInfo Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(DestroyPruneInfo){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DestroyPruneInfo"));
    pruneInfo_t info = (pruneInfo_t)in->Get<pruneInfo_t>();
    cusparseStatus_t cs = cusparseDestroyPruneInfo(info);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    LOG4CPLUS_DEBUG(logger,"cusparseDestroyPruneInfo Executed");
    return std::make_shared<Result>(cs,out);
}

#ifndef CUSPARSE_VERSION
#error CUSPARSE_VERSION not defined
#endif
