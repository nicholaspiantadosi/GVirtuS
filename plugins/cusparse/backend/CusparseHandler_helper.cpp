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

#ifndef CUSPARSE_VERSION
#error CUSPARSE_VERSION not defined
#endif
