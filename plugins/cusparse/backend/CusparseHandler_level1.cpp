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

CUSPARSE_ROUTINE_HANDLER(Saxpyi){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Saxpyi"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<long long int>();
    int nnz = in->Get<int>();
    const float * alpha = in->Assign<float>();
    const float * xVal = in->GetFromMarshal<float*>();
    const int * xInd = in->GetFromMarshal<int*>();
    float * y = in->GetFromMarshal<float*>();
    cusparseIndexBase_t idxBase = in->Get<cusparseIndexBase_t>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseSaxpyi(handle, nnz, alpha, xVal, xInd, y, idxBase);
        out->AddMarshal<float*>(y);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseSaxpyi Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Daxpyi){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Daxpyi"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<long long int>();
    int nnz = in->Get<int>();
    const double * alpha = in->Assign<double>();
    const double * xVal = in->GetFromMarshal<double*>();
    const int * xInd = in->GetFromMarshal<int*>();
    double * y = in->GetFromMarshal<double*>();
    cusparseIndexBase_t idxBase = in->Get<cusparseIndexBase_t>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseDaxpyi(handle, nnz, alpha, xVal, xInd, y, idxBase);
        out->AddMarshal<double*>(y);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseDaxpyi Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Caxpyi){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Caxpyi"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<long long int>();
    int nnz = in->Get<int>();
    const cuComplex * alpha = in->Assign<cuComplex>();
    const cuComplex * xVal = in->GetFromMarshal<cuComplex*>();
    const int * xInd = in->GetFromMarshal<int*>();
    cuComplex * y = in->GetFromMarshal<cuComplex*>();
    cusparseIndexBase_t idxBase = in->Get<cusparseIndexBase_t>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseCaxpyi(handle, nnz, alpha, xVal, xInd, y, idxBase);
        out->AddMarshal<cuComplex*>(y);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseCaxpyi Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Zaxpyi){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Zaxpyi"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<long long int>();
    int nnz = in->Get<int>();
    const cuDoubleComplex * alpha = in->Assign<cuDoubleComplex>();
    const cuDoubleComplex * xVal = in->GetFromMarshal<cuDoubleComplex*>();
    const int * xInd = in->GetFromMarshal<int*>();
    cuDoubleComplex * y = in->GetFromMarshal<cuDoubleComplex*>();
    cusparseIndexBase_t idxBase = in->Get<cusparseIndexBase_t>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseZaxpyi(handle, nnz, alpha, xVal, xInd, y, idxBase);
        out->AddMarshal<cuDoubleComplex*>(y);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseZaxpyi Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Sgthr){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Sgthr"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<long long int>();
    int nnz = in->Get<int>();
    const float * y = in->GetFromMarshal<float*>();
    float * xVal = in->GetFromMarshal<float*>();
    const int * xInd = in->GetFromMarshal<int*>();
    cusparseIndexBase_t idxBase = in->Get<cusparseIndexBase_t>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseSgthr(handle, nnz, y, xVal, xInd, idxBase);
        out->AddMarshal<float*>(xVal);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseSgthr Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Dgthr){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Dgthr"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<long long int>();
    int nnz = in->Get<int>();
    const double * y = in->GetFromMarshal<double*>();
    double * xVal = in->GetFromMarshal<double*>();
    const int * xInd = in->GetFromMarshal<int*>();
    cusparseIndexBase_t idxBase = in->Get<cusparseIndexBase_t>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseDgthr(handle, nnz, y, xVal, xInd, idxBase);
        out->AddMarshal<double*>(xVal);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseDgthr Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Cgthr){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Cgthr"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<long long int>();
    int nnz = in->Get<int>();
    const cuComplex * y = in->GetFromMarshal<cuComplex*>();
    cuComplex * xVal = in->GetFromMarshal<cuComplex*>();
    const int * xInd = in->GetFromMarshal<int*>();
    cusparseIndexBase_t idxBase = in->Get<cusparseIndexBase_t>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseCgthr(handle, nnz, y, xVal, xInd, idxBase);
        out->AddMarshal<cuComplex*>(xVal);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseCgthr Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Zgthr){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Zgthr"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<long long int>();
    int nnz = in->Get<int>();
    const cuDoubleComplex * y = in->GetFromMarshal<cuDoubleComplex*>();
    cuDoubleComplex * xVal = in->GetFromMarshal<cuDoubleComplex*>();
    const int * xInd = in->GetFromMarshal<int*>();
    cusparseIndexBase_t idxBase = in->Get<cusparseIndexBase_t>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseZgthr(handle, nnz, y, xVal, xInd, idxBase);
        out->AddMarshal<cuDoubleComplex*>(xVal);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseZgthr Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Sgthrz){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Sgthrz"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<long long int>();
    int nnz = in->Get<int>();
    float * y = in->GetFromMarshal<float*>();
    float * xVal = in->GetFromMarshal<float*>();
    const int * xInd = in->GetFromMarshal<int*>();
    cusparseIndexBase_t idxBase = in->Get<cusparseIndexBase_t>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseSgthrz(handle, nnz, y, xVal, xInd, idxBase);
        out->AddMarshal<float*>(xVal);
        out->AddMarshal<float*>(y);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseSgthrz Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Dgthrz){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Dgthrz"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<long long int>();
    int nnz = in->Get<int>();
    double * y = in->GetFromMarshal<double*>();
    double * xVal = in->GetFromMarshal<double*>();
    const int * xInd = in->GetFromMarshal<int*>();
    cusparseIndexBase_t idxBase = in->Get<cusparseIndexBase_t>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseDgthrz(handle, nnz, y, xVal, xInd, idxBase);
        out->AddMarshal<double*>(xVal);
        out->AddMarshal<double*>(y);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseDgthrz Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Cgthrz){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Cgthrz"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<long long int>();
    int nnz = in->Get<int>();
    cuComplex * y = in->GetFromMarshal<cuComplex*>();
    cuComplex * xVal = in->GetFromMarshal<cuComplex*>();
    const int * xInd = in->GetFromMarshal<int*>();
    cusparseIndexBase_t idxBase = in->Get<cusparseIndexBase_t>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseCgthrz(handle, nnz, y, xVal, xInd, idxBase);
        out->AddMarshal<cuComplex*>(xVal);
        out->AddMarshal<cuComplex*>(y);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseCgthrz Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Zgthrz){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Zgthrz"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<long long int>();
    int nnz = in->Get<int>();
    cuDoubleComplex * y = in->GetFromMarshal<cuDoubleComplex*>();
    cuDoubleComplex * xVal = in->GetFromMarshal<cuDoubleComplex*>();
    const int * xInd = in->GetFromMarshal<int*>();
    cusparseIndexBase_t idxBase = in->Get<cusparseIndexBase_t>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseZgthrz(handle, nnz, y, xVal, xInd, idxBase);
        out->AddMarshal<cuDoubleComplex*>(xVal);
        out->AddMarshal<cuDoubleComplex*>(y);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseZgthrz Executed");
    return std::make_shared<Result>(cs,out);
}

#ifndef CUSPARSE_VERSION
#error CUSPARSE_VERSION not defined
#endif
