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

CUSPARSE_ROUTINE_HANDLER(Scsric02_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Scsric02_bufferSize"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int nnz = in->Get<int>();
    cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    float * csrVal = in->GetFromMarshal<float*>();
    const int * csrRowPtr = in->GetFromMarshal<int*>();
    const int * csrColInd = in->GetFromMarshal<int*>();
    csric02Info_t info = (csric02Info_t)in->Get<size_t>();
    int * pBufferSize = new int;
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseScsric02_bufferSize(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, info, pBufferSize);
        out->Add<csric02Info_t>(info);
        out->AddMarshal<int*>(pBufferSize);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseScsric02_bufferSize Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Dcsric02_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Dcsric02_bufferSize"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int nnz = in->Get<int>();
    cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    double * csrVal = in->GetFromMarshal<double*>();
    const int * csrRowPtr = in->GetFromMarshal<int*>();
    const int * csrColInd = in->GetFromMarshal<int*>();
    csric02Info_t info = (csric02Info_t)in->Get<size_t>();
    int * pBufferSize = new int;
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseDcsric02_bufferSize(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, info, pBufferSize);
        out->Add<csric02Info_t>(info);
        out->AddMarshal<int*>(pBufferSize);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseDcsric02_bufferSize Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Ccsric02_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Ccsric02_bufferSize"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int nnz = in->Get<int>();
    cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    cuComplex * csrVal = in->GetFromMarshal<cuComplex*>();
    const int * csrRowPtr = in->GetFromMarshal<int*>();
    const int * csrColInd = in->GetFromMarshal<int*>();
    csric02Info_t info = (csric02Info_t)in->Get<size_t>();
    int * pBufferSize = new int;
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseCcsric02_bufferSize(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, info, pBufferSize);
        out->Add<csric02Info_t>(info);
        out->AddMarshal<int*>(pBufferSize);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseCcsric02_bufferSize Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Zcsric02_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Zcsric02_bufferSize"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int nnz = in->Get<int>();
    cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    cuDoubleComplex * csrVal = in->GetFromMarshal<cuDoubleComplex*>();
    const int * csrRowPtr = in->GetFromMarshal<int*>();
    const int * csrColInd = in->GetFromMarshal<int*>();
    csric02Info_t info = (csric02Info_t)in->Get<size_t>();
    int * pBufferSize = new int;
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseZcsric02_bufferSize(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, info, pBufferSize);
        out->Add<csric02Info_t>(info);
        out->AddMarshal<int*>(pBufferSize);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseZcsric02_bufferSize Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Scsric02_analysis){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Scsric02_analysis"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    const int m = in->Get<int>();
    const int nnz = in->Get<int>();
    cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    float * csrVal = in->GetFromMarshal<float*>();
    const int * csrRowPtr = in->GetFromMarshal<int*>();
    const int * csrColInd = in->GetFromMarshal<int*>();
    csric02Info_t info = (csric02Info_t)in->Get<size_t>();
    cusparseSolvePolicy_t policy = in->Get<cusparseSolvePolicy_t>();
    void * pBuffer = in->GetFromMarshal<void*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseScsric02_analysis(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, info, policy, pBuffer);
        out->AddMarshal<csric02Info_t>(info);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseScsric02_analysis Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Dcsric02_analysis){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Dcsric02_analysis"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    const int m = in->Get<int>();
    const int nnz = in->Get<int>();
    cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    double * csrVal = in->GetFromMarshal<double*>();
    const int * csrRowPtr = in->GetFromMarshal<int*>();
    const int * csrColInd = in->GetFromMarshal<int*>();
    csric02Info_t info = (csric02Info_t)in->Get<size_t>();
    cusparseSolvePolicy_t policy = in->Get<cusparseSolvePolicy_t>();
    void * pBuffer = in->GetFromMarshal<void*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseDcsric02_analysis(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, info, policy, pBuffer);
        out->AddMarshal<csric02Info_t>(info);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseDcsric02_analysis Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Ccsric02_analysis){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Ccsric02_analysis"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    const int m = in->Get<int>();
    const int nnz = in->Get<int>();
    cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    cuComplex * csrVal = in->GetFromMarshal<cuComplex*>();
    const int * csrRowPtr = in->GetFromMarshal<int*>();
    const int * csrColInd = in->GetFromMarshal<int*>();
    csric02Info_t info = (csric02Info_t)in->Get<size_t>();
    cusparseSolvePolicy_t policy = in->Get<cusparseSolvePolicy_t>();
    void * pBuffer = in->GetFromMarshal<void*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseCcsric02_analysis(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, info, policy, pBuffer);
        out->AddMarshal<csric02Info_t>(info);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseCcsric02_analysis Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Zcsric02_analysis){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Zcsric02_analysis"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    const int m = in->Get<int>();
    const int nnz = in->Get<int>();
    cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    cuDoubleComplex * csrVal = in->GetFromMarshal<cuDoubleComplex*>();
    const int * csrRowPtr = in->GetFromMarshal<int*>();
    const int * csrColInd = in->GetFromMarshal<int*>();
    csric02Info_t info = (csric02Info_t)in->Get<size_t>();
    cusparseSolvePolicy_t policy = in->Get<cusparseSolvePolicy_t>();
    void * pBuffer = in->GetFromMarshal<void*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseZcsric02_analysis(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, info, policy, pBuffer);
        out->AddMarshal<csric02Info_t>(info);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseZcsric02_analysis Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Scsric02){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Scsric02"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    const int m = in->Get<int>();
    const int nnz = in->Get<int>();
    cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    float * csrVal = in->GetFromMarshal<float*>();
    const int * csrRowPtr = in->GetFromMarshal<int*>();
    const int * csrColInd = in->GetFromMarshal<int*>();
    csric02Info_t info = (csric02Info_t)in->Get<size_t>();
    cusparseSolvePolicy_t policy = in->Get<cusparseSolvePolicy_t>();
    void * pBuffer = in->GetFromMarshal<void*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseScsric02(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, info, policy, pBuffer);
        out->AddMarshal<float*>(csrVal);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseScsric02 Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Dcsric02){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Dcsric02"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    const int m = in->Get<int>();
    const int nnz = in->Get<int>();
    cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    double * csrVal = in->GetFromMarshal<double*>();
    const int * csrRowPtr = in->GetFromMarshal<int*>();
    const int * csrColInd = in->GetFromMarshal<int*>();
    csric02Info_t info = (csric02Info_t)in->Get<size_t>();
    cusparseSolvePolicy_t policy = in->Get<cusparseSolvePolicy_t>();
    void * pBuffer = in->GetFromMarshal<void*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseDcsric02(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, info, policy, pBuffer);
        out->AddMarshal<double*>(csrVal);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseDcsric02 Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Ccsric02){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Ccsric02"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    const int m = in->Get<int>();
    const int nnz = in->Get<int>();
    cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    cuComplex * csrVal = in->GetFromMarshal<cuComplex*>();
    const int * csrRowPtr = in->GetFromMarshal<int*>();
    const int * csrColInd = in->GetFromMarshal<int*>();
    csric02Info_t info = (csric02Info_t)in->Get<size_t>();
    cusparseSolvePolicy_t policy = in->Get<cusparseSolvePolicy_t>();
    void * pBuffer = in->GetFromMarshal<void*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseCcsric02(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, info, policy, pBuffer);
        out->AddMarshal<cuComplex*>(csrVal);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseCcsric02 Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Zcsric02){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Zcsric02"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    const int m = in->Get<int>();
    const int nnz = in->Get<int>();
    cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    cuDoubleComplex * csrVal = in->GetFromMarshal<cuDoubleComplex*>();
    const int * csrRowPtr = in->GetFromMarshal<int*>();
    const int * csrColInd = in->GetFromMarshal<int*>();
    csric02Info_t info = (csric02Info_t)in->Get<size_t>();
    cusparseSolvePolicy_t policy = in->Get<cusparseSolvePolicy_t>();
    void * pBuffer = in->GetFromMarshal<void*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseZcsric02(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, info, policy, pBuffer);
        out->AddMarshal<cuDoubleComplex*>(csrVal);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseZcsric02 Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Xcsric02_zeroPivot){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Xcsric02_zeroPivot"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    csric02Info_t info = (csric02Info_t)in->Get<size_t>();
    int * position = new int;
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseXcsric02_zeroPivot(handle, info, position);
        out->AddMarshal<int*>(position);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseXcsric02_zeroPivot Executed");
    return std::make_shared<Result>(cs,out);
}

#ifndef CUSPARSE_VERSION
#error CUSPARSE_VERSION not defined
#endif
