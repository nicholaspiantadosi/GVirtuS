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

CUSPARSE_ROUTINE_HANDLER(Sbsric02_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Sbsric02_bufferSize"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    cusparseDirection_t dirA = in->Get<cusparseDirection_t>();
    int mb = in->Get<int>();
    int nnzb = in->Get<int>();
    cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    float * bsrVal = in->GetFromMarshal<float*>();
    const int * bsrRowPtr = in->GetFromMarshal<int*>();
    const int * bsrColInd = in->GetFromMarshal<int*>();
    int blockDim = in->Get<int>();
    bsric02Info_t info = (bsric02Info_t)in->Get<size_t>();
    int * pBufferSize = new int;
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseSbsric02_bufferSize(handle, dirA, mb, nnzb, descrA, bsrVal, bsrRowPtr, bsrColInd, blockDim, info, pBufferSize);
        out->Add<bsric02Info_t>(info);
        out->AddMarshal<int*>(pBufferSize);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseSbsric02_bufferSize Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Dbsric02_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Dbsric02_bufferSize"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    cusparseDirection_t dirA = in->Get<cusparseDirection_t>();
    int mb = in->Get<int>();
    int nnzb = in->Get<int>();
    cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    double * bsrVal = in->GetFromMarshal<double*>();
    const int * bsrRowPtr = in->GetFromMarshal<int*>();
    const int * bsrColInd = in->GetFromMarshal<int*>();
    int blockDim = in->Get<int>();
    bsric02Info_t info = (bsric02Info_t)in->Get<size_t>();
    int * pBufferSize = new int;
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseDbsric02_bufferSize(handle, dirA, mb, nnzb, descrA, bsrVal, bsrRowPtr, bsrColInd, blockDim, info, pBufferSize);
        out->Add<bsric02Info_t>(info);
        out->AddMarshal<int*>(pBufferSize);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseDbsric02_bufferSize Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Cbsric02_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Cbsric02_bufferSize"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    cusparseDirection_t dirA = in->Get<cusparseDirection_t>();
    int mb = in->Get<int>();
    int nnzb = in->Get<int>();
    cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    cuComplex * bsrVal = in->GetFromMarshal<cuComplex*>();
    const int * bsrRowPtr = in->GetFromMarshal<int*>();
    const int * bsrColInd = in->GetFromMarshal<int*>();
    int blockDim = in->Get<int>();
    bsric02Info_t info = (bsric02Info_t)in->Get<size_t>();
    int * pBufferSize = new int;
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseCbsric02_bufferSize(handle, dirA, mb, nnzb, descrA, bsrVal, bsrRowPtr, bsrColInd, blockDim, info, pBufferSize);
        out->Add<bsric02Info_t>(info);
        out->AddMarshal<int*>(pBufferSize);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseCbsric02_bufferSize Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Zbsric02_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Zbsric02_bufferSize"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    cusparseDirection_t dirA = in->Get<cusparseDirection_t>();
    int mb = in->Get<int>();
    int nnzb = in->Get<int>();
    cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    cuDoubleComplex * bsrVal = in->GetFromMarshal<cuDoubleComplex*>();
    const int * bsrRowPtr = in->GetFromMarshal<int*>();
    const int * bsrColInd = in->GetFromMarshal<int*>();
    int blockDim = in->Get<int>();
    bsric02Info_t info = (bsric02Info_t)in->Get<size_t>();
    int * pBufferSize = new int;
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseZbsric02_bufferSize(handle, dirA, mb, nnzb, descrA, bsrVal, bsrRowPtr, bsrColInd, blockDim, info, pBufferSize);
        out->Add<bsric02Info_t>(info);
        out->AddMarshal<int*>(pBufferSize);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseZbsric02_bufferSize Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Sbsric02_analysis){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Sbsric02_analysis"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    cusparseDirection_t dirA = in->Get<cusparseDirection_t>();
    int mb = in->Get<int>();
    int nnzb = in->Get<int>();
    cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    float * bsrVal = in->GetFromMarshal<float*>();
    const int * bsrRowPtr = in->GetFromMarshal<int*>();
    const int * bsrColInd = in->GetFromMarshal<int*>();
    int blockDim = in->Get<int>();
    bsric02Info_t info = (bsric02Info_t)in->Get<size_t>();
    cusparseSolvePolicy_t policy = in->Get<cusparseSolvePolicy_t>();
    void * pBuffer = in->GetFromMarshal<void*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseSbsric02_analysis(handle, dirA, mb, nnzb, descrA, bsrVal, bsrRowPtr, bsrColInd, blockDim, info, policy, pBuffer);
        out->AddMarshal<bsric02Info_t>(info);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseSbsric02_analysis Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Dbsric02_analysis){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Dbsric02_analysis"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    cusparseDirection_t dirA = in->Get<cusparseDirection_t>();
    int mb = in->Get<int>();
    int nnzb = in->Get<int>();
    cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    double * bsrVal = in->GetFromMarshal<double*>();
    const int * bsrRowPtr = in->GetFromMarshal<int*>();
    const int * bsrColInd = in->GetFromMarshal<int*>();
    int blockDim = in->Get<int>();
    bsric02Info_t info = (bsric02Info_t)in->Get<size_t>();
    cusparseSolvePolicy_t policy = in->Get<cusparseSolvePolicy_t>();
    void * pBuffer = in->GetFromMarshal<void*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseDbsric02_analysis(handle, dirA, mb, nnzb, descrA, bsrVal, bsrRowPtr, bsrColInd, blockDim, info, policy, pBuffer);
        out->AddMarshal<bsric02Info_t>(info);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseDbsric02_analysis Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Cbsric02_analysis){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Cbsric02_analysis"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    cusparseDirection_t dirA = in->Get<cusparseDirection_t>();
    int mb = in->Get<int>();
    int nnzb = in->Get<int>();
    cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    cuComplex * bsrVal = in->GetFromMarshal<cuComplex*>();
    const int * bsrRowPtr = in->GetFromMarshal<int*>();
    const int * bsrColInd = in->GetFromMarshal<int*>();
    int blockDim = in->Get<int>();
    bsric02Info_t info = (bsric02Info_t)in->Get<size_t>();
    cusparseSolvePolicy_t policy = in->Get<cusparseSolvePolicy_t>();
    void * pBuffer = in->GetFromMarshal<void*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseCbsric02_analysis(handle, dirA, mb, nnzb, descrA, bsrVal, bsrRowPtr, bsrColInd, blockDim, info, policy, pBuffer);
        out->AddMarshal<bsric02Info_t>(info);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseCbsric02_analysis Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Zbsric02_analysis){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Zbsric02_analysis"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    cusparseDirection_t dirA = in->Get<cusparseDirection_t>();
    int mb = in->Get<int>();
    int nnzb = in->Get<int>();
    cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    cuDoubleComplex * bsrVal = in->GetFromMarshal<cuDoubleComplex*>();
    const int * bsrRowPtr = in->GetFromMarshal<int*>();
    const int * bsrColInd = in->GetFromMarshal<int*>();
    int blockDim = in->Get<int>();
    bsric02Info_t info = (bsric02Info_t)in->Get<size_t>();
    cusparseSolvePolicy_t policy = in->Get<cusparseSolvePolicy_t>();
    void * pBuffer = in->GetFromMarshal<void*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseZbsric02_analysis(handle, dirA, mb, nnzb, descrA, bsrVal, bsrRowPtr, bsrColInd, blockDim, info, policy, pBuffer);
        out->AddMarshal<bsric02Info_t>(info);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseZbsric02_analysis Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Sbsric02){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Sbsric02"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    cusparseDirection_t dirA = in->Get<cusparseDirection_t>();
    int mb = in->Get<int>();
    int nnzb = in->Get<int>();
    cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    float * bsrVal = in->GetFromMarshal<float*>();
    const int * bsrRowPtr = in->GetFromMarshal<int*>();
    const int * bsrColInd = in->GetFromMarshal<int*>();
    int blockDim = in->Get<int>();
    bsric02Info_t info = (bsric02Info_t)in->Get<size_t>();
    cusparseSolvePolicy_t policy = in->Get<cusparseSolvePolicy_t>();
    void * pBuffer = in->GetFromMarshal<void*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseSbsric02(handle, dirA, mb, nnzb, descrA, bsrVal, bsrRowPtr, bsrColInd, blockDim, info, policy, pBuffer);
        out->AddMarshal<float*>(bsrVal);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseSbsric02 Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Dbsric02){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Dbsric02"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    cusparseDirection_t dirA = in->Get<cusparseDirection_t>();
    int mb = in->Get<int>();
    int nnzb = in->Get<int>();
    cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    double * bsrVal = in->GetFromMarshal<double*>();
    const int * bsrRowPtr = in->GetFromMarshal<int*>();
    const int * bsrColInd = in->GetFromMarshal<int*>();
    int blockDim = in->Get<int>();
    bsric02Info_t info = (bsric02Info_t)in->Get<size_t>();
    cusparseSolvePolicy_t policy = in->Get<cusparseSolvePolicy_t>();
    void * pBuffer = in->GetFromMarshal<void*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseDbsric02(handle, dirA, mb, nnzb, descrA, bsrVal, bsrRowPtr, bsrColInd, blockDim, info, policy, pBuffer);
        out->AddMarshal<double*>(bsrVal);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseDbsric02 Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Cbsric02){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Cbsric02"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    cusparseDirection_t dirA = in->Get<cusparseDirection_t>();
    int mb = in->Get<int>();
    int nnzb = in->Get<int>();
    cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    cuComplex * bsrVal = in->GetFromMarshal<cuComplex*>();
    const int * bsrRowPtr = in->GetFromMarshal<int*>();
    const int * bsrColInd = in->GetFromMarshal<int*>();
    int blockDim = in->Get<int>();
    bsric02Info_t info = (bsric02Info_t)in->Get<size_t>();
    cusparseSolvePolicy_t policy = in->Get<cusparseSolvePolicy_t>();
    void * pBuffer = in->GetFromMarshal<void*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseCbsric02(handle, dirA, mb, nnzb, descrA, bsrVal, bsrRowPtr, bsrColInd, blockDim, info, policy, pBuffer);
        out->AddMarshal<cuComplex*>(bsrVal);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseCbsric02 Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Zbsric02){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Zbsric02"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    cusparseDirection_t dirA = in->Get<cusparseDirection_t>();
    int mb = in->Get<int>();
    int nnzb = in->Get<int>();
    cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    cuDoubleComplex * bsrVal = in->GetFromMarshal<cuDoubleComplex*>();
    const int * bsrRowPtr = in->GetFromMarshal<int*>();
    const int * bsrColInd = in->GetFromMarshal<int*>();
    int blockDim = in->Get<int>();
    bsric02Info_t info = (bsric02Info_t)in->Get<size_t>();
    cusparseSolvePolicy_t policy = in->Get<cusparseSolvePolicy_t>();
    void * pBuffer = in->GetFromMarshal<void*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseZbsric02(handle, dirA, mb, nnzb, descrA, bsrVal, bsrRowPtr, bsrColInd, blockDim, info, policy, pBuffer);
        out->AddMarshal<cuDoubleComplex*>(bsrVal);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseZbsric02 Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Xbsric02_zeroPivot){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Xbsric02_zeroPivot"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    bsric02Info_t info = (bsric02Info_t)in->Get<size_t>();
    int * position = new int;
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseXbsric02_zeroPivot(handle, info, position);
        out->AddMarshal<int*>(position);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseXbsric02_zeroPivot Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Scsrilu02_numericBoost){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Scsrilu02_numericBoost"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    csrilu02Info_t info = (csrilu02Info_t)in->Get<size_t>();
    int enable_boost = in->Get<int>();
    double * tol = in->GetFromMarshal<double*>();
    float * boost_val = in->GetFromMarshal<float*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseScsrilu02_numericBoost(handle, info, enable_boost, tol, boost_val);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseScsrilu02_numericBoost Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Dcsrilu02_numericBoost){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Dcsrilu02_numericBoost"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    csrilu02Info_t info = (csrilu02Info_t)in->Get<size_t>();
    int enable_boost = in->Get<int>();
    double * tol = in->GetFromMarshal<double*>();
    double * boost_val = in->GetFromMarshal<double*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseDcsrilu02_numericBoost(handle, info, enable_boost, tol, boost_val);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseDcsrilu02_numericBoost Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Ccsrilu02_numericBoost){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Ccsrilu02_numericBoost"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    csrilu02Info_t info = (csrilu02Info_t)in->Get<size_t>();
    int enable_boost = in->Get<int>();
    double * tol = in->GetFromMarshal<double*>();
    cuComplex * boost_val = in->GetFromMarshal<cuComplex*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseCcsrilu02_numericBoost(handle, info, enable_boost, tol, boost_val);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseCcsrilu02_numericBoost Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Zcsrilu02_numericBoost){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Zcsrilu02_numericBoost"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    csrilu02Info_t info = (csrilu02Info_t)in->Get<size_t>();
    int enable_boost = in->Get<int>();
    double * tol = in->GetFromMarshal<double*>();
    cuDoubleComplex * boost_val = in->GetFromMarshal<cuDoubleComplex*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseZcsrilu02_numericBoost(handle, info, enable_boost, tol, boost_val);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseZcsrilu02_numericBoost Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Scsrilu02_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Scsrilu02_bufferSize"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int nnz = in->Get<int>();
    cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    float * csrVal = in->GetFromMarshal<float*>();
    const int * csrRowPtr = in->GetFromMarshal<int*>();
    const int * csrColInd = in->GetFromMarshal<int*>();
    csrilu02Info_t info = (csrilu02Info_t)in->Get<size_t>();
    int * pBufferSize = new int;
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseScsrilu02_bufferSize(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, info, pBufferSize);
        out->Add<csrilu02Info_t>(info);
        out->AddMarshal<int*>(pBufferSize);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseScsrilu02_bufferSize Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Dcsrilu02_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Dcsrilu02_bufferSize"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int nnz = in->Get<int>();
    cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    double * csrVal = in->GetFromMarshal<double*>();
    const int * csrRowPtr = in->GetFromMarshal<int*>();
    const int * csrColInd = in->GetFromMarshal<int*>();
    csrilu02Info_t info = (csrilu02Info_t)in->Get<size_t>();
    int * pBufferSize = new int;
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseDcsrilu02_bufferSize(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, info, pBufferSize);
        out->Add<csrilu02Info_t>(info);
        out->AddMarshal<int*>(pBufferSize);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseDcsrilu02_bufferSize Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Ccsrilu02_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Ccsrilu02_bufferSize"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int nnz = in->Get<int>();
    cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    cuComplex * csrVal = in->GetFromMarshal<cuComplex*>();
    const int * csrRowPtr = in->GetFromMarshal<int*>();
    const int * csrColInd = in->GetFromMarshal<int*>();
    csrilu02Info_t info = (csrilu02Info_t)in->Get<size_t>();
    int * pBufferSize = new int;
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseCcsrilu02_bufferSize(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, info, pBufferSize);
        out->Add<csrilu02Info_t>(info);
        out->AddMarshal<int*>(pBufferSize);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseCcsrilu02_bufferSize Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Zcsrilu02_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Zcsrilu02_bufferSize"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int nnz = in->Get<int>();
    cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    cuDoubleComplex * csrVal = in->GetFromMarshal<cuDoubleComplex*>();
    const int * csrRowPtr = in->GetFromMarshal<int*>();
    const int * csrColInd = in->GetFromMarshal<int*>();
    csrilu02Info_t info = (csrilu02Info_t)in->Get<size_t>();
    int * pBufferSize = new int;
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseZcsrilu02_bufferSize(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, info, pBufferSize);
        out->Add<csrilu02Info_t>(info);
        out->AddMarshal<int*>(pBufferSize);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseZcsrilu02_bufferSize Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Scsrilu02_analysis){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Scsrilu02_analysis"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    const int m = in->Get<int>();
    const int nnz = in->Get<int>();
    cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    float * csrVal = in->GetFromMarshal<float*>();
    const int * csrRowPtr = in->GetFromMarshal<int*>();
    const int * csrColInd = in->GetFromMarshal<int*>();
    csrilu02Info_t info = (csrilu02Info_t)in->Get<size_t>();
    cusparseSolvePolicy_t policy = in->Get<cusparseSolvePolicy_t>();
    void * pBuffer = in->GetFromMarshal<void*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseScsrilu02_analysis(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, info, policy, pBuffer);
        out->AddMarshal<csrilu02Info_t>(info);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseScsrilu02_analysis Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Dcsrilu02_analysis){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Dcsrilu02_analysis"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    const int m = in->Get<int>();
    const int nnz = in->Get<int>();
    cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    double * csrVal = in->GetFromMarshal<double*>();
    const int * csrRowPtr = in->GetFromMarshal<int*>();
    const int * csrColInd = in->GetFromMarshal<int*>();
    csrilu02Info_t info = (csrilu02Info_t)in->Get<size_t>();
    cusparseSolvePolicy_t policy = in->Get<cusparseSolvePolicy_t>();
    void * pBuffer = in->GetFromMarshal<void*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseDcsrilu02_analysis(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, info, policy, pBuffer);
        out->AddMarshal<csrilu02Info_t>(info);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseDcsrilu02_analysis Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Ccsrilu02_analysis){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Ccsrilu02_analysis"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    const int m = in->Get<int>();
    const int nnz = in->Get<int>();
    cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    cuComplex * csrVal = in->GetFromMarshal<cuComplex*>();
    const int * csrRowPtr = in->GetFromMarshal<int*>();
    const int * csrColInd = in->GetFromMarshal<int*>();
    csrilu02Info_t info = (csrilu02Info_t)in->Get<size_t>();
    cusparseSolvePolicy_t policy = in->Get<cusparseSolvePolicy_t>();
    void * pBuffer = in->GetFromMarshal<void*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseCcsrilu02_analysis(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, info, policy, pBuffer);
        out->AddMarshal<csrilu02Info_t>(info);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseCcsrilu02_analysis Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Zcsrilu02_analysis){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Zcsrilu02_analysis"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    const int m = in->Get<int>();
    const int nnz = in->Get<int>();
    cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    cuDoubleComplex * csrVal = in->GetFromMarshal<cuDoubleComplex*>();
    const int * csrRowPtr = in->GetFromMarshal<int*>();
    const int * csrColInd = in->GetFromMarshal<int*>();
    csrilu02Info_t info = (csrilu02Info_t)in->Get<size_t>();
    cusparseSolvePolicy_t policy = in->Get<cusparseSolvePolicy_t>();
    void * pBuffer = in->GetFromMarshal<void*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseZcsrilu02_analysis(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, info, policy, pBuffer);
        out->AddMarshal<csrilu02Info_t>(info);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseZcsrilu02_analysis Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Scsrilu02){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Scsrilu02"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    const int m = in->Get<int>();
    const int nnz = in->Get<int>();
    cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    float * csrVal = in->GetFromMarshal<float*>();
    const int * csrRowPtr = in->GetFromMarshal<int*>();
    const int * csrColInd = in->GetFromMarshal<int*>();
    csrilu02Info_t info = (csrilu02Info_t)in->Get<size_t>();
    cusparseSolvePolicy_t policy = in->Get<cusparseSolvePolicy_t>();
    void * pBuffer = in->GetFromMarshal<void*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseScsrilu02(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, info, policy, pBuffer);
        out->AddMarshal<float*>(csrVal);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseScsrilu02 Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Dcsrilu02){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Dcsrilu02"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    const int m = in->Get<int>();
    const int nnz = in->Get<int>();
    cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    double * csrVal = in->GetFromMarshal<double*>();
    const int * csrRowPtr = in->GetFromMarshal<int*>();
    const int * csrColInd = in->GetFromMarshal<int*>();
    csrilu02Info_t info = (csrilu02Info_t)in->Get<size_t>();
    cusparseSolvePolicy_t policy = in->Get<cusparseSolvePolicy_t>();
    void * pBuffer = in->GetFromMarshal<void*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseDcsrilu02(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, info, policy, pBuffer);
        out->AddMarshal<double*>(csrVal);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseDcsrilu02 Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Ccsrilu02){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Ccsrilu02"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    const int m = in->Get<int>();
    const int nnz = in->Get<int>();
    cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    cuComplex * csrVal = in->GetFromMarshal<cuComplex*>();
    const int * csrRowPtr = in->GetFromMarshal<int*>();
    const int * csrColInd = in->GetFromMarshal<int*>();
    csrilu02Info_t info = (csrilu02Info_t)in->Get<size_t>();
    cusparseSolvePolicy_t policy = in->Get<cusparseSolvePolicy_t>();
    void * pBuffer = in->GetFromMarshal<void*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseCcsrilu02(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, info, policy, pBuffer);
        out->AddMarshal<cuComplex*>(csrVal);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseCcsrilu02 Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Zcsrilu02){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Zcsrilu02"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    const int m = in->Get<int>();
    const int nnz = in->Get<int>();
    cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    cuDoubleComplex * csrVal = in->GetFromMarshal<cuDoubleComplex*>();
    const int * csrRowPtr = in->GetFromMarshal<int*>();
    const int * csrColInd = in->GetFromMarshal<int*>();
    csrilu02Info_t info = (csrilu02Info_t)in->Get<size_t>();
    cusparseSolvePolicy_t policy = in->Get<cusparseSolvePolicy_t>();
    void * pBuffer = in->GetFromMarshal<void*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseZcsrilu02(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, info, policy, pBuffer);
        out->AddMarshal<cuDoubleComplex*>(csrVal);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseZcsrilu02 Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Xcsrilu02_zeroPivot){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Xcsrilu02_zeroPivot"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    csrilu02Info_t info = (csrilu02Info_t)in->Get<size_t>();
    int * position = new int;
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseXcsrilu02_zeroPivot(handle, info, position);
        out->AddMarshal<int*>(position);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseXcsrilu02_zeroPivot Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Sbsrilu02_numericBoost){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Sbsrilu02_numericBoost"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    bsrilu02Info_t info = (bsrilu02Info_t)in->Get<size_t>();
    int enable_boost = in->Get<int>();
    double * tol = in->GetFromMarshal<double*>();
    float * boost_val = in->GetFromMarshal<float*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseSbsrilu02_numericBoost(handle, info, enable_boost, tol, boost_val);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseSbsrilu02_numericBoost Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Dbsrilu02_numericBoost){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Dbsrilu02_numericBoost"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    bsrilu02Info_t info = (bsrilu02Info_t)in->Get<size_t>();
    int enable_boost = in->Get<int>();
    double * tol = in->GetFromMarshal<double*>();
    double * boost_val = in->GetFromMarshal<double*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseDbsrilu02_numericBoost(handle, info, enable_boost, tol, boost_val);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseDbsrilu02_numericBoost Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Cbsrilu02_numericBoost){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Cbsrilu02_numericBoost"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    bsrilu02Info_t info = (bsrilu02Info_t)in->Get<size_t>();
    int enable_boost = in->Get<int>();
    double * tol = in->GetFromMarshal<double*>();
    cuComplex * boost_val = in->GetFromMarshal<cuComplex*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseCbsrilu02_numericBoost(handle, info, enable_boost, tol, boost_val);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseCbsrilu02_numericBoost Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Zbsrilu02_numericBoost){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Zbsrilu02_numericBoost"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    bsrilu02Info_t info = (bsrilu02Info_t)in->Get<size_t>();
    int enable_boost = in->Get<int>();
    double * tol = in->GetFromMarshal<double*>();
    cuDoubleComplex * boost_val = in->GetFromMarshal<cuDoubleComplex*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseZbsrilu02_numericBoost(handle, info, enable_boost, tol, boost_val);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseZbsrilu02_numericBoost Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Sbsrilu02_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Sbsrilu02_bufferSize"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    cusparseDirection_t dirA = in->Get<cusparseDirection_t>();
    int mb = in->Get<int>();
    int nnzb = in->Get<int>();
    cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    float * bsrVal = in->GetFromMarshal<float*>();
    const int * bsrRowPtr = in->GetFromMarshal<int*>();
    const int * bsrColInd = in->GetFromMarshal<int*>();
    int blockDim = in->Get<int>();
    bsrilu02Info_t info = (bsrilu02Info_t)in->Get<size_t>();
    int * pBufferSize = new int;
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseSbsrilu02_bufferSize(handle, dirA, mb, nnzb, descrA, bsrVal, bsrRowPtr, bsrColInd, blockDim, info, pBufferSize);
        out->Add<bsrilu02Info_t>(info);
        out->AddMarshal<int*>(pBufferSize);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseSbsrilu02_bufferSize Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Dbsrilu02_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Dbsrilu02_bufferSize"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    cusparseDirection_t dirA = in->Get<cusparseDirection_t>();
    int mb = in->Get<int>();
    int nnzb = in->Get<int>();
    cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    double * bsrVal = in->GetFromMarshal<double*>();
    const int * bsrRowPtr = in->GetFromMarshal<int*>();
    const int * bsrColInd = in->GetFromMarshal<int*>();
    int blockDim = in->Get<int>();
    bsrilu02Info_t info = (bsrilu02Info_t)in->Get<size_t>();
    int * pBufferSize = new int;
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseDbsrilu02_bufferSize(handle, dirA, mb, nnzb, descrA, bsrVal, bsrRowPtr, bsrColInd, blockDim, info, pBufferSize);
        out->Add<bsrilu02Info_t>(info);
        out->AddMarshal<int*>(pBufferSize);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseDbsrilu02_bufferSize Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Cbsrilu02_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Cbsrilu02_bufferSize"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    cusparseDirection_t dirA = in->Get<cusparseDirection_t>();
    int mb = in->Get<int>();
    int nnzb = in->Get<int>();
    cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    cuComplex * bsrVal = in->GetFromMarshal<cuComplex*>();
    const int * bsrRowPtr = in->GetFromMarshal<int*>();
    const int * bsrColInd = in->GetFromMarshal<int*>();
    int blockDim = in->Get<int>();
    bsrilu02Info_t info = (bsrilu02Info_t)in->Get<size_t>();
    int * pBufferSize = new int;
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseCbsrilu02_bufferSize(handle, dirA, mb, nnzb, descrA, bsrVal, bsrRowPtr, bsrColInd, blockDim, info, pBufferSize);
        out->Add<bsrilu02Info_t>(info);
        out->AddMarshal<int*>(pBufferSize);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseCbsrilu02_bufferSize Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Zbsrilu02_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Zbsrilu02_bufferSize"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    cusparseDirection_t dirA = in->Get<cusparseDirection_t>();
    int mb = in->Get<int>();
    int nnzb = in->Get<int>();
    cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    cuDoubleComplex * bsrVal = in->GetFromMarshal<cuDoubleComplex*>();
    const int * bsrRowPtr = in->GetFromMarshal<int*>();
    const int * bsrColInd = in->GetFromMarshal<int*>();
    int blockDim = in->Get<int>();
    bsrilu02Info_t info = (bsrilu02Info_t)in->Get<size_t>();
    int * pBufferSize = new int;
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseZbsrilu02_bufferSize(handle, dirA, mb, nnzb, descrA, bsrVal, bsrRowPtr, bsrColInd, blockDim, info, pBufferSize);
        out->Add<bsrilu02Info_t>(info);
        out->AddMarshal<int*>(pBufferSize);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseZbsrilu02_bufferSize Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Sbsrilu02_analysis){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Sbsrilu02_analysis"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    cusparseDirection_t dirA = in->Get<cusparseDirection_t>();
    int mb = in->Get<int>();
    int nnzb = in->Get<int>();
    cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    float * bsrVal = in->GetFromMarshal<float*>();
    const int * bsrRowPtr = in->GetFromMarshal<int*>();
    const int * bsrColInd = in->GetFromMarshal<int*>();
    int blockDim = in->Get<int>();
    bsrilu02Info_t info = (bsrilu02Info_t)in->Get<size_t>();
    cusparseSolvePolicy_t policy = in->Get<cusparseSolvePolicy_t>();
    void * pBuffer = in->GetFromMarshal<void*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseSbsrilu02_analysis(handle, dirA, mb, nnzb, descrA, bsrVal, bsrRowPtr, bsrColInd, blockDim, info, policy, pBuffer);
        out->AddMarshal<bsrilu02Info_t>(info);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseSbsrilu02_analysis Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Dbsrilu02_analysis){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Dbsrilu02_analysis"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    cusparseDirection_t dirA = in->Get<cusparseDirection_t>();
    int mb = in->Get<int>();
    int nnzb = in->Get<int>();
    cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    double * bsrVal = in->GetFromMarshal<double*>();
    const int * bsrRowPtr = in->GetFromMarshal<int*>();
    const int * bsrColInd = in->GetFromMarshal<int*>();
    int blockDim = in->Get<int>();
    bsrilu02Info_t info = (bsrilu02Info_t)in->Get<size_t>();
    cusparseSolvePolicy_t policy = in->Get<cusparseSolvePolicy_t>();
    void * pBuffer = in->GetFromMarshal<void*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseDbsrilu02_analysis(handle, dirA, mb, nnzb, descrA, bsrVal, bsrRowPtr, bsrColInd, blockDim, info, policy, pBuffer);
        out->AddMarshal<bsrilu02Info_t>(info);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseDbsrilu02_analysis Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Cbsrilu02_analysis){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Cbsrilu02_analysis"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    cusparseDirection_t dirA = in->Get<cusparseDirection_t>();
    int mb = in->Get<int>();
    int nnzb = in->Get<int>();
    cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    cuComplex * bsrVal = in->GetFromMarshal<cuComplex*>();
    const int * bsrRowPtr = in->GetFromMarshal<int*>();
    const int * bsrColInd = in->GetFromMarshal<int*>();
    int blockDim = in->Get<int>();
    bsrilu02Info_t info = (bsrilu02Info_t)in->Get<size_t>();
    cusparseSolvePolicy_t policy = in->Get<cusparseSolvePolicy_t>();
    void * pBuffer = in->GetFromMarshal<void*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseCbsrilu02_analysis(handle, dirA, mb, nnzb, descrA, bsrVal, bsrRowPtr, bsrColInd, blockDim, info, policy, pBuffer);
        out->AddMarshal<bsrilu02Info_t>(info);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseCbsrilu02_analysis Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Zbsrilu02_analysis){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Zbsrilu02_analysis"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    cusparseDirection_t dirA = in->Get<cusparseDirection_t>();
    int mb = in->Get<int>();
    int nnzb = in->Get<int>();
    cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    cuDoubleComplex * bsrVal = in->GetFromMarshal<cuDoubleComplex*>();
    const int * bsrRowPtr = in->GetFromMarshal<int*>();
    const int * bsrColInd = in->GetFromMarshal<int*>();
    int blockDim = in->Get<int>();
    bsrilu02Info_t info = (bsrilu02Info_t)in->Get<size_t>();
    cusparseSolvePolicy_t policy = in->Get<cusparseSolvePolicy_t>();
    void * pBuffer = in->GetFromMarshal<void*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseZbsrilu02_analysis(handle, dirA, mb, nnzb, descrA, bsrVal, bsrRowPtr, bsrColInd, blockDim, info, policy, pBuffer);
        out->AddMarshal<bsrilu02Info_t>(info);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseZbsrilu02_analysis Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Sbsrilu02){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Sbsrilu02"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    cusparseDirection_t dirA = in->Get<cusparseDirection_t>();
    int mb = in->Get<int>();
    int nnzb = in->Get<int>();
    cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    float * bsrVal = in->GetFromMarshal<float*>();
    const int * bsrRowPtr = in->GetFromMarshal<int*>();
    const int * bsrColInd = in->GetFromMarshal<int*>();
    int blockDim = in->Get<int>();
    bsrilu02Info_t info = (bsrilu02Info_t)in->Get<size_t>();
    cusparseSolvePolicy_t policy = in->Get<cusparseSolvePolicy_t>();
    void * pBuffer = in->GetFromMarshal<void*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseSbsrilu02(handle, dirA, mb, nnzb, descrA, bsrVal, bsrRowPtr, bsrColInd, blockDim, info, policy, pBuffer);
        out->AddMarshal<float*>(bsrVal);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseSbsrilu02 Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Dbsrilu02){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Dbsrilu02"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    cusparseDirection_t dirA = in->Get<cusparseDirection_t>();
    int mb = in->Get<int>();
    int nnzb = in->Get<int>();
    cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    double * bsrVal = in->GetFromMarshal<double*>();
    const int * bsrRowPtr = in->GetFromMarshal<int*>();
    const int * bsrColInd = in->GetFromMarshal<int*>();
    int blockDim = in->Get<int>();
    bsrilu02Info_t info = (bsrilu02Info_t)in->Get<size_t>();
    cusparseSolvePolicy_t policy = in->Get<cusparseSolvePolicy_t>();
    void * pBuffer = in->GetFromMarshal<void*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseDbsrilu02(handle, dirA, mb, nnzb, descrA, bsrVal, bsrRowPtr, bsrColInd, blockDim, info, policy, pBuffer);
        out->AddMarshal<double*>(bsrVal);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseDbsrilu02 Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Cbsrilu02){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Cbsrilu02"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    cusparseDirection_t dirA = in->Get<cusparseDirection_t>();
    int mb = in->Get<int>();
    int nnzb = in->Get<int>();
    cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    cuComplex * bsrVal = in->GetFromMarshal<cuComplex*>();
    const int * bsrRowPtr = in->GetFromMarshal<int*>();
    const int * bsrColInd = in->GetFromMarshal<int*>();
    int blockDim = in->Get<int>();
    bsrilu02Info_t info = (bsrilu02Info_t)in->Get<size_t>();
    cusparseSolvePolicy_t policy = in->Get<cusparseSolvePolicy_t>();
    void * pBuffer = in->GetFromMarshal<void*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseCbsrilu02(handle, dirA, mb, nnzb, descrA, bsrVal, bsrRowPtr, bsrColInd, blockDim, info, policy, pBuffer);
        out->AddMarshal<cuComplex*>(bsrVal);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseCbsrilu02 Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Zbsrilu02){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Zbsrilu02"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    cusparseDirection_t dirA = in->Get<cusparseDirection_t>();
    int mb = in->Get<int>();
    int nnzb = in->Get<int>();
    cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    cuDoubleComplex * bsrVal = in->GetFromMarshal<cuDoubleComplex*>();
    const int * bsrRowPtr = in->GetFromMarshal<int*>();
    const int * bsrColInd = in->GetFromMarshal<int*>();
    int blockDim = in->Get<int>();
    bsrilu02Info_t info = (bsrilu02Info_t)in->Get<size_t>();
    cusparseSolvePolicy_t policy = in->Get<cusparseSolvePolicy_t>();
    void * pBuffer = in->GetFromMarshal<void*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseZbsrilu02(handle, dirA, mb, nnzb, descrA, bsrVal, bsrRowPtr, bsrColInd, blockDim, info, policy, pBuffer);
        out->AddMarshal<cuDoubleComplex*>(bsrVal);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseZbsrilu02 Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Xbsrilu02_zeroPivot){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Xbsrilu02_zeroPivot"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    bsrilu02Info_t info = (bsrilu02Info_t)in->Get<size_t>();
    int * position = new int;
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseXbsrilu02_zeroPivot(handle, info, position);
        out->AddMarshal<int*>(position);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseXbsrilu02_zeroPivot Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Sgtsv2_bufferSizeExt){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Sgtsv2_bufferSizeExt"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    float * dl = in->GetFromMarshal<float*>();
    float * d = in->GetFromMarshal<float*>();
    float * du = in->GetFromMarshal<float*>();
    float * B = in->GetFromMarshal<float*>();
    int ldb = in->Get<int>();
    size_t * pBufferSize = new size_t;
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseSgtsv2_bufferSizeExt(handle, m, n, dl, d, du, B, ldb, pBufferSize);
        out->AddMarshal<size_t*>(pBufferSize);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseSgtsv2_bufferSizeExt Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Dgtsv2_bufferSizeExt){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Dgtsv2_bufferSizeExt"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    double * dl = in->GetFromMarshal<double*>();
    double * d = in->GetFromMarshal<double*>();
    double * du = in->GetFromMarshal<double*>();
    double * B = in->GetFromMarshal<double*>();
    int ldb = in->Get<int>();
    size_t * pBufferSize = new size_t;
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseDgtsv2_bufferSizeExt(handle, m, n, dl, d, du, B, ldb, pBufferSize);
        out->AddMarshal<size_t*>(pBufferSize);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseDgtsv2_bufferSizeExt Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Cgtsv2_bufferSizeExt){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Cgtsv2_bufferSizeExt"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    cuComplex * dl = in->GetFromMarshal<cuComplex*>();
    cuComplex * d = in->GetFromMarshal<cuComplex*>();
    cuComplex * du = in->GetFromMarshal<cuComplex*>();
    cuComplex * B = in->GetFromMarshal<cuComplex*>();
    int ldb = in->Get<int>();
    size_t * pBufferSize = new size_t;
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseCgtsv2_bufferSizeExt(handle, m, n, dl, d, du, B, ldb, pBufferSize);
        out->AddMarshal<size_t*>(pBufferSize);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseCgtsv2_bufferSizeExt Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Zgtsv2_bufferSizeExt){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Zgtsv2_bufferSizeExt"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    cuDoubleComplex * dl = in->GetFromMarshal<cuDoubleComplex*>();
    cuDoubleComplex * d = in->GetFromMarshal<cuDoubleComplex*>();
    cuDoubleComplex * du = in->GetFromMarshal<cuDoubleComplex*>();
    cuDoubleComplex * B = in->GetFromMarshal<cuDoubleComplex*>();
    int ldb = in->Get<int>();
    size_t * pBufferSize = new size_t;
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseZgtsv2_bufferSizeExt(handle, m, n, dl, d, du, B, ldb, pBufferSize);
        out->AddMarshal<size_t*>(pBufferSize);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseZgtsv2_bufferSizeExt Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Sgtsv2){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Sgtsv2"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    float * dl = in->GetFromMarshal<float*>();
    float * d = in->GetFromMarshal<float*>();
    float * du = in->GetFromMarshal<float*>();
    float * B = in->GetFromMarshal<float*>();
    int ldb = in->Get<int>();
    void * pBuffer = in->GetFromMarshal<void*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseSgtsv2(handle, m, n, dl, d, du, B, ldb, pBuffer);
        out->AddMarshal<float*>(B);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseSgtsv2 Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Dgtsv2){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Dgtsv2"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    double * dl = in->GetFromMarshal<double*>();
    double * d = in->GetFromMarshal<double*>();
    double * du = in->GetFromMarshal<double*>();
    double * B = in->GetFromMarshal<double*>();
    int ldb = in->Get<int>();
    void * pBuffer = in->GetFromMarshal<void*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseDgtsv2(handle, m, n, dl, d, du, B, ldb, pBuffer);
        out->AddMarshal<double*>(B);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseDgtsv2 Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Cgtsv2){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Cgtsv2"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    cuComplex * dl = in->GetFromMarshal<cuComplex*>();
    cuComplex * d = in->GetFromMarshal<cuComplex*>();
    cuComplex * du = in->GetFromMarshal<cuComplex*>();
    cuComplex * B = in->GetFromMarshal<cuComplex*>();
    int ldb = in->Get<int>();
    void * pBuffer = in->GetFromMarshal<void*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseCgtsv2(handle, m, n, dl, d, du, B, ldb, pBuffer);
        out->AddMarshal<cuComplex*>(B);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseCgtsv2 Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Zgtsv2){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Zgtsv2"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    cuDoubleComplex * dl = in->GetFromMarshal<cuDoubleComplex*>();
    cuDoubleComplex * d = in->GetFromMarshal<cuDoubleComplex*>();
    cuDoubleComplex * du = in->GetFromMarshal<cuDoubleComplex*>();
    cuDoubleComplex * B = in->GetFromMarshal<cuDoubleComplex*>();
    int ldb = in->Get<int>();
    void * pBuffer = in->GetFromMarshal<void*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseZgtsv2(handle, m, n, dl, d, du, B, ldb, pBuffer);
        out->AddMarshal<cuDoubleComplex*>(B);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseZgtsv2 Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Sgtsv2_nopivot_bufferSizeExt){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Sgtsv2_nopivot_bufferSizeExt"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    float * dl = in->GetFromMarshal<float*>();
    float * d = in->GetFromMarshal<float*>();
    float * du = in->GetFromMarshal<float*>();
    float * B = in->GetFromMarshal<float*>();
    int ldb = in->Get<int>();
    size_t * pBufferSize = new size_t;
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseSgtsv2_nopivot_bufferSizeExt(handle, m, n, dl, d, du, B, ldb, pBufferSize);
        out->AddMarshal<size_t*>(pBufferSize);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseSgtsv2_nopivot_bufferSizeExt Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Dgtsv2_nopivot_bufferSizeExt){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Dgtsv2_nopivot_bufferSizeExt"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    double * dl = in->GetFromMarshal<double*>();
    double * d = in->GetFromMarshal<double*>();
    double * du = in->GetFromMarshal<double*>();
    double * B = in->GetFromMarshal<double*>();
    int ldb = in->Get<int>();
    size_t * pBufferSize = new size_t;
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseDgtsv2_nopivot_bufferSizeExt(handle, m, n, dl, d, du, B, ldb, pBufferSize);
        out->AddMarshal<size_t*>(pBufferSize);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseDgtsv2_nopivot_bufferSizeExt Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Cgtsv2_nopivot_bufferSizeExt){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Cgtsv2_nopivot_bufferSizeExt"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    cuComplex * dl = in->GetFromMarshal<cuComplex*>();
    cuComplex * d = in->GetFromMarshal<cuComplex*>();
    cuComplex * du = in->GetFromMarshal<cuComplex*>();
    cuComplex * B = in->GetFromMarshal<cuComplex*>();
    int ldb = in->Get<int>();
    size_t * pBufferSize = new size_t;
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseCgtsv2_nopivot_bufferSizeExt(handle, m, n, dl, d, du, B, ldb, pBufferSize);
        out->AddMarshal<size_t*>(pBufferSize);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseCgtsv2_nopivot_bufferSizeExt Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Zgtsv2_nopivot_bufferSizeExt){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Zgtsv2_nopivot_bufferSizeExt"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    cuDoubleComplex * dl = in->GetFromMarshal<cuDoubleComplex*>();
    cuDoubleComplex * d = in->GetFromMarshal<cuDoubleComplex*>();
    cuDoubleComplex * du = in->GetFromMarshal<cuDoubleComplex*>();
    cuDoubleComplex * B = in->GetFromMarshal<cuDoubleComplex*>();
    int ldb = in->Get<int>();
    size_t * pBufferSize = new size_t;
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseZgtsv2_nopivot_bufferSizeExt(handle, m, n, dl, d, du, B, ldb, pBufferSize);
        out->AddMarshal<size_t*>(pBufferSize);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseZgtsv2_nopivot_bufferSizeExt Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Sgtsv2_nopivot){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Sgtsv2_nopivot"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    float * dl = in->GetFromMarshal<float*>();
    float * d = in->GetFromMarshal<float*>();
    float * du = in->GetFromMarshal<float*>();
    float * B = in->GetFromMarshal<float*>();
    int ldb = in->Get<int>();
    void * pBuffer = in->GetFromMarshal<void*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseSgtsv2_nopivot(handle, m, n, dl, d, du, B, ldb, pBuffer);
        out->AddMarshal<float*>(B);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseSgtsv2_nopivot Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Dgtsv2_nopivot){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Dgtsv2_nopivot"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    double * dl = in->GetFromMarshal<double*>();
    double * d = in->GetFromMarshal<double*>();
    double * du = in->GetFromMarshal<double*>();
    double * B = in->GetFromMarshal<double*>();
    int ldb = in->Get<int>();
    void * pBuffer = in->GetFromMarshal<void*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseDgtsv2_nopivot(handle, m, n, dl, d, du, B, ldb, pBuffer);
        out->AddMarshal<double*>(B);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseDgtsv2_nopivot Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Cgtsv2_nopivot){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Cgtsv2_nopivot"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    cuComplex * dl = in->GetFromMarshal<cuComplex*>();
    cuComplex * d = in->GetFromMarshal<cuComplex*>();
    cuComplex * du = in->GetFromMarshal<cuComplex*>();
    cuComplex * B = in->GetFromMarshal<cuComplex*>();
    int ldb = in->Get<int>();
    void * pBuffer = in->GetFromMarshal<void*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseCgtsv2_nopivot(handle, m, n, dl, d, du, B, ldb, pBuffer);
        out->AddMarshal<cuComplex*>(B);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseCgtsv2_nopivot Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Zgtsv2_nopivot){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Zgtsv2_nopivot"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    cuDoubleComplex * dl = in->GetFromMarshal<cuDoubleComplex*>();
    cuDoubleComplex * d = in->GetFromMarshal<cuDoubleComplex*>();
    cuDoubleComplex * du = in->GetFromMarshal<cuDoubleComplex*>();
    cuDoubleComplex * B = in->GetFromMarshal<cuDoubleComplex*>();
    int ldb = in->Get<int>();
    void * pBuffer = in->GetFromMarshal<void*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseZgtsv2_nopivot(handle, m, n, dl, d, du, B, ldb, pBuffer);
        out->AddMarshal<cuDoubleComplex*>(B);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseZgtsv2_nopivot Executed");
    return std::make_shared<Result>(cs,out);
}

#ifndef CUSPARSE_VERSION
#error CUSPARSE_VERSION not defined
#endif
