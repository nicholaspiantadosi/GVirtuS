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

CUSPARSE_ROUTINE_HANDLER(Sbsrmm){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Sbsrmm"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    const cusparseDirection_t dirA = in->Get<cusparseDirection_t>();
    const cusparseOperation_t transA = in->Get<cusparseOperation_t>();
    const cusparseOperation_t transB = in->Get<cusparseOperation_t>();
    const int mb = in->Get<int>();
    const int n = in->Get<int>();
    const int kb = in->Get<int>();
    const int nnzb = in->Get<int>();
    const float * alpha = in->Assign<float>();
    const cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    const float * bsrValA = in->GetFromMarshal<float*>();
    const int * bsrRowPtrA = in->GetFromMarshal<int*>();
    const int * bsrColIndA = in->GetFromMarshal<int*>();
    const int blockDim = in->Get<int>();
    float * B = in->Get<float*>();
    const int ldb = in->Get<int>();
    const float * beta = in->Assign<float>();
    float * C = in->Get<float*>();
    const int ldc = in->Get<int>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseSbsrmm(handle, dirA, transA, transB, mb, n, kb, nnzb, alpha, descrA, bsrValA, bsrRowPtrA, bsrColIndA, blockDim, B, ldb, beta, C, ldc);
        out->Add<float*>(C);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseSbsrmm Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Dbsrmm){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Dbsrmm"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    const cusparseDirection_t dirA = in->Get<cusparseDirection_t>();
    const cusparseOperation_t transA = in->Get<cusparseOperation_t>();
    const cusparseOperation_t transB = in->Get<cusparseOperation_t>();
    const int mb = in->Get<int>();
    const int n = in->Get<int>();
    const int kb = in->Get<int>();
    const int nnzb = in->Get<int>();
    const double * alpha = in->Assign<double>();
    const cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    const double * bsrValA = in->GetFromMarshal<double*>();
    const int * bsrRowPtrA = in->GetFromMarshal<int*>();
    const int * bsrColIndA = in->GetFromMarshal<int*>();
    const int blockDim = in->Get<int>();
    double * B = in->Get<double*>();
    const int ldb = in->Get<int>();
    const double * beta = in->Assign<double>();
    double * C = in->Get<double*>();
    const int ldc = in->Get<int>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseDbsrmm(handle, dirA, transA, transB, mb, n, kb, nnzb, alpha, descrA, bsrValA, bsrRowPtrA, bsrColIndA, blockDim, B, ldb, beta, C, ldc);
        out->Add<double*>(C);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseDbsrmm Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Cbsrmm){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Cbsrmm"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    const cusparseDirection_t dirA = in->Get<cusparseDirection_t>();
    const cusparseOperation_t transA = in->Get<cusparseOperation_t>();
    const cusparseOperation_t transB = in->Get<cusparseOperation_t>();
    const int mb = in->Get<int>();
    const int n = in->Get<int>();
    const int kb = in->Get<int>();
    const int nnzb = in->Get<int>();
    const cuComplex * alpha = in->Assign<cuComplex>();
    const cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    const cuComplex * bsrValA = in->GetFromMarshal<cuComplex*>();
    const int * bsrRowPtrA = in->GetFromMarshal<int*>();
    const int * bsrColIndA = in->GetFromMarshal<int*>();
    const int blockDim = in->Get<int>();
    cuComplex * B = in->Get<cuComplex*>();
    const int ldb = in->Get<int>();
    const cuComplex * beta = in->Assign<cuComplex>();
    cuComplex * C = in->Get<cuComplex*>();
    const int ldc = in->Get<int>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseCbsrmm(handle, dirA, transA, transB, mb, n, kb, nnzb, alpha, descrA, bsrValA, bsrRowPtrA, bsrColIndA, blockDim, B, ldb, beta, C, ldc);
        out->Add<cuComplex*>(C);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseCbsrmm Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Zbsrmm){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Zbsrmm"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    const cusparseDirection_t dirA = in->Get<cusparseDirection_t>();
    const cusparseOperation_t transA = in->Get<cusparseOperation_t>();
    const cusparseOperation_t transB = in->Get<cusparseOperation_t>();
    const int mb = in->Get<int>();
    const int n = in->Get<int>();
    const int kb = in->Get<int>();
    const int nnzb = in->Get<int>();
    const cuDoubleComplex * alpha = in->Assign<cuDoubleComplex>();
    const cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    const cuDoubleComplex * bsrValA = in->GetFromMarshal<cuDoubleComplex*>();
    const int * bsrRowPtrA = in->GetFromMarshal<int*>();
    const int * bsrColIndA = in->GetFromMarshal<int*>();
    const int blockDim = in->Get<int>();
    cuDoubleComplex * B = in->Get<cuDoubleComplex*>();
    const int ldb = in->Get<int>();
    const cuDoubleComplex * beta = in->Assign<cuDoubleComplex>();
    cuDoubleComplex * C = in->Get<cuDoubleComplex*>();
    const int ldc = in->Get<int>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseZbsrmm(handle, dirA, transA, transB, mb, n, kb, nnzb, alpha, descrA, bsrValA, bsrRowPtrA, bsrColIndA, blockDim, B, ldb, beta, C, ldc);
        out->Add<cuDoubleComplex*>(C);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseZbsrmm Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Sbsrsm2_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Sbsrsm2_bufferSize"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    cusparseDirection_t dirA = in->Get<cusparseDirection_t>();
    cusparseOperation_t transA = in->Get<cusparseOperation_t>();
    cusparseOperation_t transB = in->Get<cusparseOperation_t>();
    const int mb = in->Get<int>();
    const int m = in->Get<int>();
    const int nnzb = in->Get<int>();
    cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    float * bsrVal = in->GetFromMarshal<float*>();
    const int * bsrRowPtr = in->GetFromMarshal<int*>();
    const int * bsrColInd = in->GetFromMarshal<int*>();
    int blockDim = in->Get<int>();
    bsrsm2Info_t info = (bsrsm2Info_t)in->Get<size_t>();
    int * pBufferSize = new int;
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseSbsrsm2_bufferSize(handle, dirA, transA, transB, mb, m, nnzb, descrA, bsrVal, bsrRowPtr, bsrColInd, blockDim, info, pBufferSize);
        out->Add<bsrsm2Info_t>(info);
        out->AddMarshal<int*>(pBufferSize);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseSbsrsm2_bufferSize Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Dbsrsm2_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Dbsrsm2_bufferSize"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    cusparseDirection_t dirA = in->Get<cusparseDirection_t>();
    cusparseOperation_t transA = in->Get<cusparseOperation_t>();
    cusparseOperation_t transB = in->Get<cusparseOperation_t>();
    const int mb = in->Get<int>();
    const int m = in->Get<int>();
    const int nnzb = in->Get<int>();
    cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    double * bsrVal = in->GetFromMarshal<double*>();
    const int * bsrRowPtr = in->GetFromMarshal<int*>();
    const int * bsrColInd = in->GetFromMarshal<int*>();
    int blockDim = in->Get<int>();
    bsrsm2Info_t info = (bsrsm2Info_t)in->Get<size_t>();
    int * pBufferSize = new int;
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseDbsrsm2_bufferSize(handle, dirA, transA, transB, mb, m, nnzb, descrA, bsrVal, bsrRowPtr, bsrColInd, blockDim, info, pBufferSize);
        out->Add<bsrsm2Info_t>(info);
        out->AddMarshal<int*>(pBufferSize);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseDbsrsm2_bufferSize Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Cbsrsm2_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Cbsrsm2_bufferSize"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    cusparseDirection_t dirA = in->Get<cusparseDirection_t>();
    cusparseOperation_t transA = in->Get<cusparseOperation_t>();
    cusparseOperation_t transB = in->Get<cusparseOperation_t>();
    const int mb = in->Get<int>();
    const int m = in->Get<int>();
    const int nnzb = in->Get<int>();
    cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    cuComplex * bsrVal = in->GetFromMarshal<cuComplex*>();
    const int * bsrRowPtr = in->GetFromMarshal<int*>();
    const int * bsrColInd = in->GetFromMarshal<int*>();
    int blockDim = in->Get<int>();
    bsrsm2Info_t info = (bsrsm2Info_t)in->Get<size_t>();
    int * pBufferSize = new int;
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseCbsrsm2_bufferSize(handle, dirA, transA, transB, mb, m, nnzb, descrA, bsrVal, bsrRowPtr, bsrColInd, blockDim, info, pBufferSize);
        out->Add<bsrsm2Info_t>(info);
        out->AddMarshal<int*>(pBufferSize);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseCbsrsm2_bufferSize Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Zbsrsm2_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Zbsrsm2_bufferSize"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    cusparseDirection_t dirA = in->Get<cusparseDirection_t>();
    cusparseOperation_t transA = in->Get<cusparseOperation_t>();
    cusparseOperation_t transB = in->Get<cusparseOperation_t>();
    const int mb = in->Get<int>();
    const int m = in->Get<int>();
    const int nnzb = in->Get<int>();
    cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    cuDoubleComplex * bsrVal = in->GetFromMarshal<cuDoubleComplex*>();
    const int * bsrRowPtr = in->GetFromMarshal<int*>();
    const int * bsrColInd = in->GetFromMarshal<int*>();
    int blockDim = in->Get<int>();
    bsrsm2Info_t info = (bsrsm2Info_t)in->Get<size_t>();
    int * pBufferSize = new int;
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseZbsrsm2_bufferSize(handle, dirA, transA, transB, mb, m, nnzb, descrA, bsrVal, bsrRowPtr, bsrColInd, blockDim, info, pBufferSize);
        out->Add<bsrsm2Info_t>(info);
        out->AddMarshal<int*>(pBufferSize);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseZbsrsm2_bufferSize Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Sbsrsm2_analysis){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Sbsrsm2_analysis"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    cusparseDirection_t dirA = in->Get<cusparseDirection_t>();
    cusparseOperation_t transA = in->Get<cusparseOperation_t>();
    cusparseOperation_t transB = in->Get<cusparseOperation_t>();
    int mb = in->Get<int>();
    int m = in->Get<int>();
    int nnzb = in->Get<int>();
    cusparseMatDescr_t descrA = (cusparseMatDescr_t)in->Get<size_t>();
    float * bsrVal = in->GetFromMarshal<float*>();
    int * bsrRowPtr = in->GetFromMarshal<int*>();
    int * bsrColInd = in->GetFromMarshal<int*>();
    int blockDim = in->Get<int>();
    bsrsm2Info_t info = (bsrsm2Info_t)in->Get<size_t>();
    cusparseSolvePolicy_t policy = in->Get<cusparseSolvePolicy_t>();
    void * pBuffer = in->GetFromMarshal<void*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseSbsrsm2_analysis(handle, dirA, transA, transB, mb, m, nnzb, descrA, bsrVal, bsrRowPtr, bsrColInd, blockDim, info, policy, pBuffer);
        out->AddMarshal<bsrsm2Info_t>(info);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseSbsrsm2_analysis Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Dbsrsm2_analysis){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Dbsrsm2_analysis"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    cusparseDirection_t dirA = in->Get<cusparseDirection_t>();
    cusparseOperation_t transA = in->Get<cusparseOperation_t>();
    cusparseOperation_t transB = in->Get<cusparseOperation_t>();
    int mb = in->Get<int>();
    int m = in->Get<int>();
    int nnzb = in->Get<int>();
    cusparseMatDescr_t descrA = (cusparseMatDescr_t)in->Get<size_t>();
    double * bsrVal = in->GetFromMarshal<double*>();
    int * bsrRowPtr = in->GetFromMarshal<int*>();
    int * bsrColInd = in->GetFromMarshal<int*>();
    int blockDim = in->Get<int>();
    bsrsm2Info_t info = (bsrsm2Info_t)in->Get<size_t>();
    cusparseSolvePolicy_t policy = in->Get<cusparseSolvePolicy_t>();
    void * pBuffer = in->GetFromMarshal<void*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseDbsrsm2_analysis(handle, dirA, transA, transB, mb, m, nnzb, descrA, bsrVal, bsrRowPtr, bsrColInd, blockDim, info, policy, pBuffer);
        out->AddMarshal<bsrsm2Info_t>(info);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseDbsrsm2_analysis Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Cbsrsm2_analysis){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Cbsrsm2_analysis"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    cusparseDirection_t dirA = in->Get<cusparseDirection_t>();
    cusparseOperation_t transA = in->Get<cusparseOperation_t>();
    cusparseOperation_t transB = in->Get<cusparseOperation_t>();
    int mb = in->Get<int>();
    int m = in->Get<int>();
    int nnzb = in->Get<int>();
    cusparseMatDescr_t descrA = (cusparseMatDescr_t)in->Get<size_t>();
    cuComplex * bsrVal = in->GetFromMarshal<cuComplex*>();
    int * bsrRowPtr = in->GetFromMarshal<int*>();
    int * bsrColInd = in->GetFromMarshal<int*>();
    int blockDim = in->Get<int>();
    bsrsm2Info_t info = (bsrsm2Info_t)in->Get<size_t>();
    cusparseSolvePolicy_t policy = in->Get<cusparseSolvePolicy_t>();
    void * pBuffer = in->GetFromMarshal<void*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseCbsrsm2_analysis(handle, dirA, transA, transB, mb, m, nnzb, descrA, bsrVal, bsrRowPtr, bsrColInd, blockDim, info, policy, pBuffer);
        out->AddMarshal<bsrsm2Info_t>(info);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseCbsrsm2_analysis Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Zbsrsm2_analysis){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Zbsrsm2_analysis"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    cusparseDirection_t dirA = in->Get<cusparseDirection_t>();
    cusparseOperation_t transA = in->Get<cusparseOperation_t>();
    cusparseOperation_t transB = in->Get<cusparseOperation_t>();
    int mb = in->Get<int>();
    int m = in->Get<int>();
    int nnzb = in->Get<int>();
    cusparseMatDescr_t descrA = (cusparseMatDescr_t)in->Get<size_t>();
    cuDoubleComplex * bsrVal = in->GetFromMarshal<cuDoubleComplex*>();
    int * bsrRowPtr = in->GetFromMarshal<int*>();
    int * bsrColInd = in->GetFromMarshal<int*>();
    int blockDim = in->Get<int>();
    bsrsm2Info_t info = (bsrsm2Info_t)in->Get<size_t>();
    cusparseSolvePolicy_t policy = in->Get<cusparseSolvePolicy_t>();
    void * pBuffer = in->GetFromMarshal<void*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseZbsrsm2_analysis(handle, dirA, transA, transB, mb, m, nnzb, descrA, bsrVal, bsrRowPtr, bsrColInd, blockDim, info, policy, pBuffer);
        out->AddMarshal<bsrsm2Info_t>(info);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseZbsrsm2_analysis Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Sbsrsm2_solve){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Sbsrsm2_solve"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    cusparseDirection_t dirA = in->Get<cusparseDirection_t>();
    cusparseOperation_t transA = in->Get<cusparseOperation_t>();
    cusparseOperation_t transB = in->Get<cusparseOperation_t>();
    int mb = in->Get<int>();
    int m = in->Get<int>();
    int nnzb = in->Get<int>();
    const float * alpha = in->Assign<float>();
    cusparseMatDescr_t descrA = (cusparseMatDescr_t)in->Get<size_t>();
    float * bsrVal = in->GetFromMarshal<float*>();
    int * bsrRowPtr = in->GetFromMarshal<int*>();
    int * bsrColInd = in->GetFromMarshal<int*>();
    int blockDim = in->Get<int>();
    bsrsm2Info_t info = (bsrsm2Info_t)in->Get<size_t>();
    float * B = in->GetFromMarshal<float*>();
    int ldb = in->Get<int>();
    float * C = in->GetFromMarshal<float*>();
    int ldc = in->Get<int>();
    cusparseSolvePolicy_t policy = in->Get<cusparseSolvePolicy_t>();
    void * pBuffer = in->GetFromMarshal<void*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseSbsrsm2_solve(handle, dirA, transA, transB, mb, m, nnzb, alpha, descrA, bsrVal, bsrRowPtr, bsrColInd, blockDim, info, B, ldb, C, ldc, policy, pBuffer);
        out->AddMarshal<float*>(C);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseSbsrsm2_solve Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Dbsrsm2_solve){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Dbsrsm2_solve"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    cusparseDirection_t dirA = in->Get<cusparseDirection_t>();
    cusparseOperation_t transA = in->Get<cusparseOperation_t>();
    cusparseOperation_t transB = in->Get<cusparseOperation_t>();
    int mb = in->Get<int>();
    int m = in->Get<int>();
    int nnzb = in->Get<int>();
    const double * alpha = in->Assign<double>();
    cusparseMatDescr_t descrA = (cusparseMatDescr_t)in->Get<size_t>();
    double * bsrVal = in->GetFromMarshal<double*>();
    int * bsrRowPtr = in->GetFromMarshal<int*>();
    int * bsrColInd = in->GetFromMarshal<int*>();
    int blockDim = in->Get<int>();
    bsrsm2Info_t info = (bsrsm2Info_t)in->Get<size_t>();
    double * B = in->GetFromMarshal<double*>();
    int ldb = in->Get<int>();
    double * C = in->GetFromMarshal<double*>();
    int ldc = in->Get<int>();
    cusparseSolvePolicy_t policy = in->Get<cusparseSolvePolicy_t>();
    void * pBuffer = in->GetFromMarshal<void*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseDbsrsm2_solve(handle, dirA, transA, transB, mb, m, nnzb, alpha, descrA, bsrVal, bsrRowPtr, bsrColInd, blockDim, info, B, ldb, C, ldc, policy, pBuffer);
        out->AddMarshal<double*>(C);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseDbsrsm2_solve Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Cbsrsm2_solve){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Cbsrsm2_solve"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    cusparseDirection_t dirA = in->Get<cusparseDirection_t>();
    cusparseOperation_t transA = in->Get<cusparseOperation_t>();
    cusparseOperation_t transB = in->Get<cusparseOperation_t>();
    int mb = in->Get<int>();
    int m = in->Get<int>();
    int nnzb = in->Get<int>();
    const cuComplex * alpha = in->Assign<cuComplex>();
    cusparseMatDescr_t descrA = (cusparseMatDescr_t)in->Get<size_t>();
    cuComplex * bsrVal = in->GetFromMarshal<cuComplex*>();
    int * bsrRowPtr = in->GetFromMarshal<int*>();
    int * bsrColInd = in->GetFromMarshal<int*>();
    int blockDim = in->Get<int>();
    bsrsm2Info_t info = (bsrsm2Info_t)in->Get<size_t>();
    cuComplex * B = in->GetFromMarshal<cuComplex*>();
    int ldb = in->Get<int>();
    cuComplex * C = in->GetFromMarshal<cuComplex*>();
    int ldc = in->Get<int>();
    cusparseSolvePolicy_t policy = in->Get<cusparseSolvePolicy_t>();
    void * pBuffer = in->GetFromMarshal<void*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseCbsrsm2_solve(handle, dirA, transA, transB, mb, m, nnzb, alpha, descrA, bsrVal, bsrRowPtr, bsrColInd, blockDim, info, B, ldb, C, ldc, policy, pBuffer);
        out->AddMarshal<cuComplex*>(C);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseCbsrsm2_solve Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Zbsrsm2_solve){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Zbsrsm2_solve"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    cusparseDirection_t dirA = in->Get<cusparseDirection_t>();
    cusparseOperation_t transA = in->Get<cusparseOperation_t>();
    cusparseOperation_t transB = in->Get<cusparseOperation_t>();
    int mb = in->Get<int>();
    int m = in->Get<int>();
    int nnzb = in->Get<int>();
    const cuDoubleComplex * alpha = in->Assign<cuDoubleComplex>();
    cusparseMatDescr_t descrA = (cusparseMatDescr_t)in->Get<size_t>();
    cuDoubleComplex * bsrVal = in->GetFromMarshal<cuDoubleComplex*>();
    int * bsrRowPtr = in->GetFromMarshal<int*>();
    int * bsrColInd = in->GetFromMarshal<int*>();
    int blockDim = in->Get<int>();
    bsrsm2Info_t info = (bsrsm2Info_t)in->Get<size_t>();
    cuDoubleComplex * B = in->GetFromMarshal<cuDoubleComplex*>();
    int ldb = in->Get<int>();
    cuDoubleComplex * C = in->GetFromMarshal<cuDoubleComplex*>();
    int ldc = in->Get<int>();
    cusparseSolvePolicy_t policy = in->Get<cusparseSolvePolicy_t>();
    void * pBuffer = in->GetFromMarshal<void*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseZbsrsm2_solve(handle, dirA, transA, transB, mb, m, nnzb, alpha, descrA, bsrVal, bsrRowPtr, bsrColInd, blockDim, info, B, ldb, C, ldc, policy, pBuffer);
        out->AddMarshal<cuDoubleComplex*>(C);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseZbsrsm2_solve Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Xbsrsm2_zeroPivot){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Xbsrsm2_zeroPivot"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    bsrsm2Info_t info = (bsrsm2Info_t)in->Get<size_t>();
    int * position = new int;
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseXbsrsm2_zeroPivot(handle, info, position);
        out->AddMarshal<int*>(position);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseXbsrsm2_zeroPivot Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Scsrsm2_bufferSizeExt){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Scsrsm2_bufferSizeExt"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    const int algo = in->Get<int>();
    cusparseOperation_t transA = in->Get<cusparseOperation_t>();
    cusparseOperation_t transB = in->Get<cusparseOperation_t>();
    const int m = in->Get<int>();
    const int nrhs = in->Get<int>();
    const int nnz = in->Get<int>();
    const float * alpha = in->Assign<float>();
    cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    float * csrVal = in->GetFromMarshal<float*>();
    const int * csrRowPtr = in->GetFromMarshal<int*>();
    const int * csrColInd = in->GetFromMarshal<int*>();
    float * B = in->GetFromMarshal<float*>();
    int ldb = in->Get<int>();
    csrsm2Info_t info = (csrsm2Info_t)in->Get<size_t>();
    cusparseSolvePolicy_t policy = in->Get<cusparseSolvePolicy_t>();
    size_t * pBufferSize = new size_t;
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseScsrsm2_bufferSizeExt(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrVal, csrRowPtr, csrColInd, B, ldb, info, policy, pBufferSize);
        out->Add<csrsm2Info_t>(info);
        out->AddMarshal<size_t*>(pBufferSize);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseScsrsm2_bufferSizeExt Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Dcsrsm2_bufferSizeExt){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Dcsrsm2_bufferSizeExt"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    const int algo = in->Get<int>();
    cusparseOperation_t transA = in->Get<cusparseOperation_t>();
    cusparseOperation_t transB = in->Get<cusparseOperation_t>();
    const int m = in->Get<int>();
    const int nrhs = in->Get<int>();
    const int nnz = in->Get<int>();
    const double * alpha = in->Assign<double>();
    cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    double * csrVal = in->GetFromMarshal<double*>();
    const int * csrRowPtr = in->GetFromMarshal<int*>();
    const int * csrColInd = in->GetFromMarshal<int*>();
    double * B = in->GetFromMarshal<double*>();
    int ldb = in->Get<int>();
    csrsm2Info_t info = (csrsm2Info_t)in->Get<size_t>();
    cusparseSolvePolicy_t policy = in->Get<cusparseSolvePolicy_t>();
    size_t * pBufferSize = new size_t;
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseDcsrsm2_bufferSizeExt(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrVal, csrRowPtr, csrColInd, B, ldb, info, policy, pBufferSize);
        out->Add<csrsm2Info_t>(info);
        out->AddMarshal<size_t*>(pBufferSize);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseDcsrsm2_bufferSizeExt Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Ccsrsm2_bufferSizeExt){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Ccsrsm2_bufferSizeExt"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    const int algo = in->Get<int>();
    cusparseOperation_t transA = in->Get<cusparseOperation_t>();
    cusparseOperation_t transB = in->Get<cusparseOperation_t>();
    const int m = in->Get<int>();
    const int nrhs = in->Get<int>();
    const int nnz = in->Get<int>();
    const cuComplex * alpha = in->Assign<cuComplex>();
    cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    cuComplex * csrVal = in->GetFromMarshal<cuComplex*>();
    const int * csrRowPtr = in->GetFromMarshal<int*>();
    const int * csrColInd = in->GetFromMarshal<int*>();
    cuComplex * B = in->GetFromMarshal<cuComplex*>();
    int ldb = in->Get<int>();
    csrsm2Info_t info = (csrsm2Info_t)in->Get<size_t>();
    cusparseSolvePolicy_t policy = in->Get<cusparseSolvePolicy_t>();
    size_t * pBufferSize = new size_t;
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseCcsrsm2_bufferSizeExt(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrVal, csrRowPtr, csrColInd, B, ldb, info, policy, pBufferSize);
        out->Add<csrsm2Info_t>(info);
        out->AddMarshal<size_t*>(pBufferSize);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseCcsrsm2_bufferSizeExt Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Zcsrsm2_bufferSizeExt){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Zcsrsm2_bufferSizeExt"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    const int algo = in->Get<int>();
    cusparseOperation_t transA = in->Get<cusparseOperation_t>();
    cusparseOperation_t transB = in->Get<cusparseOperation_t>();
    const int m = in->Get<int>();
    const int nrhs = in->Get<int>();
    const int nnz = in->Get<int>();
    const cuDoubleComplex * alpha = in->Assign<cuDoubleComplex>();
    cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    cuDoubleComplex * csrVal = in->GetFromMarshal<cuDoubleComplex*>();
    const int * csrRowPtr = in->GetFromMarshal<int*>();
    const int * csrColInd = in->GetFromMarshal<int*>();
    cuDoubleComplex * B = in->GetFromMarshal<cuDoubleComplex*>();
    int ldb = in->Get<int>();
    csrsm2Info_t info = (csrsm2Info_t)in->Get<size_t>();
    cusparseSolvePolicy_t policy = in->Get<cusparseSolvePolicy_t>();
    size_t * pBufferSize = new size_t;
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseZcsrsm2_bufferSizeExt(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrVal, csrRowPtr, csrColInd, B, ldb, info, policy, pBufferSize);
        out->Add<csrsm2Info_t>(info);
        out->AddMarshal<size_t*>(pBufferSize);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseZcsrsm2_bufferSizeExt Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Scsrsm2_analysis){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Scsrsm2_analysis"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    const int algo = in->Get<int>();
    cusparseOperation_t transA = in->Get<cusparseOperation_t>();
    cusparseOperation_t transB = in->Get<cusparseOperation_t>();
    const int m = in->Get<int>();
    const int nrhs = in->Get<int>();
    const int nnz = in->Get<int>();
    const float * alpha = in->Assign<float>();
    cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    float * csrVal = in->GetFromMarshal<float*>();
    const int * csrRowPtr = in->GetFromMarshal<int*>();
    const int * csrColInd = in->GetFromMarshal<int*>();
    float * B = in->GetFromMarshal<float*>();
    int ldb = in->Get<int>();
    csrsm2Info_t info = (csrsm2Info_t)in->Get<size_t>();
    cusparseSolvePolicy_t policy = in->Get<cusparseSolvePolicy_t>();
    void * pBuffer = in->GetFromMarshal<void*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseScsrsm2_analysis(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrVal, csrRowPtr, csrColInd, B, ldb, info, policy, pBuffer);
        out->AddMarshal<csrsm2Info_t>(info);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseScsrsm2_analysis Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Dcsrsm2_analysis){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Dcsrsm2_analysis"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    const int algo = in->Get<int>();
    cusparseOperation_t transA = in->Get<cusparseOperation_t>();
    cusparseOperation_t transB = in->Get<cusparseOperation_t>();
    const int m = in->Get<int>();
    const int nrhs = in->Get<int>();
    const int nnz = in->Get<int>();
    const double * alpha = in->Assign<double>();
    cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    double * csrVal = in->GetFromMarshal<double*>();
    const int * csrRowPtr = in->GetFromMarshal<int*>();
    const int * csrColInd = in->GetFromMarshal<int*>();
    double * B = in->GetFromMarshal<double*>();
    int ldb = in->Get<int>();
    csrsm2Info_t info = (csrsm2Info_t)in->Get<size_t>();
    cusparseSolvePolicy_t policy = in->Get<cusparseSolvePolicy_t>();
    void * pBuffer = in->GetFromMarshal<void*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseDcsrsm2_analysis(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrVal, csrRowPtr, csrColInd, B, ldb, info, policy, pBuffer);
        out->AddMarshal<csrsm2Info_t>(info);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseDcsrsm2_analysis Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Ccsrsm2_analysis){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Ccsrsm2_analysis"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    const int algo = in->Get<int>();
    cusparseOperation_t transA = in->Get<cusparseOperation_t>();
    cusparseOperation_t transB = in->Get<cusparseOperation_t>();
    const int m = in->Get<int>();
    const int nrhs = in->Get<int>();
    const int nnz = in->Get<int>();
    const cuComplex * alpha = in->Assign<cuComplex>();
    cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    cuComplex * csrVal = in->GetFromMarshal<cuComplex*>();
    const int * csrRowPtr = in->GetFromMarshal<int*>();
    const int * csrColInd = in->GetFromMarshal<int*>();
    cuComplex * B = in->GetFromMarshal<cuComplex*>();
    int ldb = in->Get<int>();
    csrsm2Info_t info = (csrsm2Info_t)in->Get<size_t>();
    cusparseSolvePolicy_t policy = in->Get<cusparseSolvePolicy_t>();
    void * pBuffer = in->GetFromMarshal<void*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseCcsrsm2_analysis(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrVal, csrRowPtr, csrColInd, B, ldb, info, policy, pBuffer);
        out->AddMarshal<csrsm2Info_t>(info);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseCcsrsm2_analysis Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Zcsrsm2_analysis){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Zcsrsm2_analysis"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    const int algo = in->Get<int>();
    cusparseOperation_t transA = in->Get<cusparseOperation_t>();
    cusparseOperation_t transB = in->Get<cusparseOperation_t>();
    const int m = in->Get<int>();
    const int nrhs = in->Get<int>();
    const int nnz = in->Get<int>();
    const cuDoubleComplex * alpha = in->Assign<cuDoubleComplex>();
    cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    cuDoubleComplex * csrVal = in->GetFromMarshal<cuDoubleComplex*>();
    const int * csrRowPtr = in->GetFromMarshal<int*>();
    const int * csrColInd = in->GetFromMarshal<int*>();
    cuDoubleComplex * B = in->GetFromMarshal<cuDoubleComplex*>();
    int ldb = in->Get<int>();
    csrsm2Info_t info = (csrsm2Info_t)in->Get<size_t>();
    cusparseSolvePolicy_t policy = in->Get<cusparseSolvePolicy_t>();
    void * pBuffer = in->GetFromMarshal<void*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseZcsrsm2_analysis(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrVal, csrRowPtr, csrColInd, B, ldb, info, policy, pBuffer);
        out->AddMarshal<csrsm2Info_t>(info);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseZcsrsm2_analysis Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Scsrsm2_solve){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Scsrsm2_solve"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    const int algo = in->Get<int>();
    cusparseOperation_t transA = in->Get<cusparseOperation_t>();
    cusparseOperation_t transB = in->Get<cusparseOperation_t>();
    const int m = in->Get<int>();
    const int nrhs = in->Get<int>();
    const int nnz = in->Get<int>();
    const float * alpha = in->Assign<float>();
    cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    float * csrVal = in->GetFromMarshal<float*>();
    const int * csrRowPtr = in->GetFromMarshal<int*>();
    const int * csrColInd = in->GetFromMarshal<int*>();
    float * B = in->GetFromMarshal<float*>();
    int ldb = in->Get<int>();
    csrsm2Info_t info = (csrsm2Info_t)in->Get<size_t>();
    cusparseSolvePolicy_t policy = in->Get<cusparseSolvePolicy_t>();
    void * pBuffer = in->GetFromMarshal<void*>();
    float * C = new float;
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseScsrsm2_solve(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrVal, csrRowPtr, csrColInd, B, ldb, info, policy, pBuffer);
        out->AddMarshal<float*>(C);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseScsrsm2_solve Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Dcsrsm2_solve){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Dcsrsm2_solve"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    const int algo = in->Get<int>();
    cusparseOperation_t transA = in->Get<cusparseOperation_t>();
    cusparseOperation_t transB = in->Get<cusparseOperation_t>();
    const int m = in->Get<int>();
    const int nrhs = in->Get<int>();
    const int nnz = in->Get<int>();
    const double * alpha = in->Assign<double>();
    cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    double * csrVal = in->GetFromMarshal<double*>();
    const int * csrRowPtr = in->GetFromMarshal<int*>();
    const int * csrColInd = in->GetFromMarshal<int*>();
    double * B = in->GetFromMarshal<double*>();
    int ldb = in->Get<int>();
    csrsm2Info_t info = (csrsm2Info_t)in->Get<size_t>();
    cusparseSolvePolicy_t policy = in->Get<cusparseSolvePolicy_t>();
    void * pBuffer = in->GetFromMarshal<void*>();
    double * C = new double;
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseDcsrsm2_solve(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrVal, csrRowPtr, csrColInd, B, ldb, info, policy, pBuffer);
        out->AddMarshal<double*>(C);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseDcsrsm2_solve Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Ccsrsm2_solve){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Ccsrsm2_solve"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    const int algo = in->Get<int>();
    cusparseOperation_t transA = in->Get<cusparseOperation_t>();
    cusparseOperation_t transB = in->Get<cusparseOperation_t>();
    const int m = in->Get<int>();
    const int nrhs = in->Get<int>();
    const int nnz = in->Get<int>();
    const cuComplex * alpha = in->Assign<cuComplex>();
    cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    cuComplex * csrVal = in->GetFromMarshal<cuComplex*>();
    const int * csrRowPtr = in->GetFromMarshal<int*>();
    const int * csrColInd = in->GetFromMarshal<int*>();
    cuComplex * B = in->GetFromMarshal<cuComplex*>();
    int ldb = in->Get<int>();
    csrsm2Info_t info = (csrsm2Info_t)in->Get<size_t>();
    cusparseSolvePolicy_t policy = in->Get<cusparseSolvePolicy_t>();
    void * pBuffer = in->GetFromMarshal<void*>();
    cuComplex * C = new cuComplex;
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseCcsrsm2_solve(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrVal, csrRowPtr, csrColInd, B, ldb, info, policy, pBuffer);
        out->AddMarshal<cuComplex*>(C);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseCcsrsm2_solve Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Zcsrsm2_solve){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Zcsrsm2_solve"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    const int algo = in->Get<int>();
    cusparseOperation_t transA = in->Get<cusparseOperation_t>();
    cusparseOperation_t transB = in->Get<cusparseOperation_t>();
    const int m = in->Get<int>();
    const int nrhs = in->Get<int>();
    const int nnz = in->Get<int>();
    const cuDoubleComplex * alpha = in->Assign<cuDoubleComplex>();
    cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    cuDoubleComplex * csrVal = in->GetFromMarshal<cuDoubleComplex*>();
    const int * csrRowPtr = in->GetFromMarshal<int*>();
    const int * csrColInd = in->GetFromMarshal<int*>();
    cuDoubleComplex * B = in->GetFromMarshal<cuDoubleComplex*>();
    int ldb = in->Get<int>();
    csrsm2Info_t info = (csrsm2Info_t)in->Get<size_t>();
    cusparseSolvePolicy_t policy = in->Get<cusparseSolvePolicy_t>();
    void * pBuffer = in->GetFromMarshal<void*>();
    cuDoubleComplex * C = new cuDoubleComplex;
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseZcsrsm2_solve(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrVal, csrRowPtr, csrColInd, B, ldb, info, policy, pBuffer);
        out->AddMarshal<cuDoubleComplex*>(C);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseZcsrsm2_solve Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Xcsrsm2_zeroPivot){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Xcbsrsm2_zeroPivot"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    csrsm2Info_t info = (csrsm2Info_t)in->Get<size_t>();
    int * position = new int;
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseXcsrsm2_zeroPivot(handle, info, position);
        out->AddMarshal<int*>(position);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseXcsrsm2_zeroPivot Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Sgemmi){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Sgemmi"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    const int m = in->Get<int>();
    const int n = in->Get<int>();
    const int k = in->Get<int>();
    const int nnz = in->Get<int>();
    const float * alpha = in->Assign<float>();
    const float * A = in->Get<float*>();
    const int lda = in->Get<int>();
    const float * cscValB = in->Get<float*>();
    const int * cscColPtrB = in->Get<int*>();
    const int * cscRowIndB = in->Get<int*>();
    const float * beta = in->Assign<float>();
    float * C = in->Get<float*>();
    const int ldc = in->Get<int>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseSgemmi(handle, m, n, k, nnz, alpha, A, lda, cscValB, cscColPtrB, cscRowIndB, beta, C, ldc);
        out->AddMarshal<float*>(C);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseSgemmi Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Dgemmi){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Dgemmi"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    const int m = in->Get<int>();
    const int n = in->Get<int>();
    const int k = in->Get<int>();
    const int nnz = in->Get<int>();
    const double * alpha = in->Assign<double>();
    const double * A = in->Get<double*>();
    const int lda = in->Get<int>();
    const double * cscValB = in->Get<double*>();
    const int * cscColPtrB = in->Get<int*>();
    const int * cscRowIndB = in->Get<int*>();
    const double * beta = in->Assign<double>();
    double * C = in->Get<double*>();
    const int ldc = in->Get<int>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseDgemmi(handle, m, n, k, nnz, alpha, A, lda, cscValB, cscColPtrB, cscRowIndB, beta, C, ldc);
        out->AddMarshal<double*>(C);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseDgemmi Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Cgemmi){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Cgemmi"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    const int m = in->Get<int>();
    const int n = in->Get<int>();
    const int k = in->Get<int>();
    const int nnz = in->Get<int>();
    const cuComplex * alpha = in->Assign<cuComplex>();
    const cuComplex * A = in->Get<cuComplex*>();
    const int lda = in->Get<int>();
    const cuComplex * cscValB = in->Get<cuComplex*>();
    const int * cscColPtrB = in->Get<int*>();
    const int * cscRowIndB = in->Get<int*>();
    const cuComplex * beta = in->Assign<cuComplex>();
    cuComplex * C = in->Get<cuComplex*>();
    const int ldc = in->Get<int>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseCgemmi(handle, m, n, k, nnz, alpha, A, lda, cscValB, cscColPtrB, cscRowIndB, beta, C, ldc);
        out->AddMarshal<cuComplex*>(C);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseCgemmi Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Zgemmi){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Zgemmi"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    const int m = in->Get<int>();
    const int n = in->Get<int>();
    const int k = in->Get<int>();
    const int nnz = in->Get<int>();
    const cuDoubleComplex * alpha = in->Assign<cuDoubleComplex>();
    const cuDoubleComplex * A = in->Get<cuDoubleComplex*>();
    const int lda = in->Get<int>();
    const cuDoubleComplex * cscValB = in->Get<cuDoubleComplex*>();
    const int * cscColPtrB = in->Get<int*>();
    const int * cscRowIndB = in->Get<int*>();
    const cuDoubleComplex * beta = in->Assign<cuDoubleComplex>();
    cuDoubleComplex * C = in->Get<cuDoubleComplex*>();
    const int ldc = in->Get<int>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseZgemmi(handle, m, n, k, nnz, alpha, A, lda, cscValB, cscColPtrB, cscRowIndB, beta, C, ldc);
        out->AddMarshal<cuDoubleComplex*>(C);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseZgemmi Executed");
    return std::make_shared<Result>(cs,out);
}

#ifndef CUSPARSE_VERSION
#error CUSPARSE_VERSION not defined
#endif
