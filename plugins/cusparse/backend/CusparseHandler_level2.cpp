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

CUSPARSE_ROUTINE_HANDLER(Sbsrmv){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Sbsrmv"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    const cusparseDirection_t dir = in->Get<cusparseDirection_t>();
    const cusparseOperation_t trans = in->Get<cusparseOperation_t>();
    const int mb = in->Get<int>();
    const int nb = in->Get<int>();
    const int nnzb = in->Get<int>();
    const float * alpha = in->Assign<float>();
    const cusparseMatDescr_t descr = in->Get<cusparseMatDescr_t>();
    const float * bsrVal = in->GetFromMarshal<float*>();
    const int * bsrRowPtr = in->GetFromMarshal<int*>();
    const int * bsrColInd = in->GetFromMarshal<int*>();
    const int blockDim = in->Get<int>();
    float * x = in->GetFromMarshal<float*>();
    const float * beta = in->Assign<float>();
    float * y = in->GetFromMarshal<float*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseSbsrmv(handle, dir, trans, mb, nb, nnzb, alpha, descr, bsrVal, bsrRowPtr, bsrColInd, blockDim, x, beta, y);
        out->AddMarshal<float*>(y);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseSbsrmv Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Dbsrmv){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Dbsrmv"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    const cusparseDirection_t dir = in->Get<cusparseDirection_t>();
    const cusparseOperation_t trans = in->Get<cusparseOperation_t>();
    const int mb = in->Get<int>();
    const int nb = in->Get<int>();
    const int nnzb = in->Get<int>();
    const double * alpha = in->Assign<double>();
    const cusparseMatDescr_t descr = in->Get<cusparseMatDescr_t>();
    const double * bsrVal = in->GetFromMarshal<double*>();
    const int * bsrRowPtr = in->GetFromMarshal<int*>();
    const int * bsrColInd = in->GetFromMarshal<int*>();
    const int blockDim = in->Get<int>();
    double * x = in->GetFromMarshal<double*>();
    const double * beta = in->Assign<double>();
    double * y = in->GetFromMarshal<double*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseDbsrmv(handle, dir, trans, mb, nb, nnzb, alpha, descr, bsrVal, bsrRowPtr, bsrColInd, blockDim, x, beta, y);
        out->AddMarshal<double*>(y);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseDbsrmv Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Cbsrmv){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Cbsrmv"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    const cusparseDirection_t dir = in->Get<cusparseDirection_t>();
    const cusparseOperation_t trans = in->Get<cusparseOperation_t>();
    const int mb = in->Get<int>();
    const int nb = in->Get<int>();
    const int nnzb = in->Get<int>();
    const cuComplex * alpha = in->Assign<cuComplex>();
    const cusparseMatDescr_t descr = in->Get<cusparseMatDescr_t>();
    const cuComplex * bsrVal = in->GetFromMarshal<cuComplex*>();
    const int * bsrRowPtr = in->GetFromMarshal<int*>();
    const int * bsrColInd = in->GetFromMarshal<int*>();
    const int blockDim = in->Get<int>();
    cuComplex * x = in->GetFromMarshal<cuComplex*>();
    const cuComplex * beta = in->Assign<cuComplex>();
    cuComplex * y = in->GetFromMarshal<cuComplex*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseCbsrmv(handle, dir, trans, mb, nb, nnzb, alpha, descr, bsrVal, bsrRowPtr, bsrColInd, blockDim, x, beta, y);
        out->AddMarshal<cuComplex*>(y);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseCbsrmv Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Zbsrmv){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Zbsrmv"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    const cusparseDirection_t dir = in->Get<cusparseDirection_t>();
    const cusparseOperation_t trans = in->Get<cusparseOperation_t>();
    const int mb = in->Get<int>();
    const int nb = in->Get<int>();
    const int nnzb = in->Get<int>();
    const cuDoubleComplex * alpha = in->Assign<cuDoubleComplex>();
    const cusparseMatDescr_t descr = in->Get<cusparseMatDescr_t>();
    const cuDoubleComplex * bsrVal = in->GetFromMarshal<cuDoubleComplex*>();
    const int * bsrRowPtr = in->GetFromMarshal<int*>();
    const int * bsrColInd = in->GetFromMarshal<int*>();
    const int blockDim = in->Get<int>();
    cuDoubleComplex * x = in->GetFromMarshal<cuDoubleComplex*>();
    const cuDoubleComplex * beta = in->Assign<cuDoubleComplex>();
    cuDoubleComplex * y = in->GetFromMarshal<cuDoubleComplex*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseZbsrmv(handle, dir, trans, mb, nb, nnzb, alpha, descr, bsrVal, bsrRowPtr, bsrColInd, blockDim, x, beta, y);
        out->AddMarshal<cuDoubleComplex*>(y);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseZbsrmv Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Sbsrxmv){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Sbsrxmv"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    const cusparseDirection_t dir = in->Get<cusparseDirection_t>();
    const cusparseOperation_t trans = in->Get<cusparseOperation_t>();
    const int sizeOfMask = in->Get<int>();
    const int mb = in->Get<int>();
    const int nb = in->Get<int>();
    const int nnzb = in->Get<int>();
    const float * alpha = in->Assign<float>();
    const cusparseMatDescr_t descr = in->Get<cusparseMatDescr_t>();
    const float * bsrVal = in->GetFromMarshal<float*>();
    const int * bsrMaskPtr = in->GetFromMarshal<int*>();
    const int * bsrRowPtr = in->GetFromMarshal<int*>();
    const int * bsrRowPtrEnd = in->GetFromMarshal<int*>();
    const int * bsrColInd = in->GetFromMarshal<int*>();
    const int blockDim = in->Get<int>();
    float * x = in->GetFromMarshal<float*>();
    const float * beta = in->Assign<float>();
    float * y = in->GetFromMarshal<float*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseSbsrxmv(handle, dir, trans, sizeOfMask, mb, nb, nnzb, alpha, descr, bsrVal, bsrMaskPtr, bsrRowPtr, bsrRowPtrEnd, bsrColInd, blockDim, x, beta, y);
        out->AddMarshal<float*>(y);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseSbsrxmv Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Dbsrxmv){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Dbsrxmv"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    const cusparseDirection_t dir = in->Get<cusparseDirection_t>();
    const cusparseOperation_t trans = in->Get<cusparseOperation_t>();
    const int sizeOfMask = in->Get<int>();
    const int mb = in->Get<int>();
    const int nb = in->Get<int>();
    const int nnzb = in->Get<int>();
    const double * alpha = in->Assign<double>();
    const cusparseMatDescr_t descr = in->Get<cusparseMatDescr_t>();
    const double * bsrVal = in->GetFromMarshal<double*>();
    const int * bsrMaskPtr = in->GetFromMarshal<int*>();
    const int * bsrRowPtr = in->GetFromMarshal<int*>();
    const int * bsrRowPtrEnd = in->GetFromMarshal<int*>();
    const int * bsrColInd = in->GetFromMarshal<int*>();
    const int blockDim = in->Get<int>();
    double * x = in->GetFromMarshal<double*>();
    const double * beta = in->Assign<double>();
    double * y = in->GetFromMarshal<double*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseDbsrxmv(handle, dir, trans, sizeOfMask, mb, nb, nnzb, alpha, descr, bsrVal, bsrMaskPtr, bsrRowPtr, bsrRowPtrEnd, bsrColInd, blockDim, x, beta, y);
        out->AddMarshal<double*>(y);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseDbsrxmv Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Cbsrxmv){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Cbsrxmv"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    const cusparseDirection_t dir = in->Get<cusparseDirection_t>();
    const cusparseOperation_t trans = in->Get<cusparseOperation_t>();
    const int sizeOfMask = in->Get<int>();
    const int mb = in->Get<int>();
    const int nb = in->Get<int>();
    const int nnzb = in->Get<int>();
    const cuComplex * alpha = in->Assign<cuComplex>();
    const cusparseMatDescr_t descr = in->Get<cusparseMatDescr_t>();
    const cuComplex * bsrVal = in->GetFromMarshal<cuComplex*>();
    const int * bsrMaskPtr = in->GetFromMarshal<int*>();
    const int * bsrRowPtr = in->GetFromMarshal<int*>();
    const int * bsrRowPtrEnd = in->GetFromMarshal<int*>();
    const int * bsrColInd = in->GetFromMarshal<int*>();
    const int blockDim = in->Get<int>();
    cuComplex * x = in->GetFromMarshal<cuComplex*>();
    const cuComplex * beta = in->Assign<cuComplex>();
    cuComplex * y = in->GetFromMarshal<cuComplex*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseCbsrxmv(handle, dir, trans, sizeOfMask, mb, nb, nnzb, alpha, descr, bsrVal, bsrMaskPtr, bsrRowPtr, bsrRowPtrEnd, bsrColInd, blockDim, x, beta, y);
        out->AddMarshal<cuComplex*>(y);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseCbsrxmv Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Zbsrxmv){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Zbsrxmv"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    const cusparseDirection_t dir = in->Get<cusparseDirection_t>();
    const cusparseOperation_t trans = in->Get<cusparseOperation_t>();
    const int sizeOfMask = in->Get<int>();
    const int mb = in->Get<int>();
    const int nb = in->Get<int>();
    const int nnzb = in->Get<int>();
    const cuDoubleComplex * alpha = in->Assign<cuDoubleComplex>();
    const cusparseMatDescr_t descr = in->Get<cusparseMatDescr_t>();
    const cuDoubleComplex * bsrVal = in->GetFromMarshal<cuDoubleComplex*>();
    const int * bsrMaskPtr = in->GetFromMarshal<int*>();
    const int * bsrRowPtr = in->GetFromMarshal<int*>();
    const int * bsrRowPtrEnd = in->GetFromMarshal<int*>();
    const int * bsrColInd = in->GetFromMarshal<int*>();
    const int blockDim = in->Get<int>();
    cuDoubleComplex * x = in->GetFromMarshal<cuDoubleComplex*>();
    const cuDoubleComplex * beta = in->Assign<cuDoubleComplex>();
    cuDoubleComplex * y = in->GetFromMarshal<cuDoubleComplex*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseZbsrxmv(handle, dir, trans, sizeOfMask, mb, nb, nnzb, alpha, descr, bsrVal, bsrMaskPtr, bsrRowPtr, bsrRowPtrEnd, bsrColInd, blockDim, x, beta, y);
        out->AddMarshal<cuDoubleComplex*>(y);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseZbsrxmv Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Sbsrsv2_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Sbsrsv2_bufferSize"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    cusparseDirection_t dir = in->Get<cusparseDirection_t>();
    cusparseOperation_t trans = in->Get<cusparseOperation_t>();
    const int mb = in->Get<int>();
    const int nnzb = in->Get<int>();
    cusparseMatDescr_t descr = in->Get<cusparseMatDescr_t>();
    float * bsrVal = in->GetFromMarshal<float*>();
    const int * bsrRowPtr = in->GetFromMarshal<int*>();
    const int * bsrColInd = in->GetFromMarshal<int*>();
    int blockDim = in->Get<int>();
    bsrsv2Info_t info = (bsrsv2Info_t)in->Get<size_t>();
    int * pBufferSizeInBytes = in->GetFromMarshal<int*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseSbsrsv2_bufferSize(handle, dir, trans, mb, nnzb, descr, bsrVal, bsrRowPtr, bsrColInd, blockDim, info, pBufferSizeInBytes);
        out->Add<bsrsv2Info_t>(info);
        out->AddMarshal<int*>(pBufferSizeInBytes);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseSbsrsv2_bufferSize Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Dbsrsv2_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Dbsrsv2_bufferSize"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    cusparseDirection_t dir = in->Get<cusparseDirection_t>();
    cusparseOperation_t trans = in->Get<cusparseOperation_t>();
    const int mb = in->Get<int>();
    const int nnzb = in->Get<int>();
    cusparseMatDescr_t descr = in->Get<cusparseMatDescr_t>();
    double * bsrVal = in->GetFromMarshal<double*>();
    const int * bsrRowPtr = in->GetFromMarshal<int*>();
    const int * bsrColInd = in->GetFromMarshal<int*>();
    int blockDim = in->Get<int>();
    bsrsv2Info_t info = (bsrsv2Info_t)in->Get<size_t>();
    int * pBufferSizeInBytes = in->GetFromMarshal<int*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseDbsrsv2_bufferSize(handle, dir, trans, mb, nnzb, descr, bsrVal, bsrRowPtr, bsrColInd, blockDim, info, pBufferSizeInBytes);
        out->Add<bsrsv2Info_t>(info);
        out->AddMarshal<int*>(pBufferSizeInBytes);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseDbsrsv2_bufferSize Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Cbsrsv2_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Cbsrsv2_bufferSize"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    cusparseDirection_t dir = in->Get<cusparseDirection_t>();
    cusparseOperation_t trans = in->Get<cusparseOperation_t>();
    const int mb = in->Get<int>();
    const int nnzb = in->Get<int>();
    cusparseMatDescr_t descr = in->Get<cusparseMatDescr_t>();
    cuComplex * bsrVal = in->GetFromMarshal<cuComplex*>();
    const int * bsrRowPtr = in->GetFromMarshal<int*>();
    const int * bsrColInd = in->GetFromMarshal<int*>();
    int blockDim = in->Get<int>();
    bsrsv2Info_t info = (bsrsv2Info_t)in->Get<size_t>();
    int * pBufferSizeInBytes = in->GetFromMarshal<int*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseCbsrsv2_bufferSize(handle, dir, trans, mb, nnzb, descr, bsrVal, bsrRowPtr, bsrColInd, blockDim, info, pBufferSizeInBytes);
        out->Add<bsrsv2Info_t>(info);
        out->AddMarshal<int*>(pBufferSizeInBytes);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseCbsrsv2_bufferSize Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Zbsrsv2_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Zbsrsv2_bufferSize"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    cusparseDirection_t dir = in->Get<cusparseDirection_t>();
    cusparseOperation_t trans = in->Get<cusparseOperation_t>();
    const int mb = in->Get<int>();
    const int nnzb = in->Get<int>();
    cusparseMatDescr_t descr = in->Get<cusparseMatDescr_t>();
    cuDoubleComplex * bsrVal = in->GetFromMarshal<cuDoubleComplex*>();
    const int * bsrRowPtr = in->GetFromMarshal<int*>();
    const int * bsrColInd = in->GetFromMarshal<int*>();
    int blockDim = in->Get<int>();
    bsrsv2Info_t info = (bsrsv2Info_t)in->Get<size_t>();
    int * pBufferSizeInBytes = in->GetFromMarshal<int*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseZbsrsv2_bufferSize(handle, dir, trans, mb, nnzb, descr, bsrVal, bsrRowPtr, bsrColInd, blockDim, info, pBufferSizeInBytes);
        out->Add<bsrsv2Info_t>(info);
        out->AddMarshal<int*>(pBufferSizeInBytes);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseZbsrsv2_bufferSize Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Sbsrsv2_analysis){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Sbsrsv2_analysis"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    cusparseDirection_t dir = in->Get<cusparseDirection_t>();
    cusparseOperation_t trans = in->Get<cusparseOperation_t>();
    int mb = in->Get<int>();
    int nnzb = in->Get<int>();
    cusparseMatDescr_t descr = (cusparseMatDescr_t)in->Get<size_t>();
    float * bsrVal = in->GetFromMarshal<float*>();
    int * bsrRowPtr = in->GetFromMarshal<int*>();
    int * bsrColInd = in->GetFromMarshal<int*>();
    int blockDim = in->Get<int>();
    bsrsv2Info_t info = (bsrsv2Info_t)in->Get<size_t>();
    cusparseSolvePolicy_t policy = in->Get<cusparseSolvePolicy_t>();
    void * pBuffer = in->GetFromMarshal<void*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseSbsrsv2_analysis(handle, dir, trans, mb, nnzb, descr, bsrVal, bsrRowPtr, bsrColInd, blockDim, info, policy, pBuffer);
        out->AddMarshal<bsrsv2Info_t>(info);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseSbsrsv2_analysis Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Dbsrsv2_analysis){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Dbsrsv2_analysis"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    const cusparseDirection_t dir = in->Get<cusparseDirection_t>();
    const cusparseOperation_t trans = in->Get<cusparseOperation_t>();
    const int mb = in->Get<int>();
    const int nnzb = in->Get<int>();
    const cusparseMatDescr_t descr = in->Get<cusparseMatDescr_t>();
    double * bsrVal = in->GetFromMarshal<double*>();
    const int * bsrRowPtr = in->GetFromMarshal<int*>();
    const int * bsrColInd = in->GetFromMarshal<int*>();
    const int blockDim = in->Get<int>();
    bsrsv2Info_t info = in->Get<bsrsv2Info_t>();
    const cusparseSolvePolicy_t policy = in->Get<cusparseSolvePolicy_t>();
    void * pBuffer = in->GetFromMarshal<void*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseDbsrsv2_analysis(handle, dir, trans, mb, nnzb, descr, bsrVal, bsrRowPtr, bsrColInd, blockDim, info, policy, pBuffer);
        out->AddMarshal<bsrsv2Info_t>(info);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseDbsrsv2_analysis Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Cbsrsv2_analysis){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Cbsrsv2_analysis"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    const cusparseDirection_t dir = in->Get<cusparseDirection_t>();
    const cusparseOperation_t trans = in->Get<cusparseOperation_t>();
    const int mb = in->Get<int>();
    const int nnzb = in->Get<int>();
    const cusparseMatDescr_t descr = in->Get<cusparseMatDescr_t>();
    cuComplex * bsrVal = in->GetFromMarshal<cuComplex*>();
    const int * bsrRowPtr = in->GetFromMarshal<int*>();
    const int * bsrColInd = in->GetFromMarshal<int*>();
    const int blockDim = in->Get<int>();
    bsrsv2Info_t info = in->Get<bsrsv2Info_t>();
    const cusparseSolvePolicy_t policy = in->Get<cusparseSolvePolicy_t>();
    void * pBuffer = in->GetFromMarshal<void*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseCbsrsv2_analysis(handle, dir, trans, mb, nnzb, descr, bsrVal, bsrRowPtr, bsrColInd, blockDim, info, policy, pBuffer);
        out->AddMarshal<bsrsv2Info_t>(info);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseCbsrsv2_analysis Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Zbsrsv2_analysis){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Zbsrsv2_analysis"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    const cusparseDirection_t dir = in->Get<cusparseDirection_t>();
    const cusparseOperation_t trans = in->Get<cusparseOperation_t>();
    const int mb = in->Get<int>();
    const int nnzb = in->Get<int>();
    const cusparseMatDescr_t descr = in->Get<cusparseMatDescr_t>();
    cuDoubleComplex * bsrVal = in->GetFromMarshal<cuDoubleComplex*>();
    const int * bsrRowPtr = in->GetFromMarshal<int*>();
    const int * bsrColInd = in->GetFromMarshal<int*>();
    const int blockDim = in->Get<int>();
    bsrsv2Info_t info = in->Get<bsrsv2Info_t>();
    const cusparseSolvePolicy_t policy = in->Get<cusparseSolvePolicy_t>();
    void * pBuffer = in->GetFromMarshal<void*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseZbsrsv2_analysis(handle, dir, trans, mb, nnzb, descr, bsrVal, bsrRowPtr, bsrColInd, blockDim, info, policy, pBuffer);
        out->AddMarshal<bsrsv2Info_t>(info);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseZbsrsv2_analysis Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Sbsrsv2_solve){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Sbsrsv2_solve"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    cusparseDirection_t dir = in->Get<cusparseDirection_t>();
    cusparseOperation_t trans = in->Get<cusparseOperation_t>();
    int mb = in->Get<int>();
    int nnzb = in->Get<int>();
    const float * alpha = in->Assign<float>();
    cusparseMatDescr_t descr = (cusparseMatDescr_t)in->Get<size_t>();
    float * bsrVal = in->GetFromMarshal<float*>();
    int * bsrRowPtr = in->GetFromMarshal<int*>();
    int * bsrColInd = in->GetFromMarshal<int*>();
    int blockDim = in->Get<int>();
    bsrsv2Info_t info = (bsrsv2Info_t)in->Get<size_t>();
    float * x = in->GetFromMarshal<float*>();
    float * y = in->GetFromMarshal<float*>();
    cusparseSolvePolicy_t policy = in->Get<cusparseSolvePolicy_t>();
    void * pBuffer = in->GetFromMarshal<void*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseSbsrsv2_solve(handle, dir, trans, mb, nnzb, alpha, descr, bsrVal, bsrRowPtr, bsrColInd, blockDim, info, x, y, policy, pBuffer);
        out->AddMarshal<float*>(y);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseSbsrsv2_solve Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Dbsrsv2_solve){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Dbsrsv2_solve"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    cusparseDirection_t dir = in->Get<cusparseDirection_t>();
    cusparseOperation_t trans = in->Get<cusparseOperation_t>();
    int mb = in->Get<int>();
    int nnzb = in->Get<int>();
    const double * alpha = in->Assign<double>();
    cusparseMatDescr_t descr = (cusparseMatDescr_t)in->Get<size_t>();
    double * bsrVal = in->GetFromMarshal<double*>();
    int * bsrRowPtr = in->GetFromMarshal<int*>();
    int * bsrColInd = in->GetFromMarshal<int*>();
    int blockDim = in->Get<int>();
    bsrsv2Info_t info = (bsrsv2Info_t)in->Get<size_t>();
    double * x = in->GetFromMarshal<double*>();
    double * y = in->GetFromMarshal<double*>();
    cusparseSolvePolicy_t policy = in->Get<cusparseSolvePolicy_t>();
    void * pBuffer = in->GetFromMarshal<void*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseDbsrsv2_solve(handle, dir, trans, mb, nnzb, alpha, descr, bsrVal, bsrRowPtr, bsrColInd, blockDim, info, x, y, policy, pBuffer);
        out->AddMarshal<double*>(y);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseDbsrsv2_solve Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Cbsrsv2_solve){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Cbsrsv2_solve"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    cusparseDirection_t dir = in->Get<cusparseDirection_t>();
    cusparseOperation_t trans = in->Get<cusparseOperation_t>();
    int mb = in->Get<int>();
    int nnzb = in->Get<int>();
    const cuComplex * alpha = in->Assign<cuComplex>();
    cusparseMatDescr_t descr = (cusparseMatDescr_t)in->Get<size_t>();
    cuComplex * bsrVal = in->GetFromMarshal<cuComplex*>();
    int * bsrRowPtr = in->GetFromMarshal<int*>();
    int * bsrColInd = in->GetFromMarshal<int*>();
    int blockDim = in->Get<int>();
    bsrsv2Info_t info = (bsrsv2Info_t)in->Get<size_t>();
    cuComplex * x = in->GetFromMarshal<cuComplex*>();
    cuComplex * y = in->GetFromMarshal<cuComplex*>();
    cusparseSolvePolicy_t policy = in->Get<cusparseSolvePolicy_t>();
    void * pBuffer = in->GetFromMarshal<void*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseCbsrsv2_solve(handle, dir, trans, mb, nnzb, alpha, descr, bsrVal, bsrRowPtr, bsrColInd, blockDim, info, x, y, policy, pBuffer);
        out->AddMarshal<cuComplex*>(y);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseCbsrsv2_solve Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Zbsrsv2_solve){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Zbsrsv2_solve"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    cusparseDirection_t dir = in->Get<cusparseDirection_t>();
    cusparseOperation_t trans = in->Get<cusparseOperation_t>();
    int mb = in->Get<int>();
    int nnzb = in->Get<int>();
    const cuDoubleComplex * alpha = in->Assign<cuDoubleComplex>();
    cusparseMatDescr_t descr = (cusparseMatDescr_t)in->Get<size_t>();
    cuDoubleComplex * bsrVal = in->GetFromMarshal<cuDoubleComplex*>();
    int * bsrRowPtr = in->GetFromMarshal<int*>();
    int * bsrColInd = in->GetFromMarshal<int*>();
    int blockDim = in->Get<int>();
    bsrsv2Info_t info = (bsrsv2Info_t)in->Get<size_t>();
    cuDoubleComplex * x = in->GetFromMarshal<cuDoubleComplex*>();
    cuDoubleComplex * y = in->GetFromMarshal<cuDoubleComplex*>();
    cusparseSolvePolicy_t policy = in->Get<cusparseSolvePolicy_t>();
    void * pBuffer = in->GetFromMarshal<void*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseZbsrsv2_solve(handle, dir, trans, mb, nnzb, alpha, descr, bsrVal, bsrRowPtr, bsrColInd, blockDim, info, x, y, policy, pBuffer);
        out->AddMarshal<cuDoubleComplex*>(y);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseZbsrsv2_solve Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Xbsrsv2_zeroPivot){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Xbsrsv2_zeroPivot"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    bsrsv2Info_t info = (bsrsv2Info_t)in->Get<size_t>();
    int * position = in->GetFromMarshal<int*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseXbsrsv2_zeroPivot(handle, info, position);
        out->AddMarshal<int*>(position);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseXbsrsv2_zeroPivot Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(CsrmvEx_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("CsrmvEx_bufferSize"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    cusparseAlgMode_t alg = in->Get<cusparseAlgMode_t>();
    cusparseOperation_t transA = in->Get<cusparseOperation_t>();
    const int m = in->Get<int>();
    const int n = in->Get<int>();
    const int nnz = in->Get<int>();
    cudaDataType alphatype = in->Get<cudaDataType>();
    void* alpha;
    if (alphatype == CUDA_R_32F) {
        // float
        float alphaFloat = in->Get<float>();
        alpha = &alphaFloat;
    } else if (alphatype == CUDA_R_64F) {
        // double
        double alphaDouble = in->Get<double>();
        alpha = &alphaDouble;
    } else if (alphatype == CUDA_C_32F) {
        // cuComplex
        cuComplex alphaCuComplex = in->Get<cuComplex>();
        alpha = &alphaCuComplex;
    } else if (alphatype == CUDA_C_64F) {
        // cuDoubleComplex
        cuDoubleComplex alphaCuDoubleComplex = in->Get<cuDoubleComplex>();
        alpha = &alphaCuDoubleComplex;
    } else {
        throw "Type not supported by GVirtus!";
    }
    cusparseMatDescr_t descrA = (cusparseMatDescr_t) in->Get<size_t>();
    void * csrValA = in->GetFromMarshal<void*>();
    cudaDataType csrValAtype = in->Get<cudaDataType>();
    int * csrRowPtrA = in->GetFromMarshal<int*>();
    int * csrColIndA = in->GetFromMarshal<int*>();
    void * x = in->Get<void*>();
    cudaDataType xtype = in->Get<cudaDataType>();
    cudaDataType betatype = in->Get<cudaDataType>();
    void* beta;
    if (betatype == CUDA_R_32F) {
        // float
        float betaFloat = in->Get<float>();
        beta = &betaFloat;
    } else if (betatype == CUDA_R_64F) {
        // double
        double betaDouble = in->Get<double>();
        beta = &betaDouble;
    } else if (betatype == CUDA_C_32F) {
        // cuComplex
        cuComplex betaCuComplex = in->Get<cuComplex>();
        beta = &betaCuComplex;
    } else if (betatype == CUDA_C_64F) {
        // cuDoubleComplex
        cuDoubleComplex betaCuDoubleComplex = in->Get<cuDoubleComplex>();
        beta = &betaCuDoubleComplex;
    } else {
        throw "Type not supported by GVirtus!";
    }
    void * y = in->Get<void*>();
    cudaDataType ytype = in->Get<cudaDataType>();
    cudaDataType executiontype = in->Get<cudaDataType>();
    size_t * bufferSizeInBytes = new size_t;
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseCsrmvEx_bufferSize(handle, alg, transA, m, n, nnz, alpha, alphatype, descrA, csrValA, csrValAtype, csrRowPtrA, csrColIndA, x, xtype, beta, betatype, y, ytype, executiontype, bufferSizeInBytes);
        out->Add<size_t>(bufferSizeInBytes);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseCsrmvEx_bufferSize Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(CsrmvEx){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("CsrmvEx"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    cusparseAlgMode_t alg = in->Get<cusparseAlgMode_t>();
    cusparseOperation_t transA = in->Get<cusparseOperation_t>();
    const int m = in->Get<int>();
    const int n = in->Get<int>();
    const int nnz = in->Get<int>();
    cudaDataType alphatype = in->Get<cudaDataType>();
    void* alpha;
    if (alphatype == CUDA_R_32F) {
        // float
        float alphaFloat = in->Get<float>();
        alpha = &alphaFloat;
    } else if (alphatype == CUDA_R_64F) {
        // double
        double alphaDouble = in->Get<double>();
        alpha = &alphaDouble;
    } else if (alphatype == CUDA_C_32F) {
        // cuComplex
        cuComplex alphaCuComplex = in->Get<cuComplex>();
        alpha = &alphaCuComplex;
    } else if (alphatype == CUDA_C_64F) {
        // cuDoubleComplex
        cuDoubleComplex alphaCuDoubleComplex = in->Get<cuDoubleComplex>();
        alpha = &alphaCuDoubleComplex;
    } else {
        throw "Type not supported by GVirtus!";
    }
    cusparseMatDescr_t descrA = (cusparseMatDescr_t) in->Get<size_t>();
    void * csrValA = in->GetFromMarshal<void*>();
    cudaDataType csrValAtype = in->Get<cudaDataType>();
    int * csrRowPtrA = in->GetFromMarshal<int*>();
    int * csrColIndA = in->GetFromMarshal<int*>();
    void * x = in->GetFromMarshal<void*>();
    cudaDataType xtype = in->Get<cudaDataType>();
    cudaDataType betatype = in->Get<cudaDataType>();
    void* beta;
    if (betatype == CUDA_R_32F) {
        // float
        float betaFloat = in->Get<float>();
        beta = &betaFloat;
    } else if (betatype == CUDA_R_64F) {
        // double
        double betaDouble = in->Get<double>();
        beta = &betaDouble;
    } else if (betatype == CUDA_C_32F) {
        // cuComplex
        cuComplex betaCuComplex = in->Get<cuComplex>();
        beta = &betaCuComplex;
    } else if (betatype == CUDA_C_64F) {
        // cuDoubleComplex
        cuDoubleComplex betaCuDoubleComplex = in->Get<cuDoubleComplex>();
        beta = &betaCuDoubleComplex;
    } else {
        throw "Type not supported by GVirtus!";
    }
    void * y = in->GetFromMarshal<void*>();
    cudaDataType ytype = in->Get<cudaDataType>();
    cudaDataType executiontype = in->Get<cudaDataType>();
    void* buffer = in->Get<void*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseCsrmvEx(handle, alg, transA, m, n, nnz, alpha, alphatype, descrA, csrValA, csrValAtype, csrRowPtrA, csrColIndA, x, xtype, beta, betatype, y, ytype, executiontype, buffer);
        out->Add<void*>(y);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseCsrmvEx Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Scsrsv2_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Scsrsv2_bufferSize"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    cusparseOperation_t trans = in->Get<cusparseOperation_t>();
    const int m = in->Get<int>();
    const int nnz = in->Get<int>();
    cusparseMatDescr_t descr = in->Get<cusparseMatDescr_t>();
    float * csrVal = in->GetFromMarshal<float*>();
    const int * csrRowPtr = in->GetFromMarshal<int*>();
    const int * csrColInd = in->GetFromMarshal<int*>();
    csrsv2Info_t info = (csrsv2Info_t)in->Get<size_t>();
    int * pBufferSizeInBytes = in->GetFromMarshal<int*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseScsrsv2_bufferSize(handle, trans, m, nnz, descr, csrVal, csrRowPtr, csrColInd, info, pBufferSizeInBytes);
        out->Add<csrsv2Info_t>(info);
        out->AddMarshal<int*>(pBufferSizeInBytes);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseScsrsv2_bufferSize Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Dcsrsv2_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Dcsrsv2_bufferSize"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    cusparseOperation_t trans = in->Get<cusparseOperation_t>();
    const int m = in->Get<int>();
    const int nnz = in->Get<int>();
    cusparseMatDescr_t descr = in->Get<cusparseMatDescr_t>();
    double * csrVal = in->GetFromMarshal<double*>();
    const int * csrRowPtr = in->GetFromMarshal<int*>();
    const int * csrColInd = in->GetFromMarshal<int*>();
    csrsv2Info_t info = (csrsv2Info_t)in->Get<size_t>();
    int * pBufferSizeInBytes = in->GetFromMarshal<int*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseDcsrsv2_bufferSize(handle, trans, m, nnz, descr, csrVal, csrRowPtr, csrColInd, info, pBufferSizeInBytes);
        out->Add<csrsv2Info_t>(info);
        out->AddMarshal<int*>(pBufferSizeInBytes);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseDcsrsv2_bufferSize Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Ccsrsv2_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Ccsrsv2_bufferSize"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    cusparseOperation_t trans = in->Get<cusparseOperation_t>();
    const int m = in->Get<int>();
    const int nnz = in->Get<int>();
    cusparseMatDescr_t descr = in->Get<cusparseMatDescr_t>();
    cuComplex * csrVal = in->GetFromMarshal<cuComplex*>();
    const int * csrRowPtr = in->GetFromMarshal<int*>();
    const int * csrColInd = in->GetFromMarshal<int*>();
    csrsv2Info_t info = (csrsv2Info_t)in->Get<size_t>();
    int * pBufferSizeInBytes = in->GetFromMarshal<int*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseCcsrsv2_bufferSize(handle, trans, m, nnz, descr, csrVal, csrRowPtr, csrColInd, info, pBufferSizeInBytes);
        out->Add<csrsv2Info_t>(info);
        out->AddMarshal<int*>(pBufferSizeInBytes);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseCcsrsv2_bufferSize Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Zcsrsv2_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Zcsrsv2_bufferSize"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    cusparseOperation_t trans = in->Get<cusparseOperation_t>();
    const int m = in->Get<int>();
    const int nnz = in->Get<int>();
    cusparseMatDescr_t descr = in->Get<cusparseMatDescr_t>();
    cuDoubleComplex * csrVal = in->GetFromMarshal<cuDoubleComplex*>();
    const int * csrRowPtr = in->GetFromMarshal<int*>();
    const int * csrColInd = in->GetFromMarshal<int*>();
    csrsv2Info_t info = (csrsv2Info_t)in->Get<size_t>();
    int * pBufferSizeInBytes = in->GetFromMarshal<int*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseZcsrsv2_bufferSize(handle, trans, m, nnz, descr, csrVal, csrRowPtr, csrColInd, info, pBufferSizeInBytes);
        out->Add<csrsv2Info_t>(info);
        out->AddMarshal<int*>(pBufferSizeInBytes);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseZcsrsv2_bufferSize Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Scsrsv2_analysis){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Scsrsv2_analysis"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    cusparseOperation_t trans = in->Get<cusparseOperation_t>();
    int m = in->Get<int>();
    int nnz = in->Get<int>();
    cusparseMatDescr_t descr = (cusparseMatDescr_t)in->Get<size_t>();
    float * csrVal = in->GetFromMarshal<float*>();
    int * csrRowPtr = in->GetFromMarshal<int*>();
    int * csrColInd = in->GetFromMarshal<int*>();
    csrsv2Info_t info = (csrsv2Info_t)in->Get<size_t>();
    cusparseSolvePolicy_t policy = in->Get<cusparseSolvePolicy_t>();
    void * pBuffer = in->GetFromMarshal<void*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseScsrsv2_analysis(handle, trans, m, nnz, descr, csrVal, csrRowPtr, csrColInd, info, policy, pBuffer);
        out->AddMarshal<csrsv2Info_t>(info);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseScsrsv2_analysis Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Dcsrsv2_analysis){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Dcsrsv2_analysis"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    const cusparseOperation_t trans = in->Get<cusparseOperation_t>();
    const int m = in->Get<int>();
    const int nnz = in->Get<int>();
    const cusparseMatDescr_t descr = in->Get<cusparseMatDescr_t>();
    double * csrVal = in->GetFromMarshal<double*>();
    const int * csrRowPtr = in->GetFromMarshal<int*>();
    const int * csrColInd = in->GetFromMarshal<int*>();
    csrsv2Info_t info = in->Get<csrsv2Info_t>();
    const cusparseSolvePolicy_t policy = in->Get<cusparseSolvePolicy_t>();
    void * pBuffer = in->GetFromMarshal<void*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseDcsrsv2_analysis(handle, trans, m, nnz, descr, csrVal, csrRowPtr, csrColInd, info, policy, pBuffer);
        out->AddMarshal<csrsv2Info_t>(info);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseDcsrsv2_analysis Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Ccsrsv2_analysis){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Ccsrsv2_analysis"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    const cusparseOperation_t trans = in->Get<cusparseOperation_t>();
    const int m = in->Get<int>();
    const int nnz = in->Get<int>();
    const cusparseMatDescr_t descr = in->Get<cusparseMatDescr_t>();
    cuComplex * csrVal = in->GetFromMarshal<cuComplex*>();
    const int * csrRowPtr = in->GetFromMarshal<int*>();
    const int * csrColInd = in->GetFromMarshal<int*>();
    csrsv2Info_t info = in->Get<csrsv2Info_t>();
    const cusparseSolvePolicy_t policy = in->Get<cusparseSolvePolicy_t>();
    void * pBuffer = in->GetFromMarshal<void*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseCcsrsv2_analysis(handle, trans, m, nnz, descr, csrVal, csrRowPtr, csrColInd, info, policy, pBuffer);
        out->AddMarshal<csrsv2Info_t>(info);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseCcsrsv2_analysis Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Zcsrsv2_analysis){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Zcsrsv2_analysis"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    const cusparseOperation_t trans = in->Get<cusparseOperation_t>();
    const int m = in->Get<int>();
    const int nnz = in->Get<int>();
    const cusparseMatDescr_t descr = in->Get<cusparseMatDescr_t>();
    cuDoubleComplex * csrVal = in->GetFromMarshal<cuDoubleComplex*>();
    const int * csrRowPtr = in->GetFromMarshal<int*>();
    const int * csrColInd = in->GetFromMarshal<int*>();
    csrsv2Info_t info = in->Get<csrsv2Info_t>();
    const cusparseSolvePolicy_t policy = in->Get<cusparseSolvePolicy_t>();
    void * pBuffer = in->GetFromMarshal<void*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseZcsrsv2_analysis(handle, trans, m, nnz, descr, csrVal, csrRowPtr, csrColInd, info, policy, pBuffer);
        out->AddMarshal<csrsv2Info_t>(info);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseZcsrsv2_analysis Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Scsrsv2_solve){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Scsrsv2_solve"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    cusparseOperation_t trans = in->Get<cusparseOperation_t>();
    int m = in->Get<int>();
    int nnz = in->Get<int>();
    const float * alpha = in->Assign<float>();
    cusparseMatDescr_t descr = (cusparseMatDescr_t)in->Get<size_t>();
    float * csrVal = in->GetFromMarshal<float*>();
    int * csrRowPtr = in->GetFromMarshal<int*>();
    int * csrColInd = in->GetFromMarshal<int*>();
    csrsv2Info_t info = (csrsv2Info_t)in->Get<size_t>();
    float * x = in->GetFromMarshal<float*>();
    float * y = in->GetFromMarshal<float*>();
    cusparseSolvePolicy_t policy = in->Get<cusparseSolvePolicy_t>();
    void * pBuffer = in->GetFromMarshal<void*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseScsrsv2_solve(handle, trans, m, nnz, alpha, descr, csrVal, csrRowPtr, csrColInd, info, x, y, policy, pBuffer);
        out->AddMarshal<float*>(y);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseScsrsv2_solve Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Dcsrsv2_solve){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Dcsrsv2_solve"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    cusparseOperation_t trans = in->Get<cusparseOperation_t>();
    int m = in->Get<int>();
    int nnz = in->Get<int>();
    const double * alpha = in->Assign<double>();
    cusparseMatDescr_t descr = (cusparseMatDescr_t)in->Get<size_t>();
    double * csrVal = in->GetFromMarshal<double*>();
    int * csrRowPtr = in->GetFromMarshal<int*>();
    int * csrColInd = in->GetFromMarshal<int*>();
    csrsv2Info_t info = (csrsv2Info_t)in->Get<size_t>();
    double * x = in->GetFromMarshal<double*>();
    double * y = in->GetFromMarshal<double*>();
    cusparseSolvePolicy_t policy = in->Get<cusparseSolvePolicy_t>();
    void * pBuffer = in->GetFromMarshal<void*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseDcsrsv2_solve(handle, trans, m, nnz, alpha, descr, csrVal, csrRowPtr, csrColInd, info, x, y, policy, pBuffer);
        out->AddMarshal<double*>(y);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseDcsrsv2_solve Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Ccsrsv2_solve){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Ccsrsv2_solve"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    cusparseOperation_t trans = in->Get<cusparseOperation_t>();
    int m = in->Get<int>();
    int nnz = in->Get<int>();
    const cuComplex * alpha = in->Assign<cuComplex>();
    cusparseMatDescr_t descr = (cusparseMatDescr_t)in->Get<size_t>();
    cuComplex * csrVal = in->GetFromMarshal<cuComplex*>();
    int * csrRowPtr = in->GetFromMarshal<int*>();
    int * csrColInd = in->GetFromMarshal<int*>();
    csrsv2Info_t info = (csrsv2Info_t)in->Get<size_t>();
    cuComplex * x = in->GetFromMarshal<cuComplex*>();
    cuComplex * y = in->GetFromMarshal<cuComplex*>();
    cusparseSolvePolicy_t policy = in->Get<cusparseSolvePolicy_t>();
    void * pBuffer = in->GetFromMarshal<void*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseCcsrsv2_solve(handle, trans, m, nnz, alpha, descr, csrVal, csrRowPtr, csrColInd, info, x, y, policy, pBuffer);
        out->AddMarshal<cuComplex*>(y);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseCcsrsv2_solve Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Zcsrsv2_solve){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Zcsrsv2_solve"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    cusparseOperation_t trans = in->Get<cusparseOperation_t>();
    int m = in->Get<int>();
    int nnz = in->Get<int>();
    const cuDoubleComplex * alpha = in->Assign<cuDoubleComplex>();
    cusparseMatDescr_t descr = (cusparseMatDescr_t)in->Get<size_t>();
    cuDoubleComplex * csrVal = in->GetFromMarshal<cuDoubleComplex*>();
    int * csrRowPtr = in->GetFromMarshal<int*>();
    int * csrColInd = in->GetFromMarshal<int*>();
    csrsv2Info_t info = (csrsv2Info_t)in->Get<size_t>();
    cuDoubleComplex * x = in->GetFromMarshal<cuDoubleComplex*>();
    cuDoubleComplex * y = in->GetFromMarshal<cuDoubleComplex*>();
    cusparseSolvePolicy_t policy = in->Get<cusparseSolvePolicy_t>();
    void * pBuffer = in->GetFromMarshal<void*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseZcsrsv2_solve(handle, trans, m, nnz, alpha, descr, csrVal, csrRowPtr, csrColInd, info, x, y, policy, pBuffer);
        out->AddMarshal<cuDoubleComplex*>(y);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseZcsrsv2_solve Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Xcsrsv2_zeroPivot){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Xcsrsv2_zeroPivot"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    csrsv2Info_t info = (csrsv2Info_t)in->Get<size_t>();
    int * position = in->GetFromMarshal<int*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseXcsrsv2_zeroPivot(handle, info, position);
        out->AddMarshal<int*>(position);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseXcsrsv2_zeroPivot Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Sgemvi_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Sgemvi_bufferSize"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    cusparseOperation_t transA = in->Get<cusparseOperation_t>();
    const int m = in->Get<int>();
    const int n = in->Get<int>();
    const int nnz = in->Get<int>();
    int * pBufferSize = in->GetFromMarshal<int*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseSgemvi_bufferSize(handle, transA, m, n, nnz, pBufferSize);
        out->AddMarshal<int*>(pBufferSize);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseSgemvi_bufferSize Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Dgemvi_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Dgemvi_bufferSize"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    cusparseOperation_t transA = in->Get<cusparseOperation_t>();
    const int m = in->Get<int>();
    const int n = in->Get<int>();
    const int nnz = in->Get<int>();
    int * pBufferSize = new int;
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseDgemvi_bufferSize(handle, transA, m, n, nnz, pBufferSize);
        out->Add<int*>(pBufferSize);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseDgemvi_bufferSize Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Cgemvi_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Cgemvi_bufferSize"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    cusparseOperation_t transA = in->Get<cusparseOperation_t>();
    const int m = in->Get<int>();
    const int n = in->Get<int>();
    const int nnz = in->Get<int>();
    int * pBufferSize = new int;
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseCgemvi_bufferSize(handle, transA, m, n, nnz, pBufferSize);
        out->Add<int*>(pBufferSize);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseCgemvi_bufferSize Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Zgemvi_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Zgemvi_bufferSize"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    cusparseOperation_t transA = in->Get<cusparseOperation_t>();
    const int m = in->Get<int>();
    const int n = in->Get<int>();
    const int nnz = in->Get<int>();
    int * pBufferSize = new int;
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseZgemvi_bufferSize(handle, transA, m, n, nnz, pBufferSize);
        out->Add<int*>(pBufferSize);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseZgemvi_bufferSize Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Sgemvi){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Sgemvi"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    cusparseOperation_t transA = in->Get<cusparseOperation_t>();
    const int m = in->Get<int>();
    const int n = in->Get<int>();
    const float * alpha = in->Assign<float>();
    const float * A = in->GetFromMarshal<float*>();
    const int lda = in->Get<int>();
    const int nnz = in->Get<int>();
    const float * x = in->Get<float*>();
    const int * xInd = in->Get<int*>();
    const float * beta = in->Assign<float>();
    float * y = in->Get<float*>();
    cusparseIndexBase_t idxBase = in->Get<cusparseIndexBase_t>();
    void* pBuffer = in->Get<void*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseSgemvi(handle, transA, m, n, alpha, A, lda, nnz, x, xInd, beta, y, idxBase, pBuffer);
        out->AddMarshal<float*>(y);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseSgemvi Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Dgemvi){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Dgemvi"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    cusparseOperation_t transA = in->Get<cusparseOperation_t>();
    const int m = in->Get<int>();
    const int n = in->Get<int>();
    const double * alpha = in->Assign<double>();
    const double * A = in->GetFromMarshal<double*>();
    const int lda = in->Get<int>();
    const int nnz = in->Get<int>();
    const double * x = in->Get<double*>();
    const int * xInd = in->Get<int*>();
    const double * beta = in->Assign<double>();
    double * y = in->Get<double*>();
    cusparseIndexBase_t idxBase = in->Get<cusparseIndexBase_t>();
    void* pBuffer = in->Get<void*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseDgemvi(handle, transA, m, n, alpha, A, lda, nnz, x, xInd, beta, y, idxBase, pBuffer);
        out->AddMarshal<double*>(y);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseDgemvi Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Cgemvi){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Cgemvi"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    cusparseOperation_t transA = in->Get<cusparseOperation_t>();
    const int m = in->Get<int>();
    const int n = in->Get<int>();
    const cuComplex * alpha = in->Assign<cuComplex>();
    const cuComplex * A = in->GetFromMarshal<cuComplex*>();
    const int lda = in->Get<int>();
    const int nnz = in->Get<int>();
    const cuComplex * x = in->Get<cuComplex*>();
    const int * xInd = in->Get<int*>();
    const cuComplex * beta = in->Assign<cuComplex>();
    cuComplex * y = in->Get<cuComplex*>();
    cusparseIndexBase_t idxBase = in->Get<cusparseIndexBase_t>();
    void* pBuffer = in->Get<void*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseCgemvi(handle, transA, m, n, alpha, A, lda, nnz, x, xInd, beta, y, idxBase, pBuffer);
        out->AddMarshal<cuComplex*>(y);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseCgemvi Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Zgemvi){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Zgemvi"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    cusparseOperation_t transA = in->Get<cusparseOperation_t>();
    const int m = in->Get<int>();
    const int n = in->Get<int>();
    const cuDoubleComplex * alpha = in->Assign<cuDoubleComplex>();
    const cuDoubleComplex * A = in->GetFromMarshal<cuDoubleComplex*>();
    const int lda = in->Get<int>();
    const int nnz = in->Get<int>();
    const cuDoubleComplex * x = in->Get<cuDoubleComplex*>();
    const int * xInd = in->Get<int*>();
    const cuDoubleComplex * beta = in->Assign<cuDoubleComplex>();
    cuDoubleComplex * y = in->Get<cuDoubleComplex*>();
    cusparseIndexBase_t idxBase = in->Get<cusparseIndexBase_t>();
    void* pBuffer = in->Get<void*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseZgemvi(handle, transA, m, n, alpha, A, lda, nnz, x, xInd, beta, y, idxBase, pBuffer);
        out->AddMarshal<cuDoubleComplex*>(y);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseZgemvi Executed");
    return std::make_shared<Result>(cs,out);
}

#ifndef CUSPARSE_VERSION
#error CUSPARSE_VERSION not defined
#endif
