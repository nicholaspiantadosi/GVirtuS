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
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<long long int>();
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
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<long long int>();
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
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<long long int>();
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
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<long long int>();
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
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<long long int>();
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
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<long long int>();
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
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<long long int>();
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
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<long long int>();
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

#ifndef CUSPARSE_VERSION
#error CUSPARSE_VERSION not defined
#endif
