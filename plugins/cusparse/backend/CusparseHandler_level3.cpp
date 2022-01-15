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

#ifndef CUSPARSE_VERSION
#error CUSPARSE_VERSION not defined
#endif
