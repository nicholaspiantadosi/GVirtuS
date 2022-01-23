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
#include "Utilities.h"

using namespace log4cplus;

using gvirtus::communicators::Buffer;
using gvirtus::communicators::Result;

CUSPARSE_ROUTINE_HANDLER(Scsrgeam2_bufferSizeExt){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Scsrgeam2_bufferSizeExt"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    const int m = in->Get<int>();
    const int n = in->Get<int>();
    const float * alpha = in->Assign<float>();
    const cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    const int nnzA = in->Get<int>();
    const float * csrValA = in->Get<float*>();
    const int * csrRowPtrA = in->Get<int*>();
    const int * csrColIndA = in->Get<int*>();
    const float * beta = in->Assign<float>();
    const cusparseMatDescr_t descrB = in->Get<cusparseMatDescr_t>();
    const int nnzB = in->Get<int>();
    const float * csrValB = in->Get<float*>();
    const int * csrRowPtrB = in->Get<int*>();
    const int * csrColIndB = in->Get<int*>();
    const cusparseMatDescr_t descrC = in->Get<cusparseMatDescr_t>();
    const float * csrValC = in->Get<float*>();
    const int * csrRowPtrC = in->Get<int*>();
    const int * csrColIndC = in->Get<int*>();
    size_t * pBufferSize = (in->Assign<size_t>());
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseScsrgeam2_bufferSizeExt(handle, m, n, alpha, descrA, nnzA, csrValA, csrRowPtrA, csrColIndA, beta, descrB, nnzB, csrValB, csrRowPtrB, csrColIndB, descrC, csrValC, csrRowPtrC, csrColIndC, pBufferSize);
        out->Add(pBufferSize);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseScsrgeam2_bufferSizeExt Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Dcsrgeam2_bufferSizeExt){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Dcsrgeam2_bufferSizeExt"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    const int m = in->Get<int>();
    const int n = in->Get<int>();
    const double * alpha = in->Assign<double>();
    const cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    const int nnzA = in->Get<int>();
    const double * csrValA = in->Get<double*>();
    const int * csrRowPtrA = in->Get<int*>();
    const int * csrColIndA = in->Get<int*>();
    const double * beta = in->Assign<double>();
    const cusparseMatDescr_t descrB = in->Get<cusparseMatDescr_t>();
    const int nnzB = in->Get<int>();
    const double * csrValB = in->Get<double*>();
    const int * csrRowPtrB = in->Get<int*>();
    const int * csrColIndB = in->Get<int*>();
    const cusparseMatDescr_t descrC = in->Get<cusparseMatDescr_t>();
    const double * csrValC = in->Get<double*>();
    const int * csrRowPtrC = in->Get<int*>();
    const int * csrColIndC = in->Get<int*>();
    size_t * pBufferSize = (in->Assign<size_t>());
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseDcsrgeam2_bufferSizeExt(handle, m, n, alpha, descrA, nnzA, csrValA, csrRowPtrA, csrColIndA, beta, descrB, nnzB, csrValB, csrRowPtrB, csrColIndB, descrC, csrValC, csrRowPtrC, csrColIndC, pBufferSize);
        out->Add(pBufferSize);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseDcsrgeam2_bufferSizeExt Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Ccsrgeam2_bufferSizeExt){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Ccsrgeam2_bufferSizeExt"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    const int m = in->Get<int>();
    const int n = in->Get<int>();
    const cuComplex * alpha = in->Assign<cuComplex>();
    const cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    const int nnzA = in->Get<int>();
    const cuComplex * csrValA = in->Get<cuComplex*>();
    const int * csrRowPtrA = in->Get<int*>();
    const int * csrColIndA = in->Get<int*>();
    const cuComplex * beta = in->Assign<cuComplex>();
    const cusparseMatDescr_t descrB = in->Get<cusparseMatDescr_t>();
    const int nnzB = in->Get<int>();
    const cuComplex * csrValB = in->Get<cuComplex*>();
    const int * csrRowPtrB = in->Get<int*>();
    const int * csrColIndB = in->Get<int*>();
    const cusparseMatDescr_t descrC = in->Get<cusparseMatDescr_t>();
    const cuComplex * csrValC = in->Get<cuComplex*>();
    const int * csrRowPtrC = in->Get<int*>();
    const int * csrColIndC = in->Get<int*>();
    size_t * pBufferSize = (in->Assign<size_t>());
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseCcsrgeam2_bufferSizeExt(handle, m, n, alpha, descrA, nnzA, csrValA, csrRowPtrA, csrColIndA, beta, descrB, nnzB, csrValB, csrRowPtrB, csrColIndB, descrC, csrValC, csrRowPtrC, csrColIndC, pBufferSize);
        out->Add(pBufferSize);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseCcsrgeam2_bufferSizeExt Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Zcsrgeam2_bufferSizeExt){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Zcsrgeam2_bufferSizeExt"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    const int m = in->Get<int>();
    const int n = in->Get<int>();
    const cuDoubleComplex * alpha = in->Assign<cuDoubleComplex>();
    const cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    const int nnzA = in->Get<int>();
    const cuDoubleComplex * csrValA = in->Get<cuDoubleComplex*>();
    const int * csrRowPtrA = in->Get<int*>();
    const int * csrColIndA = in->Get<int*>();
    const cuDoubleComplex * beta = in->Assign<cuDoubleComplex>();
    const cusparseMatDescr_t descrB = in->Get<cusparseMatDescr_t>();
    const int nnzB = in->Get<int>();
    const cuDoubleComplex * csrValB = in->Get<cuDoubleComplex*>();
    const int * csrRowPtrB = in->Get<int*>();
    const int * csrColIndB = in->Get<int*>();
    const cusparseMatDescr_t descrC = in->Get<cusparseMatDescr_t>();
    const cuDoubleComplex * csrValC = in->Get<cuDoubleComplex*>();
    const int * csrRowPtrC = in->Get<int*>();
    const int * csrColIndC = in->Get<int*>();
    size_t * pBufferSize = (in->Assign<size_t>());
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseZcsrgeam2_bufferSizeExt(handle, m, n, alpha, descrA, nnzA, csrValA, csrRowPtrA, csrColIndA, beta, descrB, nnzB, csrValB, csrRowPtrB, csrColIndB, descrC, csrValC, csrRowPtrC, csrColIndC, pBufferSize);
        out->Add(pBufferSize);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseZcsrgeam2_bufferSizeExt Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Xcsrgeam2Nnz){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Xcsrgeam2Nnz"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    const int m = in->Get<int>();
    const int n = in->Get<int>();
    const cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    const int nnzA = in->Get<int>();
    const int * csrRowPtrA = in->Get<int*>();
    const int * csrColIndA = in->Get<int*>();
    const cusparseMatDescr_t descrB = in->Get<cusparseMatDescr_t>();
    const int nnzB = in->Get<int>();
    const int * csrRowPtrB = in->Get<int*>();
    const int * csrColIndB = in->Get<int*>();
    const cusparseMatDescr_t descrC = in->Get<cusparseMatDescr_t>();
    int * csrRowPtrC = in->Get<int*>();
    void * workspace = in->GetFromMarshal<void*>();
    int * nnzTotalDevHostPtr = in->Assign<int>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseXcsrgeam2Nnz(handle, m, n, descrA, nnzA, csrRowPtrA, csrColIndA, descrB, nnzB, csrRowPtrB, csrColIndB, descrC, csrRowPtrC, nnzTotalDevHostPtr, workspace);
        out->Add<int*>(csrRowPtrC);
        out->Add(nnzTotalDevHostPtr);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseXcsrgeam2Nnz Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Scsrgeam2){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Scsrgeam2"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    const int m = in->Get<int>();
    const int n = in->Get<int>();
    const float * alpha = in->Assign<float>();
    const cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    const int nnzA = in->Get<int>();
    const float * csrValA = in->Get<float*>();
    const int * csrRowPtrA = in->Get<int*>();
    const int * csrColIndA = in->Get<int*>();
    const float * beta = in->Assign<float>();
    const cusparseMatDescr_t descrB = in->Get<cusparseMatDescr_t>();
    const int nnzB = in->Get<int>();
    const float * csrValB = in->Get<float*>();
    const int * csrRowPtrB = in->Get<int*>();
    const int * csrColIndB = in->Get<int*>();
    const cusparseMatDescr_t descrC = in->Get<cusparseMatDescr_t>();
    float * csrValC = in->Get<float*>();
    int * csrRowPtrC = in->Get<int*>();
    int * csrColIndC = in->Get<int*>();
    void * pBuffer = in->Get<void*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseScsrgeam2(handle, m, n, alpha, descrA, nnzA, csrValA, csrRowPtrA, csrColIndA, beta, descrB, nnzB, csrValB, csrRowPtrB, csrColIndB, descrC, csrValC, csrRowPtrC, csrColIndC, pBuffer);
        out->Add<float*>(csrValC);
        out->Add<int*>(csrRowPtrC);
        out->Add<int*>(csrColIndC);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseScsrgeam2 Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Dcsrgeam2){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Dcsrgeam2"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    const int m = in->Get<int>();
    const int n = in->Get<int>();
    const double * alpha = in->Assign<double>();
    const cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    const int nnzA = in->Get<int>();
    const double * csrValA = in->Get<double*>();
    const int * csrRowPtrA = in->Get<int*>();
    const int * csrColIndA = in->Get<int*>();
    const double * beta = in->Assign<double>();
    const cusparseMatDescr_t descrB = in->Get<cusparseMatDescr_t>();
    const int nnzB = in->Get<int>();
    const double * csrValB = in->Get<double*>();
    const int * csrRowPtrB = in->Get<int*>();
    const int * csrColIndB = in->Get<int*>();
    const cusparseMatDescr_t descrC = in->Get<cusparseMatDescr_t>();
    double * csrValC = in->Get<double*>();
    int * csrRowPtrC = in->Get<int*>();
    int * csrColIndC = in->Get<int*>();
    void * pBuffer = in->Get<void*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseDcsrgeam2(handle, m, n, alpha, descrA, nnzA, csrValA, csrRowPtrA, csrColIndA, beta, descrB, nnzB, csrValB, csrRowPtrB, csrColIndB, descrC, csrValC, csrRowPtrC, csrColIndC, pBuffer);
        out->Add<double*>(csrValC);
        out->Add<int*>(csrRowPtrC);
        out->Add<int*>(csrColIndC);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseDcsrgeam2 Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Ccsrgeam2){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Ccsrgeam2"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    const int m = in->Get<int>();
    const int n = in->Get<int>();
    const cuComplex * alpha = in->Assign<cuComplex>();
    const cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    const int nnzA = in->Get<int>();
    const cuComplex * csrValA = in->Get<cuComplex*>();
    const int * csrRowPtrA = in->Get<int*>();
    const int * csrColIndA = in->Get<int*>();
    const cuComplex * beta = in->Assign<cuComplex>();
    const cusparseMatDescr_t descrB = in->Get<cusparseMatDescr_t>();
    const int nnzB = in->Get<int>();
    const cuComplex * csrValB = in->Get<cuComplex*>();
    const int * csrRowPtrB = in->Get<int*>();
    const int * csrColIndB = in->Get<int*>();
    const cusparseMatDescr_t descrC = in->Get<cusparseMatDescr_t>();
    cuComplex * csrValC = in->Get<cuComplex*>();
    int * csrRowPtrC = in->Get<int*>();
    int * csrColIndC = in->Get<int*>();
    void * pBuffer = in->Get<void*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseCcsrgeam2(handle, m, n, alpha, descrA, nnzA, csrValA, csrRowPtrA, csrColIndA, beta, descrB, nnzB, csrValB, csrRowPtrB, csrColIndB, descrC, csrValC, csrRowPtrC, csrColIndC, pBuffer);
        out->Add<cuComplex*>(csrValC);
        out->Add<int*>(csrRowPtrC);
        out->Add<int*>(csrColIndC);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseCcsrgeam2 Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Zcsrgeam2){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Zcsrgeam2"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    const int m = in->Get<int>();
    const int n = in->Get<int>();
    const cuDoubleComplex * alpha = in->Assign<cuDoubleComplex>();
    const cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    const int nnzA = in->Get<int>();
    const cuDoubleComplex * csrValA = in->Get<cuDoubleComplex*>();
    const int * csrRowPtrA = in->Get<int*>();
    const int * csrColIndA = in->Get<int*>();
    const cuDoubleComplex * beta = in->Assign<cuDoubleComplex>();
    const cusparseMatDescr_t descrB = in->Get<cusparseMatDescr_t>();
    const int nnzB = in->Get<int>();
    const cuDoubleComplex * csrValB = in->Get<cuDoubleComplex*>();
    const int * csrRowPtrB = in->Get<int*>();
    const int * csrColIndB = in->Get<int*>();
    const cusparseMatDescr_t descrC = in->Get<cusparseMatDescr_t>();
    cuDoubleComplex * csrValC = in->Get<cuDoubleComplex*>();
    int * csrRowPtrC = in->Get<int*>();
    int * csrColIndC = in->Get<int*>();
    void * pBuffer = in->Get<void*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseZcsrgeam2(handle, m, n, alpha, descrA, nnzA, csrValA, csrRowPtrA, csrColIndA, beta, descrB, nnzB, csrValB, csrRowPtrB, csrColIndB, descrC, csrValC, csrRowPtrC, csrColIndC, pBuffer);
        out->Add<cuDoubleComplex*>(csrValC);
        out->Add<int*>(csrRowPtrC);
        out->Add<int*>(csrColIndC);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseZcsrgeam2 Executed");
    return std::make_shared<Result>(cs,out);
}

#ifndef CUSPARSE_VERSION
#error CUSPARSE_VERSION not defined
#endif
