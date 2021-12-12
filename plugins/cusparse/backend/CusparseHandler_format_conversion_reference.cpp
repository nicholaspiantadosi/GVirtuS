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

CUSPARSE_ROUTINE_HANDLER(Xcsr2bsrNnz){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Xcsr2bsrNnz"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<long long int>();
    cusparseDirection_t dir = (cusparseDirection_t)in->Get<cusparseDirection_t>();
    const int m = in->Get<int>();
    const int n = in->Get<int>();
    const cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    const int * csrRowPtrA = in->GetFromMarshal<int*>();
    const int * csrColIndA = in->GetFromMarshal<int*>();
    const int blockDim = in->Get<int>();
    const cusparseMatDescr_t descrC = in->Get<cusparseMatDescr_t>();
    int * bsrRowPtrC = in->GetFromMarshal<int*>();
    int nnzTotalDevHostPtr = 0;
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseXcsr2bsrNnz(handle, dir, m, n, descrA, csrRowPtrA, csrColIndA, blockDim, descrC, bsrRowPtrC, &nnzTotalDevHostPtr);
        out->AddMarshal<int*>(bsrRowPtrC);
        out->Add<int>(nnzTotalDevHostPtr);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseXcsr2bsrNnz Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Scsr2bsr){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Scsr2bsr"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<long long int>();
    cusparseDirection_t dir = (cusparseDirection_t)in->Get<cusparseDirection_t>();
    const int m = in->Get<int>();
    const int n = in->Get<int>();
    const cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    const float * csrValA = in->GetFromMarshal<float*>();
    const int * csrRowPtrA = in->GetFromMarshal<int*>();
    const int * csrColIndA = in->GetFromMarshal<int*>();
    const int blockDim = in->Get<int>();
    const cusparseMatDescr_t descrC = in->Get<cusparseMatDescr_t>();
    float * bsrValC = in->GetFromMarshal<float*>();
    int * bsrRowPtrC = in->GetFromMarshal<int*>();
    int * bsrColIndC = in->GetFromMarshal<int*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseScsr2bsr(handle, dir, m, n, descrA, csrValA, csrRowPtrA, csrColIndA, blockDim, descrC, bsrValC, bsrRowPtrC, bsrColIndC);
        out->AddMarshal<float*>(bsrValC);
        out->AddMarshal<int*>(bsrRowPtrC);
        out->AddMarshal<int*>(bsrColIndC);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseScsr2bsr Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Dcsr2bsr){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Dcsr2bsr"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<long long int>();
    cusparseDirection_t dir = (cusparseDirection_t)in->Get<cusparseDirection_t>();
    const int m = in->Get<int>();
    const int n = in->Get<int>();
    const cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    double * csrValA = in->GetFromMarshal<double*>();
    int * csrRowPtrA = in->GetFromMarshal<int*>();
    int * csrColIndA = in->GetFromMarshal<int*>();
    const int blockDim = in->Get<int>();
    const cusparseMatDescr_t descrC = in->Get<cusparseMatDescr_t>();
    double * bsrValC = in->GetFromMarshal<double*>();
    int * bsrRowPtrC = in->GetFromMarshal<int*>();
    int * bsrColIndC = in->GetFromMarshal<int*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseDcsr2bsr(handle, dir, m, n, descrA, csrValA, csrRowPtrA, csrColIndA, blockDim, descrC, bsrValC, bsrRowPtrC, bsrColIndC);
        out->AddMarshal<double*>(bsrValC);
        out->AddMarshal<int*>(bsrRowPtrC);
        out->AddMarshal<int*>(bsrColIndC);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseDcsr2bsr Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Ccsr2bsr){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Ccsr2bsr"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<long long int>();
    cusparseDirection_t dir = (cusparseDirection_t)in->Get<cusparseDirection_t>();
    const int m = in->Get<int>();
    const int n = in->Get<int>();
    const cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    const cuComplex * csrValA = in->GetFromMarshal<cuComplex*>();
    const int * csrRowPtrA = in->GetFromMarshal<int*>();
    const int * csrColIndA = in->GetFromMarshal<int*>();
    const int blockDim = in->Get<int>();
    const cusparseMatDescr_t descrC = in->Get<cusparseMatDescr_t>();
    cuComplex * bsrValC = in->GetFromMarshal<cuComplex*>();
    int * bsrRowPtrC = in->GetFromMarshal<int*>();
    int * bsrColIndC = in->GetFromMarshal<int*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseCcsr2bsr(handle, dir, m, n, descrA, csrValA, csrRowPtrA, csrColIndA, blockDim, descrC, bsrValC, bsrRowPtrC, bsrColIndC);
        out->AddMarshal<cuComplex*>(bsrValC);
        out->AddMarshal<int*>(bsrRowPtrC);
        out->AddMarshal<int*>(bsrColIndC);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseCcsr2bsr Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Zcsr2bsr){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Zcsr2bsr"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<long long int>();
    cusparseDirection_t dir = (cusparseDirection_t)in->Get<cusparseDirection_t>();
    const int m = in->Get<int>();
    const int n = in->Get<int>();
    const cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    const cuDoubleComplex * csrValA = in->GetFromMarshal<cuDoubleComplex*>();
    const int * csrRowPtrA = in->GetFromMarshal<int*>();
    const int * csrColIndA = in->GetFromMarshal<int*>();
    const int blockDim = in->Get<int>();
    const cusparseMatDescr_t descrC = in->Get<cusparseMatDescr_t>();
    cuDoubleComplex * bsrValC = in->GetFromMarshal<cuDoubleComplex*>();
    int * bsrRowPtrC = in->GetFromMarshal<int*>();
    int * bsrColIndC = in->GetFromMarshal<int*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseZcsr2bsr(handle, dir, m, n, descrA, csrValA, csrRowPtrA, csrColIndA, blockDim, descrC, bsrValC, bsrRowPtrC, bsrColIndC);
        out->AddMarshal<cuDoubleComplex*>(bsrValC);
        out->AddMarshal<int*>(bsrRowPtrC);
        out->AddMarshal<int*>(bsrColIndC);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseZcsr2bsr Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Sdense2csr){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Sdense2csr"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<long long int>();
    const int m = in->Get<int>();
    const int n = in->Get<int>();
    const cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    const float * A = in->GetFromMarshal<float*>();
    int lda = in->Get<int>();
    const int * nnzPerRow = in->GetFromMarshal<int*>();
    float * csrValA = in->GetFromMarshal<float*>();
    int * csrRowPtrA = in->GetFromMarshal<int*>();
    int * csrColIndA = in->GetFromMarshal<int*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseSdense2csr(handle, m, n, descrA, A, lda, nnzPerRow, csrValA, csrRowPtrA, csrColIndA);
        out->AddMarshal<float*>(csrValA);
        out->AddMarshal<int*>(csrRowPtrA);
        out->AddMarshal<int*>(csrColIndA);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseSdense2csr Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Ddense2csr){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Ddense2csr"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<long long int>();
    const int m = in->Get<int>();
    const int n = in->Get<int>();
    const cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    const double * A = in->GetFromMarshal<double*>();
    int lda = in->Get<int>();
    const int * nnzPerRow = in->GetFromMarshal<int*>();
    double * csrValA = in->GetFromMarshal<double*>();
    int * csrRowPtrA = in->GetFromMarshal<int*>();
    int * csrColIndA = in->GetFromMarshal<int*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseDdense2csr(handle, m, n, descrA, A, lda, nnzPerRow, csrValA, csrRowPtrA, csrColIndA);
        out->AddMarshal<double*>(csrValA);
        out->AddMarshal<int*>(csrRowPtrA);
        out->AddMarshal<int*>(csrColIndA);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseDdense2csr Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Cdense2csr){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Cdense2csr"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<long long int>();
    const int m = in->Get<int>();
    const int n = in->Get<int>();
    const cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    const cuComplex * A = in->GetFromMarshal<cuComplex*>();
    int lda = in->Get<int>();
    const int * nnzPerRow = in->GetFromMarshal<int*>();
    cuComplex * csrValA = in->GetFromMarshal<cuComplex*>();
    int * csrRowPtrA = in->GetFromMarshal<int*>();
    int * csrColIndA = in->GetFromMarshal<int*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseCdense2csr(handle, m, n, descrA, A, lda, nnzPerRow, csrValA, csrRowPtrA, csrColIndA);
        out->AddMarshal<cuComplex*>(csrValA);
        out->AddMarshal<int*>(csrRowPtrA);
        out->AddMarshal<int*>(csrColIndA);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseCdense2csr Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Zdense2csr){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Zdense2csr"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<long long int>();
    const int m = in->Get<int>();
    const int n = in->Get<int>();
    const cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    const cuDoubleComplex * A = in->GetFromMarshal<cuDoubleComplex*>();
    int lda = in->Get<int>();
    const int * nnzPerRow = in->GetFromMarshal<int*>();
    cuDoubleComplex * csrValA = in->GetFromMarshal<cuDoubleComplex*>();
    int * csrRowPtrA = in->GetFromMarshal<int*>();
    int * csrColIndA = in->GetFromMarshal<int*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseZdense2csr(handle, m, n, descrA, A, lda, nnzPerRow, csrValA, csrRowPtrA, csrColIndA);
        out->AddMarshal<cuDoubleComplex*>(csrValA);
        out->AddMarshal<int*>(csrRowPtrA);
        out->AddMarshal<int*>(csrColIndA);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseZdense2csr Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Snnz){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Snnz"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<long long int>();
    const cusparseDirection_t dirA = in->Get<cusparseDirection_t>();
    const int m = in->Get<int>();
    const int n = in->Get<int>();
    const cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    const float * A = in->GetFromMarshal<float*>();
    int lda = in->Get<int>();
    int * nnzPerRowColumn = in->GetFromMarshal<int*>();
    int nnzTotalDevHostPtr = 0;
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseSnnz(handle, dirA, m, n, descrA, A, lda, nnzPerRowColumn, &nnzTotalDevHostPtr);
        out->AddMarshal<int*>(nnzPerRowColumn);
        out->Add<int>(nnzTotalDevHostPtr);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseSnnz Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Dnnz){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Dnnz"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<long long int>();
    const cusparseDirection_t dirA = in->Get<cusparseDirection_t>();
    const int m = in->Get<int>();
    const int n = in->Get<int>();
    const cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    const double * A = in->GetFromMarshal<double*>();
    int lda = in->Get<int>();
    int * nnzPerRowColumn = in->GetFromMarshal<int*>();
    int nnzTotalDevHostPtr = 0;
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseDnnz(handle, dirA, m, n, descrA, A, lda, nnzPerRowColumn, &nnzTotalDevHostPtr);
        out->AddMarshal<int*>(nnzPerRowColumn);
        out->Add<int>(nnzTotalDevHostPtr);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseDnnz Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Cnnz){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Cnnz"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<long long int>();
    const cusparseDirection_t dirA = in->Get<cusparseDirection_t>();
    const int m = in->Get<int>();
    const int n = in->Get<int>();
    const cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    const cuComplex * A = in->GetFromMarshal<cuComplex*>();
    int lda = in->Get<int>();
    int * nnzPerRowColumn = in->GetFromMarshal<int*>();
    int nnzTotalDevHostPtr = 0;
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseCnnz(handle, dirA, m, n, descrA, A, lda, nnzPerRowColumn, &nnzTotalDevHostPtr);
        out->AddMarshal<int*>(nnzPerRowColumn);
        out->Add<int>(nnzTotalDevHostPtr);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseCnnz Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Znnz){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Znnz"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<long long int>();
    const cusparseDirection_t dirA = in->Get<cusparseDirection_t>();
    const int m = in->Get<int>();
    const int n = in->Get<int>();
    const cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    const cuDoubleComplex * A = in->GetFromMarshal<cuDoubleComplex*>();
    int lda = in->Get<int>();
    int * nnzPerRowColumn = in->GetFromMarshal<int*>();
    int nnzTotalDevHostPtr = 0;
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseZnnz(handle, dirA, m, n, descrA, A, lda, nnzPerRowColumn, &nnzTotalDevHostPtr);
        out->AddMarshal<int*>(nnzPerRowColumn);
        out->Add<int>(nnzTotalDevHostPtr);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseZnnz Executed");
    return std::make_shared<Result>(cs,out);
}

#ifndef CUSPARSE_VERSION
#error CUSPARSE_VERSION not defined
#endif
