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

CUSPARSE_ROUTINE_HANDLER(Sbsr2csr){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Sbsr2csr"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    cusparseDirection_t dir = (cusparseDirection_t)in->Get<cusparseDirection_t>();
    const int mb = in->Get<int>();
    const int nb = in->Get<int>();
    const cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    const float * bsrValA = in->GetFromMarshal<float*>();
    const int * bsrRowPtrA = in->GetFromMarshal<int*>();
    const int * bsrColIndA = in->GetFromMarshal<int*>();
    const int blockDim = in->Get<int>();
    const cusparseMatDescr_t descrC = in->Get<cusparseMatDescr_t>();
    float * csrValC = in->GetFromMarshal<float*>();
    int * csrRowPtrC = in->GetFromMarshal<int*>();
    int * csrColIndC = in->GetFromMarshal<int*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseSbsr2csr(handle, dir, mb, nb, descrA, bsrValA, bsrRowPtrA, bsrColIndA, blockDim, descrC, csrValC, csrRowPtrC, csrColIndC);
        out->AddMarshal<float*>(csrValC);
        out->AddMarshal<int*>(csrRowPtrC);
        out->AddMarshal<int*>(csrColIndC);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseSbsr2csr Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Dbsr2csr){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Dbsr2csr"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    cusparseDirection_t dir = (cusparseDirection_t)in->Get<cusparseDirection_t>();
    const int mb = in->Get<int>();
    const int nb = in->Get<int>();
    const cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    const double * bsrValA = in->GetFromMarshal<double*>();
    const int * bsrRowPtrA = in->GetFromMarshal<int*>();
    const int * bsrColIndA = in->GetFromMarshal<int*>();
    const int blockDim = in->Get<int>();
    const cusparseMatDescr_t descrC = in->Get<cusparseMatDescr_t>();
    double * csrValC = in->GetFromMarshal<double*>();
    int * csrRowPtrC = in->GetFromMarshal<int*>();
    int * csrColIndC = in->GetFromMarshal<int*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseDbsr2csr(handle, dir, mb, nb, descrA, bsrValA, bsrRowPtrA, bsrColIndA, blockDim, descrC, csrValC, csrRowPtrC, csrColIndC);
        out->AddMarshal<double*>(csrValC);
        out->AddMarshal<int*>(csrRowPtrC);
        out->AddMarshal<int*>(csrColIndC);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseDbsr2csr Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Cbsr2csr){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Cbsr2csr"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    cusparseDirection_t dir = (cusparseDirection_t)in->Get<cusparseDirection_t>();
    const int mb = in->Get<int>();
    const int nb = in->Get<int>();
    const cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    const cuComplex * bsrValA = in->GetFromMarshal<cuComplex*>();
    const int * bsrRowPtrA = in->GetFromMarshal<int*>();
    const int * bsrColIndA = in->GetFromMarshal<int*>();
    const int blockDim = in->Get<int>();
    const cusparseMatDescr_t descrC = in->Get<cusparseMatDescr_t>();
    cuComplex * csrValC = in->GetFromMarshal<cuComplex*>();
    int * csrRowPtrC = in->GetFromMarshal<int*>();
    int * csrColIndC = in->GetFromMarshal<int*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseCbsr2csr(handle, dir, mb, nb, descrA, bsrValA, bsrRowPtrA, bsrColIndA, blockDim, descrC, csrValC, csrRowPtrC, csrColIndC);
        out->AddMarshal<cuComplex*>(csrValC);
        out->AddMarshal<int*>(csrRowPtrC);
        out->AddMarshal<int*>(csrColIndC);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseCbsr2csr Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Zbsr2csr){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Zbsr2csr"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    cusparseDirection_t dir = (cusparseDirection_t)in->Get<cusparseDirection_t>();
    const int mb = in->Get<int>();
    const int nb = in->Get<int>();
    const cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    const cuDoubleComplex * bsrValA = in->GetFromMarshal<cuDoubleComplex*>();
    const int * bsrRowPtrA = in->GetFromMarshal<int*>();
    const int * bsrColIndA = in->GetFromMarshal<int*>();
    const int blockDim = in->Get<int>();
    const cusparseMatDescr_t descrC = in->Get<cusparseMatDescr_t>();
    cuDoubleComplex * csrValC = in->GetFromMarshal<cuDoubleComplex*>();
    int * csrRowPtrC = in->GetFromMarshal<int*>();
    int * csrColIndC = in->GetFromMarshal<int*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseZbsr2csr(handle, dir, mb, nb, descrA, bsrValA, bsrRowPtrA, bsrColIndA, blockDim, descrC, csrValC, csrRowPtrC, csrColIndC);
        out->AddMarshal<cuDoubleComplex*>(csrValC);
        out->AddMarshal<int*>(csrRowPtrC);
        out->AddMarshal<int*>(csrColIndC);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseZbsr2csr Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Sgebsr2gebsc_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Sgebsr2gebsc_bufferSize"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    int mb = in->Get<int>();
    int nb = in->Get<int>();
    int nnzb = in->Get<int>();
    const float * bsrValA = in->GetFromMarshal<float*>();
    const int * bsrRowPtrA = in->GetFromMarshal<int*>();
    const int * bsrColIndA = in->GetFromMarshal<int*>();
    int rowBlockDim = in->Get<int>();
    int colBlockDim = in->Get<int>();
    int * pBufferSize = new int;
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseSgebsr2gebsc_bufferSize(handle, mb, nb, nnzb, bsrValA, bsrRowPtrA, bsrColIndA, rowBlockDim, colBlockDim, pBufferSize);
        out->AddMarshal<int*>(pBufferSize);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseSgebsr2gebsc_bufferSize Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Dgebsr2gebsc_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Dgebsr2gebsc_bufferSize"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    int mb = in->Get<int>();
    int nb = in->Get<int>();
    int nnzb = in->Get<int>();
    const double * bsrValA = in->GetFromMarshal<double*>();
    const int * bsrRowPtrA = in->GetFromMarshal<int*>();
    const int * bsrColIndA = in->GetFromMarshal<int*>();
    int rowBlockDim = in->Get<int>();
    int colBlockDim = in->Get<int>();
    int * pBufferSize = new int;
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseDgebsr2gebsc_bufferSize(handle, mb, nb, nnzb, bsrValA, bsrRowPtrA, bsrColIndA, rowBlockDim, colBlockDim, pBufferSize);
        out->AddMarshal<int*>(pBufferSize);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseDgebsr2gebsc_bufferSize Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Cgebsr2gebsc_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Cgebsr2gebsc_bufferSize"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    int mb = in->Get<int>();
    int nb = in->Get<int>();
    int nnzb = in->Get<int>();
    const cuComplex * bsrValA = in->GetFromMarshal<cuComplex*>();
    const int * bsrRowPtrA = in->GetFromMarshal<int*>();
    const int * bsrColIndA = in->GetFromMarshal<int*>();
    int rowBlockDim = in->Get<int>();
    int colBlockDim = in->Get<int>();
    int * pBufferSize = new int;
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseCgebsr2gebsc_bufferSize(handle, mb, nb, nnzb, bsrValA, bsrRowPtrA, bsrColIndA, rowBlockDim, colBlockDim, pBufferSize);
        out->AddMarshal<int*>(pBufferSize);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseCgebsr2gebsc_bufferSize Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Zgebsr2gebsc_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Zgebsr2gebsc_bufferSize"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    int mb = in->Get<int>();
    int nb = in->Get<int>();
    int nnzb = in->Get<int>();
    const cuDoubleComplex * bsrValA = in->GetFromMarshal<cuDoubleComplex*>();
    const int * bsrRowPtrA = in->GetFromMarshal<int*>();
    const int * bsrColIndA = in->GetFromMarshal<int*>();
    int rowBlockDim = in->Get<int>();
    int colBlockDim = in->Get<int>();
    int * pBufferSize = new int;
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseZgebsr2gebsc_bufferSize(handle, mb, nb, nnzb, bsrValA, bsrRowPtrA, bsrColIndA, rowBlockDim, colBlockDim, pBufferSize);
        out->AddMarshal<int*>(pBufferSize);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseZgebsr2gebsc_bufferSize Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Sgebsr2gebsc){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Sgebsr2gebsc"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    int mb = in->Get<int>();
    int nb = in->Get<int>();
    int nnzb = in->Get<int>();
    const float * bsrValA = in->GetFromMarshal<float*>();
    const int * bsrRowPtrA = in->GetFromMarshal<int*>();
    const int * bsrColIndA = in->GetFromMarshal<int*>();
    int rowBlockDim = in->Get<int>();
    int colBlockDim = in->Get<int>();
    float* bscVal = in->GetFromMarshal<float*>();
    int* bscRowInd = in->GetFromMarshal<int*>();
    int* bscColPtr = in->GetFromMarshal<int*>();
    cusparseAction_t copyValues = in->Get<cusparseAction_t>();
    cusparseIndexBase_t baseIdx = in->Get<cusparseIndexBase_t>();
    void * pBuffer = in->GetFromMarshal<void*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseSgebsr2gebsc(handle, mb, nb, nnzb, bsrValA, bsrRowPtrA, bsrColIndA, rowBlockDim, colBlockDim, bscVal, bscRowInd, bscColPtr, copyValues, baseIdx, pBuffer);
        out->AddMarshal<float*>(bscVal);
        out->AddMarshal<int*>(bscRowInd);
        out->AddMarshal<int*>(bscColPtr);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseSgebsr2gebsc Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Dgebsr2gebsc){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Dgebsr2gebsc"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    int mb = in->Get<int>();
    int nb = in->Get<int>();
    int nnzb = in->Get<int>();
    const double * bsrValA = in->GetFromMarshal<double*>();
    const int * bsrRowPtrA = in->GetFromMarshal<int*>();
    const int * bsrColIndA = in->GetFromMarshal<int*>();
    int rowBlockDim = in->Get<int>();
    int colBlockDim = in->Get<int>();
    double* bscVal = in->GetFromMarshal<double*>();
    int* bscRowInd = in->GetFromMarshal<int*>();
    int* bscColPtr = in->GetFromMarshal<int*>();
    cusparseAction_t copyValues = in->Get<cusparseAction_t>();
    cusparseIndexBase_t baseIdx = in->Get<cusparseIndexBase_t>();
    void * pBuffer = in->GetFromMarshal<void*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseDgebsr2gebsc(handle, mb, nb, nnzb, bsrValA, bsrRowPtrA, bsrColIndA, rowBlockDim, colBlockDim, bscVal, bscRowInd, bscColPtr, copyValues, baseIdx, pBuffer);
        out->AddMarshal<double*>(bscVal);
        out->AddMarshal<int*>(bscRowInd);
        out->AddMarshal<int*>(bscColPtr);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseDgebsr2gebsc Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Cgebsr2gebsc){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Cgebsr2gebsc"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    int mb = in->Get<int>();
    int nb = in->Get<int>();
    int nnzb = in->Get<int>();
    const cuComplex * bsrValA = in->GetFromMarshal<cuComplex*>();
    const int * bsrRowPtrA = in->GetFromMarshal<int*>();
    const int * bsrColIndA = in->GetFromMarshal<int*>();
    int rowBlockDim = in->Get<int>();
    int colBlockDim = in->Get<int>();
    cuComplex* bscVal = in->GetFromMarshal<cuComplex*>();
    int* bscRowInd = in->GetFromMarshal<int*>();
    int* bscColPtr = in->GetFromMarshal<int*>();
    cusparseAction_t copyValues = in->Get<cusparseAction_t>();
    cusparseIndexBase_t baseIdx = in->Get<cusparseIndexBase_t>();
    void * pBuffer = in->GetFromMarshal<void*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseCgebsr2gebsc(handle, mb, nb, nnzb, bsrValA, bsrRowPtrA, bsrColIndA, rowBlockDim, colBlockDim, bscVal, bscRowInd, bscColPtr, copyValues, baseIdx, pBuffer);
        out->AddMarshal<cuComplex*>(bscVal);
        out->AddMarshal<int*>(bscRowInd);
        out->AddMarshal<int*>(bscColPtr);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseCgebsr2gebsc Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Zgebsr2gebsc){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Zgebsr2gebsc"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    int mb = in->Get<int>();
    int nb = in->Get<int>();
    int nnzb = in->Get<int>();
    const cuDoubleComplex * bsrValA = in->GetFromMarshal<cuDoubleComplex*>();
    const int * bsrRowPtrA = in->GetFromMarshal<int*>();
    const int * bsrColIndA = in->GetFromMarshal<int*>();
    int rowBlockDim = in->Get<int>();
    int colBlockDim = in->Get<int>();
    cuDoubleComplex* bscVal = in->GetFromMarshal<cuDoubleComplex*>();
    int* bscRowInd = in->GetFromMarshal<int*>();
    int* bscColPtr = in->GetFromMarshal<int*>();
    cusparseAction_t copyValues = in->Get<cusparseAction_t>();
    cusparseIndexBase_t baseIdx = in->Get<cusparseIndexBase_t>();
    void * pBuffer = in->GetFromMarshal<void*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseZgebsr2gebsc(handle, mb, nb, nnzb, bsrValA, bsrRowPtrA, bsrColIndA, rowBlockDim, colBlockDim, bscVal, bscRowInd, bscColPtr, copyValues, baseIdx, pBuffer);
        out->AddMarshal<cuDoubleComplex*>(bscVal);
        out->AddMarshal<int*>(bscRowInd);
        out->AddMarshal<int*>(bscColPtr);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseZgebsr2gebsc Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Sgebsr2gebsr_bufferSize) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Sgebsr2gebsr_bufferSize"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t) in->Get<size_t>();
    cusparseDirection_t dir = in->Get<cusparseDirection_t>();
    int mb = in->Get<int>();
    int nb = in->Get<int>();
    int nnzb = in->Get<int>();
    const cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    const float *bsrValA = in->GetFromMarshal<float *>();
    const int *bsrRowPtrA = in->GetFromMarshal<int *>();
    const int *bsrColIndA = in->GetFromMarshal<int *>();
    int rowBlockDimA = in->Get<int>();
    int colBlockDimA = in->Get<int>();
    int rowBlockDimC = in->Get<int>();
    int colBlockDimC = in->Get<int>();
    int *pBufferSize = new int;
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        cs = cusparseSgebsr2gebsr_bufferSize(handle, dir, mb, nb, nnzb, descrA, bsrValA, bsrRowPtrA, bsrColIndA, rowBlockDimA,
                                             colBlockDimA, rowBlockDimC, colBlockDimC, pBufferSize);
        out->Add<int*>(pBufferSize);
    } catch (string e) {
        LOG4CPLUS_DEBUG(logger, e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger, "Sgebsr2gebsr_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSPARSE_ROUTINE_HANDLER(Dgebsr2gebsr_bufferSize) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Dgebsr2gebsr_bufferSize"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t) in->Get<size_t>();
    cusparseDirection_t dir = in->Get<cusparseDirection_t>();
    int mb = in->Get<int>();
    int nb = in->Get<int>();
    int nnzb = in->Get<int>();
    const cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    const double *bsrValA = in->GetFromMarshal<double *>();
    const int *bsrRowPtrA = in->GetFromMarshal<int *>();
    const int *bsrColIndA = in->GetFromMarshal<int *>();
    int rowBlockDimA = in->Get<int>();
    int colBlockDimA = in->Get<int>();
    int rowBlockDimC = in->Get<int>();
    int colBlockDimC = in->Get<int>();
    int *pBufferSize = new int;
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        cs = cusparseDgebsr2gebsr_bufferSize(handle, dir, mb, nb, nnzb, descrA, bsrValA, bsrRowPtrA, bsrColIndA, rowBlockDimA,
                                             colBlockDimA, rowBlockDimC, colBlockDimC, pBufferSize);
        out->AddMarshal<int *>(pBufferSize);
    } catch (string e) {
        LOG4CPLUS_DEBUG(logger, e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger, "Dgebsr2gebsr_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSPARSE_ROUTINE_HANDLER(Cgebsr2gebsr_bufferSize) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Cgebsr2gebsr_bufferSize"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t) in->Get<size_t>();
    cusparseDirection_t dir = in->Get<cusparseDirection_t>();
    int mb = in->Get<int>();
    int nb = in->Get<int>();
    int nnzb = in->Get<int>();
    const cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    const cuComplex *bsrValA = in->GetFromMarshal<cuComplex *>();
    const int *bsrRowPtrA = in->GetFromMarshal<int *>();
    const int *bsrColIndA = in->GetFromMarshal<int *>();
    int rowBlockDimA = in->Get<int>();
    int colBlockDimA = in->Get<int>();
    int rowBlockDimC = in->Get<int>();
    int colBlockDimC = in->Get<int>();
    int *pBufferSize = new int;
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        cs = cusparseCgebsr2gebsr_bufferSize(handle, dir, mb, nb, nnzb, descrA, bsrValA, bsrRowPtrA, bsrColIndA, rowBlockDimA,
                                             colBlockDimA, rowBlockDimC, colBlockDimC, pBufferSize);
        out->AddMarshal<int *>(pBufferSize);
    } catch (string e) {
        LOG4CPLUS_DEBUG(logger, e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger, "Cgebsr2gebsr_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSPARSE_ROUTINE_HANDLER(Zgebsr2gebsr_bufferSize) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Zgebsr2gebsr_bufferSize"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t) in->Get<size_t>();
    cusparseDirection_t dir = in->Get<cusparseDirection_t>();
    int mb = in->Get<int>();
    int nb = in->Get<int>();
    int nnzb = in->Get<int>();
    const cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    const cuDoubleComplex *bsrValA = in->GetFromMarshal<cuDoubleComplex *>();
    const int *bsrRowPtrA = in->GetFromMarshal<int *>();
    const int *bsrColIndA = in->GetFromMarshal<int *>();
    int rowBlockDimA = in->Get<int>();
    int colBlockDimA = in->Get<int>();
    int rowBlockDimC = in->Get<int>();
    int colBlockDimC = in->Get<int>();
    int *pBufferSize = new int;
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        cs = cusparseZgebsr2gebsr_bufferSize(handle, dir, mb, nb, nnzb, descrA, bsrValA, bsrRowPtrA, bsrColIndA, rowBlockDimA,
                                             colBlockDimA, rowBlockDimC, colBlockDimC, pBufferSize);
        out->AddMarshal<int *>(pBufferSize);
    } catch (string e) {
        LOG4CPLUS_DEBUG(logger, e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger, "Zgebsr2gebsr_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSPARSE_ROUTINE_HANDLER(Xgebsr2gebsrNnz){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Xgebsr2gebsrNnz"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    cusparseDirection_t dir = in->Get<cusparseDirection_t>();
    const int mb = in->Get<int>();
    const int nb = in->Get<int>();
    const int nnzb = in->Get<int>();
    const cusparseMatDescr_t descrA = (cusparseMatDescr_t)in->Get<size_t>();
    const int * bsrRowPtrA = in->GetFromMarshal<int*>();
    const int * bsrColIndA = in->GetFromMarshal<int*>();
    const int rowBlockDimA = in->Get<int>();
    const int colBlockDimA = in->Get<int>();
    const cusparseMatDescr_t descrC = (cusparseMatDescr_t)in->Get<size_t>();
    int * bsrRowPtrC = in->GetFromMarshal<int*>();
    const int rowBlockDimC = in->Get<int>();
    const int colBlockDimC = in->Get<int>();
    void * pBuffer = in->GetFromMarshal<void*>();
    int nnzTotalDevHostPtr = 0;
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseXgebsr2gebsrNnz(handle, dir, mb, nb, nnzb, descrA, bsrRowPtrA, bsrColIndA, rowBlockDimA, colBlockDimA, descrC, bsrRowPtrC, rowBlockDimC, colBlockDimC, &nnzTotalDevHostPtr, pBuffer);
        out->AddMarshal<int*>(bsrRowPtrC);
        out->Add<int>(nnzTotalDevHostPtr);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseXgebsr2gebsrNnz Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Sgebsr2gebsr) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Sgebsr2gebsr"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t) in->Get<size_t>();
    cusparseDirection_t dir = in->Get<cusparseDirection_t>();
    int mb = in->Get<int>();
    int nb = in->Get<int>();
    int nnzb = in->Get<int>();
    const cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    float *bsrValA = in->GetFromMarshal<float *>();
    int *bsrRowPtrA = in->GetFromMarshal<int *>();
    int *bsrColIndA = in->GetFromMarshal<int *>();
    int rowBlockDimA = in->Get<int>();
    int colBlockDimA = in->Get<int>();
    const cusparseMatDescr_t descrC = in->Get<cusparseMatDescr_t>();
    float* bsrValC = in->GetFromMarshal<float *>();
    int* bsrRowPtrC = in->GetFromMarshal<int *>();
    int* bsrColIndC = in->GetFromMarshal<int *>();
    int rowBlockDimC = in->Get<int>();
    int colBlockDimC = in->Get<int>();
    void * pBuffer = in->GetFromMarshal<void*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        cs = cusparseSgebsr2gebsr(handle, dir, mb, nb, nnzb, descrA, bsrValA, bsrRowPtrA, bsrColIndA, rowBlockDimA,
                                             colBlockDimA, descrC, bsrValC, bsrRowPtrC, bsrColIndC, rowBlockDimC, colBlockDimC, pBuffer);
        out->AddMarshal<float*>(bsrValC);
        out->AddMarshal<int*>(bsrRowPtrC);
        out->AddMarshal<int*>(bsrColIndC);
    } catch (string e) {
        LOG4CPLUS_DEBUG(logger, e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger, "cusparseSgebsr2gebsr Executed");
    return std::make_shared<Result>(cs, out);
}

CUSPARSE_ROUTINE_HANDLER(Dgebsr2gebsr) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Dgebsr2gebsr"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t) in->Get<size_t>();
    cusparseDirection_t dir = in->Get<cusparseDirection_t>();
    int mb = in->Get<int>();
    int nb = in->Get<int>();
    int nnzb = in->Get<int>();
    const cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    const double *bsrValA = in->GetFromMarshal<double *>();
    const int *bsrRowPtrA = in->GetFromMarshal<int *>();
    const int *bsrColIndA = in->GetFromMarshal<int *>();
    int rowBlockDimA = in->Get<int>();
    int colBlockDimA = in->Get<int>();
    const cusparseMatDescr_t descrC = in->Get<cusparseMatDescr_t>();
    double* bsrValC = in->GetFromMarshal<double *>();
    int* bsrRowPtrC = in->GetFromMarshal<int *>();
    int* bsrColIndC = in->GetFromMarshal<int *>();
    int rowBlockDimC = in->Get<int>();
    int colBlockDimC = in->Get<int>();
    void * pBuffer = in->GetFromMarshal<void*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        cs = cusparseDgebsr2gebsr(handle, dir, mb, nb, nnzb, descrA, bsrValA, bsrRowPtrA, bsrColIndA, rowBlockDimA,
                                  colBlockDimA, descrC, bsrValC, bsrRowPtrC, bsrColIndC, rowBlockDimC, colBlockDimC, pBuffer);
        out->AddMarshal<double*>(bsrValC);
        out->AddMarshal<int*>(bsrRowPtrC);
        out->AddMarshal<int*>(bsrColIndC);
    } catch (string e) {
        LOG4CPLUS_DEBUG(logger, e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger, "cusparseDgebsr2gebsr Executed");
    return std::make_shared<Result>(cs, out);
}

CUSPARSE_ROUTINE_HANDLER(Cgebsr2gebsr) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Cgebsr2gebsr"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t) in->Get<size_t>();
    cusparseDirection_t dir = in->Get<cusparseDirection_t>();
    int mb = in->Get<int>();
    int nb = in->Get<int>();
    int nnzb = in->Get<int>();
    const cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    const cuComplex *bsrValA = in->GetFromMarshal<cuComplex *>();
    const int *bsrRowPtrA = in->GetFromMarshal<int *>();
    const int *bsrColIndA = in->GetFromMarshal<int *>();
    int rowBlockDimA = in->Get<int>();
    int colBlockDimA = in->Get<int>();
    const cusparseMatDescr_t descrC = in->Get<cusparseMatDescr_t>();
    cuComplex* bsrValC = in->GetFromMarshal<cuComplex *>();
    int* bsrRowPtrC = in->GetFromMarshal<int *>();
    int* bsrColIndC = in->GetFromMarshal<int *>();
    int rowBlockDimC = in->Get<int>();
    int colBlockDimC = in->Get<int>();
    void * pBuffer = in->GetFromMarshal<void*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        cs = cusparseCgebsr2gebsr(handle, dir, mb, nb, nnzb, descrA, bsrValA, bsrRowPtrA, bsrColIndA, rowBlockDimA,
                                  colBlockDimA, descrC, bsrValC, bsrRowPtrC, bsrColIndC, rowBlockDimC, colBlockDimC, pBuffer);
        out->AddMarshal<cuComplex*>(bsrValC);
        out->AddMarshal<int*>(bsrRowPtrC);
        out->AddMarshal<int*>(bsrColIndC);
    } catch (string e) {
        LOG4CPLUS_DEBUG(logger, e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger, "cusparseCgebsr2gebsr Executed");
    return std::make_shared<Result>(cs, out);
}

CUSPARSE_ROUTINE_HANDLER(Zgebsr2gebsr) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Zgebsr2gebsr"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t) in->Get<size_t>();
    cusparseDirection_t dir = in->Get<cusparseDirection_t>();
    int mb = in->Get<int>();
    int nb = in->Get<int>();
    int nnzb = in->Get<int>();
    const cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    const cuDoubleComplex *bsrValA = in->GetFromMarshal<cuDoubleComplex *>();
    const int *bsrRowPtrA = in->GetFromMarshal<int *>();
    const int *bsrColIndA = in->GetFromMarshal<int *>();
    int rowBlockDimA = in->Get<int>();
    int colBlockDimA = in->Get<int>();
    const cusparseMatDescr_t descrC = in->Get<cusparseMatDescr_t>();
    cuDoubleComplex* bsrValC = in->GetFromMarshal<cuDoubleComplex *>();
    int* bsrRowPtrC = in->GetFromMarshal<int *>();
    int* bsrColIndC = in->GetFromMarshal<int *>();
    int rowBlockDimC = in->Get<int>();
    int colBlockDimC = in->Get<int>();
    void * pBuffer = in->GetFromMarshal<void*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        cs = cusparseZgebsr2gebsr(handle, dir, mb, nb, nnzb, descrA, bsrValA, bsrRowPtrA, bsrColIndA, rowBlockDimA,
                                  colBlockDimA, descrC, bsrValC, bsrRowPtrC, bsrColIndC, rowBlockDimC, colBlockDimC, pBuffer);
        out->AddMarshal<cuDoubleComplex*>(bsrValC);
        out->AddMarshal<int*>(bsrRowPtrC);
        out->AddMarshal<int*>(bsrColIndC);
    } catch (string e) {
        LOG4CPLUS_DEBUG(logger, e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger, "cusparseZgebsr2gebsr Executed");
    return std::make_shared<Result>(cs, out);
}

CUSPARSE_ROUTINE_HANDLER(Sgebsr2csr) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Sgebsr2csr"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t) in->Get<size_t>();
    cusparseDirection_t dir = in->Get<cusparseDirection_t>();
    int mb = in->Get<int>();
    int nb = in->Get<int>();
    const cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    float *bsrValA = in->GetFromMarshal<float *>();
    int *bsrRowPtrA = in->GetFromMarshal<int *>();
    int *bsrColIndA = in->GetFromMarshal<int *>();
    int rowBlockDim = in->Get<int>();
    int colBlockDim = in->Get<int>();
    const cusparseMatDescr_t descrC = in->Get<cusparseMatDescr_t>();
    float* csrValC = in->GetFromMarshal<float *>();
    int* csrRowPtrC = in->GetFromMarshal<int *>();
    int* csrColIndC = in->GetFromMarshal<int *>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        cs = cusparseSgebsr2csr(handle, dir, mb, nb, descrA, bsrValA, bsrRowPtrA, bsrColIndA, rowBlockDim,
                                  colBlockDim, descrC, csrValC, csrRowPtrC, csrColIndC);
        out->AddMarshal<float*>(csrValC);
        out->AddMarshal<int*>(csrRowPtrC);
        out->AddMarshal<int*>(csrColIndC);
    } catch (string e) {
        LOG4CPLUS_DEBUG(logger, e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger, "cusparseSgebsr2csr Executed");
    return std::make_shared<Result>(cs, out);
}

CUSPARSE_ROUTINE_HANDLER(Dgebsr2csr) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Dgebsr2csr"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t) in->Get<size_t>();
    cusparseDirection_t dir = in->Get<cusparseDirection_t>();
    int mb = in->Get<int>();
    int nb = in->Get<int>();
    const cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    double *bsrValA = in->GetFromMarshal<double *>();
    int *bsrRowPtrA = in->GetFromMarshal<int *>();
    int *bsrColIndA = in->GetFromMarshal<int *>();
    int rowBlockDim = in->Get<int>();
    int colBlockDim = in->Get<int>();
    const cusparseMatDescr_t descrC = in->Get<cusparseMatDescr_t>();
    double* csrValC = in->GetFromMarshal<double *>();
    int* csrRowPtrC = in->GetFromMarshal<int *>();
    int* csrColIndC = in->GetFromMarshal<int *>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        cs = cusparseDgebsr2csr(handle, dir, mb, nb, descrA, bsrValA, bsrRowPtrA, bsrColIndA, rowBlockDim,
                                colBlockDim, descrC, csrValC, csrRowPtrC, csrColIndC);
        out->AddMarshal<double*>(csrValC);
        out->AddMarshal<int*>(csrRowPtrC);
        out->AddMarshal<int*>(csrColIndC);
    } catch (string e) {
        LOG4CPLUS_DEBUG(logger, e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger, "cusparseDgebsr2csr Executed");
    return std::make_shared<Result>(cs, out);
}

CUSPARSE_ROUTINE_HANDLER(Cgebsr2csr) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Cgebsr2csr"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t) in->Get<size_t>();
    cusparseDirection_t dir = in->Get<cusparseDirection_t>();
    int mb = in->Get<int>();
    int nb = in->Get<int>();
    const cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    cuComplex *bsrValA = in->GetFromMarshal<cuComplex *>();
    int *bsrRowPtrA = in->GetFromMarshal<int *>();
    int *bsrColIndA = in->GetFromMarshal<int *>();
    int rowBlockDim = in->Get<int>();
    int colBlockDim = in->Get<int>();
    const cusparseMatDescr_t descrC = in->Get<cusparseMatDescr_t>();
    cuComplex* csrValC = in->GetFromMarshal<cuComplex *>();
    int* csrRowPtrC = in->GetFromMarshal<int *>();
    int* csrColIndC = in->GetFromMarshal<int *>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        cs = cusparseCgebsr2csr(handle, dir, mb, nb, descrA, bsrValA, bsrRowPtrA, bsrColIndA, rowBlockDim,
                                colBlockDim, descrC, csrValC, csrRowPtrC, csrColIndC);
        out->AddMarshal<cuComplex*>(csrValC);
        out->AddMarshal<int*>(csrRowPtrC);
        out->AddMarshal<int*>(csrColIndC);
    } catch (string e) {
        LOG4CPLUS_DEBUG(logger, e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger, "cusparseCgebsr2csr Executed");
    return std::make_shared<Result>(cs, out);
}

CUSPARSE_ROUTINE_HANDLER(Zgebsr2csr) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Zgebsr2csr"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t) in->Get<size_t>();
    cusparseDirection_t dir = in->Get<cusparseDirection_t>();
    int mb = in->Get<int>();
    int nb = in->Get<int>();
    const cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    cuDoubleComplex *bsrValA = in->GetFromMarshal<cuDoubleComplex *>();
    int *bsrRowPtrA = in->GetFromMarshal<int *>();
    int *bsrColIndA = in->GetFromMarshal<int *>();
    int rowBlockDim = in->Get<int>();
    int colBlockDim = in->Get<int>();
    const cusparseMatDescr_t descrC = in->Get<cusparseMatDescr_t>();
    cuDoubleComplex* csrValC = in->GetFromMarshal<cuDoubleComplex *>();
    int* csrRowPtrC = in->GetFromMarshal<int *>();
    int* csrColIndC = in->GetFromMarshal<int *>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        cs = cusparseZgebsr2csr(handle, dir, mb, nb, descrA, bsrValA, bsrRowPtrA, bsrColIndA, rowBlockDim,
                                colBlockDim, descrC, csrValC, csrRowPtrC, csrColIndC);
        out->AddMarshal<cuDoubleComplex*>(csrValC);
        out->AddMarshal<int*>(csrRowPtrC);
        out->AddMarshal<int*>(csrColIndC);
    } catch (string e) {
        LOG4CPLUS_DEBUG(logger, e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger, "cusparseZgebsr2csr Executed");
    return std::make_shared<Result>(cs, out);
}

CUSPARSE_ROUTINE_HANDLER(Scsr2gebsr_bufferSize) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Scsr2gebsr_bufferSize"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t) in->Get<size_t>();
    cusparseDirection_t dir = in->Get<cusparseDirection_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    const cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    const float *csrValA = in->GetFromMarshal<float *>();
    const int *csrRowPtrA = in->GetFromMarshal<int *>();
    const int *csrColIndA = in->GetFromMarshal<int *>();
    int rowBlockDim = in->Get<int>();
    int colBlockDim = in->Get<int>();
    int *pBufferSize = new int;
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        cs = cusparseScsr2gebsr_bufferSize(handle, dir, m, n, descrA, csrValA, csrRowPtrA, csrColIndA, rowBlockDim,
                                             colBlockDim, pBufferSize);
        out->Add<int*>(pBufferSize);
    } catch (string e) {
        LOG4CPLUS_DEBUG(logger, e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger, "Scsr2gebsr_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSPARSE_ROUTINE_HANDLER(Dcsr2gebsr_bufferSize) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Dcsr2gebsr_bufferSize"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t) in->Get<size_t>();
    cusparseDirection_t dir = in->Get<cusparseDirection_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    const cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    const double *csrValA = in->GetFromMarshal<double *>();
    const int *csrRowPtrA = in->GetFromMarshal<int *>();
    const int *csrColIndA = in->GetFromMarshal<int *>();
    int rowBlockDim = in->Get<int>();
    int colBlockDim = in->Get<int>();
    int *pBufferSize = new int;
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        cs = cusparseDcsr2gebsr_bufferSize(handle, dir, m, n, descrA, csrValA, csrRowPtrA, csrColIndA, rowBlockDim,
                                           colBlockDim, pBufferSize);
        out->Add<int*>(pBufferSize);
    } catch (string e) {
        LOG4CPLUS_DEBUG(logger, e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger, "Dcsr2gebsr_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSPARSE_ROUTINE_HANDLER(Ccsr2gebsr_bufferSize) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Ccsr2gebsr_bufferSize"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t) in->Get<size_t>();
    cusparseDirection_t dir = in->Get<cusparseDirection_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    const cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    const cuComplex *csrValA = in->GetFromMarshal<cuComplex *>();
    const int *csrRowPtrA = in->GetFromMarshal<int *>();
    const int *csrColIndA = in->GetFromMarshal<int *>();
    int rowBlockDim = in->Get<int>();
    int colBlockDim = in->Get<int>();
    int *pBufferSize = new int;
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        cs = cusparseCcsr2gebsr_bufferSize(handle, dir, m, n, descrA, csrValA, csrRowPtrA, csrColIndA, rowBlockDim,
                                           colBlockDim, pBufferSize);
        out->Add<int*>(pBufferSize);
    } catch (string e) {
        LOG4CPLUS_DEBUG(logger, e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger, "Ccsr2gebsr_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSPARSE_ROUTINE_HANDLER(Zcsr2gebsr_bufferSize) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Zcsr2gebsr_bufferSize"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t) in->Get<size_t>();
    cusparseDirection_t dir = in->Get<cusparseDirection_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    const cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    const cuDoubleComplex *csrValA = in->GetFromMarshal<cuDoubleComplex *>();
    const int *csrRowPtrA = in->GetFromMarshal<int *>();
    const int *csrColIndA = in->GetFromMarshal<int *>();
    int rowBlockDim = in->Get<int>();
    int colBlockDim = in->Get<int>();
    int *pBufferSize = new int;
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        cs = cusparseZcsr2gebsr_bufferSize(handle, dir, m, n, descrA, csrValA, csrRowPtrA, csrColIndA, rowBlockDim,
                                           colBlockDim, pBufferSize);
        out->Add<int*>(pBufferSize);
    } catch (string e) {
        LOG4CPLUS_DEBUG(logger, e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger, "Zcsr2gebsr_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSPARSE_ROUTINE_HANDLER(Xcsr2gebsrNnz){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Xcsr2gebsrNnz"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    cusparseDirection_t dir = in->Get<cusparseDirection_t>();
    const int m = in->Get<int>();
    const int n = in->Get<int>();
    const cusparseMatDescr_t descrA = (cusparseMatDescr_t)in->Get<size_t>();
    const int * csrRowPtrA = in->GetFromMarshal<int*>();
    const int * csrColIndA = in->GetFromMarshal<int*>();
    const cusparseMatDescr_t descrC = (cusparseMatDescr_t)in->Get<size_t>();
    int * bsrRowPtrC = in->GetFromMarshal<int*>();
    const int rowBlockDim = in->Get<int>();
    const int colBlockDim = in->Get<int>();
    void * pBuffer = in->GetFromMarshal<void*>();
    int nnzTotalDevHostPtr = 0;
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseXcsr2gebsrNnz(handle, dir, m, n, descrA, csrRowPtrA, csrColIndA, descrC, bsrRowPtrC, rowBlockDim, colBlockDim, &nnzTotalDevHostPtr, pBuffer);
        out->AddMarshal<int*>(bsrRowPtrC);
        out->Add<int>(nnzTotalDevHostPtr);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseXcsr2gebsrNnz Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Scsr2gebsr) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Scsr2gebsr"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t) in->Get<size_t>();
    cusparseDirection_t dir = in->Get<cusparseDirection_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    const cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    float *csrValA = in->GetFromMarshal<float *>();
    int *csrRowPtrA = in->GetFromMarshal<int *>();
    int *csrColIndA = in->GetFromMarshal<int *>();
    const cusparseMatDescr_t descrC = in->Get<cusparseMatDescr_t>();
    float* bsrValC = in->GetFromMarshal<float *>();
    int* bsrRowPtrC = in->GetFromMarshal<int *>();
    int* bsrColIndC = in->GetFromMarshal<int *>();
    int rowBlockDim = in->Get<int>();
    int colBlockDim = in->Get<int>();
    void * pBuffer = in->GetFromMarshal<void*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        cs = cusparseScsr2gebsr(handle, dir, m, n, descrA, csrValA, csrRowPtrA, csrColIndA, descrC, bsrValC, bsrRowPtrC, bsrColIndC, rowBlockDim, colBlockDim, pBuffer);
        out->AddMarshal<float*>(bsrValC);
        out->AddMarshal<int*>(bsrRowPtrC);
        out->AddMarshal<int*>(bsrColIndC);
    } catch (string e) {
        LOG4CPLUS_DEBUG(logger, e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger, "cusparseScsr2gebsr Executed");
    return std::make_shared<Result>(cs, out);
}

CUSPARSE_ROUTINE_HANDLER(Dcsr2gebsr) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Dcsr2gebsr"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t) in->Get<size_t>();
    cusparseDirection_t dir = in->Get<cusparseDirection_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    const cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    double *csrValA = in->GetFromMarshal<double *>();
    int *csrRowPtrA = in->GetFromMarshal<int *>();
    int *csrColIndA = in->GetFromMarshal<int *>();
    const cusparseMatDescr_t descrC = in->Get<cusparseMatDescr_t>();
    double* bsrValC = in->GetFromMarshal<double *>();
    int* bsrRowPtrC = in->GetFromMarshal<int *>();
    int* bsrColIndC = in->GetFromMarshal<int *>();
    int rowBlockDim = in->Get<int>();
    int colBlockDim = in->Get<int>();
    void * pBuffer = in->GetFromMarshal<void*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        cs = cusparseDcsr2gebsr(handle, dir, m, n, descrA, csrValA, csrRowPtrA, csrColIndA, descrC, bsrValC, bsrRowPtrC, bsrColIndC, rowBlockDim, colBlockDim, pBuffer);
        out->AddMarshal<double*>(bsrValC);
        out->AddMarshal<int*>(bsrRowPtrC);
        out->AddMarshal<int*>(bsrColIndC);
    } catch (string e) {
        LOG4CPLUS_DEBUG(logger, e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger, "cusparseDcsr2gebsr Executed");
    return std::make_shared<Result>(cs, out);
}

CUSPARSE_ROUTINE_HANDLER(Ccsr2gebsr) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Ccsr2gebsr"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t) in->Get<size_t>();
    cusparseDirection_t dir = in->Get<cusparseDirection_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    const cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    cuComplex *csrValA = in->GetFromMarshal<cuComplex *>();
    int *csrRowPtrA = in->GetFromMarshal<int *>();
    int *csrColIndA = in->GetFromMarshal<int *>();
    const cusparseMatDescr_t descrC = in->Get<cusparseMatDescr_t>();
    cuComplex* bsrValC = in->GetFromMarshal<cuComplex *>();
    int* bsrRowPtrC = in->GetFromMarshal<int *>();
    int* bsrColIndC = in->GetFromMarshal<int *>();
    int rowBlockDim = in->Get<int>();
    int colBlockDim = in->Get<int>();
    void * pBuffer = in->GetFromMarshal<void*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        cs = cusparseCcsr2gebsr(handle, dir, m, n, descrA, csrValA, csrRowPtrA, csrColIndA, descrC, bsrValC, bsrRowPtrC, bsrColIndC, rowBlockDim, colBlockDim, pBuffer);
        out->AddMarshal<cuComplex*>(bsrValC);
        out->AddMarshal<int*>(bsrRowPtrC);
        out->AddMarshal<int*>(bsrColIndC);
    } catch (string e) {
        LOG4CPLUS_DEBUG(logger, e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger, "cusparseCcsr2gebsr Executed");
    return std::make_shared<Result>(cs, out);
}

CUSPARSE_ROUTINE_HANDLER(Zcsr2gebsr) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Zcsr2gebsr"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t) in->Get<size_t>();
    cusparseDirection_t dir = in->Get<cusparseDirection_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    const cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    cuDoubleComplex *csrValA = in->GetFromMarshal<cuDoubleComplex *>();
    int *csrRowPtrA = in->GetFromMarshal<int *>();
    int *csrColIndA = in->GetFromMarshal<int *>();
    const cusparseMatDescr_t descrC = in->Get<cusparseMatDescr_t>();
    cuDoubleComplex* bsrValC = in->GetFromMarshal<cuDoubleComplex *>();
    int* bsrRowPtrC = in->GetFromMarshal<int *>();
    int* bsrColIndC = in->GetFromMarshal<int *>();
    int rowBlockDim = in->Get<int>();
    int colBlockDim = in->Get<int>();
    void * pBuffer = in->GetFromMarshal<void*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        cs = cusparseZcsr2gebsr(handle, dir, m, n, descrA, csrValA, csrRowPtrA, csrColIndA, descrC, bsrValC, bsrRowPtrC, bsrColIndC, rowBlockDim, colBlockDim, pBuffer);
        out->AddMarshal<cuDoubleComplex*>(bsrValC);
        out->AddMarshal<int*>(bsrRowPtrC);
        out->AddMarshal<int*>(bsrColIndC);
    } catch (string e) {
        LOG4CPLUS_DEBUG(logger, e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger, "cusparseZcsr2gebsr Executed");
    return std::make_shared<Result>(cs, out);
}

CUSPARSE_ROUTINE_HANDLER(Xcoo2csr) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Xcoo2csr"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t) in->Get<size_t>();
    const int* cooRowInd = in->GetFromMarshal<int *>();
    int nnz = in->Get<int>();
    int m = in->Get<int>();
    int *csrRowPtr = in->GetFromMarshal<int *>();
    cusparseIndexBase_t idxBase = in->Get<cusparseIndexBase_t>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        cs = cusparseXcoo2csr(handle, cooRowInd, nnz, m, csrRowPtr, idxBase);
        out->AddMarshal<int*>(csrRowPtr);
    } catch (string e) {
        LOG4CPLUS_DEBUG(logger, e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger, "cusparseXcoo2csr Executed");
    return std::make_shared<Result>(cs, out);
}

CUSPARSE_ROUTINE_HANDLER(Scsc2dense) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Scsc2dense"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t) in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    const cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    float *cscValA = in->GetFromMarshal<float *>();
    int *cscRowIndA = in->GetFromMarshal<int *>();
    int *cscColPtrA = in->GetFromMarshal<int *>();
    float* A = in->GetFromMarshal<float *>();
    int lda = in->Get<int>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        cs = cusparseScsc2dense(handle, m, n, descrA, cscValA, cscRowIndA, cscColPtrA, A, lda);
        out->AddMarshal<float*>(A);
    } catch (string e) {
        LOG4CPLUS_DEBUG(logger, e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger, "cusparseScsc2dense Executed");
    return std::make_shared<Result>(cs, out);
}

CUSPARSE_ROUTINE_HANDLER(Dcsc2dense) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Dcsc2dense"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t) in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    const cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    double *cscValA = in->GetFromMarshal<double *>();
    int *cscRowIndA = in->GetFromMarshal<int *>();
    int *cscColPtrA = in->GetFromMarshal<int *>();
    double* A = in->GetFromMarshal<double *>();
    int lda = in->Get<int>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        cs = cusparseDcsc2dense(handle, m, n, descrA, cscValA, cscRowIndA, cscColPtrA, A, lda);
        out->AddMarshal<double*>(A);
    } catch (string e) {
        LOG4CPLUS_DEBUG(logger, e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger, "cusparseDcsc2dense Executed");
    return std::make_shared<Result>(cs, out);
}

CUSPARSE_ROUTINE_HANDLER(Ccsc2dense) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Ccsc2dense"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t) in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    const cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    cuComplex *cscValA = in->GetFromMarshal<cuComplex *>();
    int *cscRowIndA = in->GetFromMarshal<int *>();
    int *cscColPtrA = in->GetFromMarshal<int *>();
    cuComplex* A = in->GetFromMarshal<cuComplex *>();
    int lda = in->Get<int>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        cs = cusparseCcsc2dense(handle, m, n, descrA, cscValA, cscRowIndA, cscColPtrA, A, lda);
        out->AddMarshal<cuComplex*>(A);
    } catch (string e) {
        LOG4CPLUS_DEBUG(logger, e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger, "cusparseCcsc2dense Executed");
    return std::make_shared<Result>(cs, out);
}

CUSPARSE_ROUTINE_HANDLER(Zcsc2dense) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Zcsc2dense"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t) in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    const cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    cuDoubleComplex *cscValA = in->GetFromMarshal<cuDoubleComplex *>();
    int *cscRowIndA = in->GetFromMarshal<int *>();
    int *cscColPtrA = in->GetFromMarshal<int *>();
    cuDoubleComplex* A = in->GetFromMarshal<cuDoubleComplex *>();
    int lda = in->Get<int>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        cs = cusparseZcsc2dense(handle, m, n, descrA, cscValA, cscRowIndA, cscColPtrA, A, lda);
        out->AddMarshal<cuDoubleComplex*>(A);
    } catch (string e) {
        LOG4CPLUS_DEBUG(logger, e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger, "cusparseZcsc2dense Executed");
    return std::make_shared<Result>(cs, out);
}

CUSPARSE_ROUTINE_HANDLER(Xcsr2bsrNnz){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Xcsr2bsrNnz"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
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
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
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
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
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
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
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
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
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

CUSPARSE_ROUTINE_HANDLER(Xcsr2coo) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Xcsr2coo"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t) in->Get<size_t>();
    const int *csrRowPtr = in->GetFromMarshal<int *>();
    int nnz = in->Get<int>();
    int m = in->Get<int>();
    int* cooRowInd = in->GetFromMarshal<int *>();
    cusparseIndexBase_t idxBase = in->Get<cusparseIndexBase_t>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        cs = cusparseXcsr2coo(handle, csrRowPtr, nnz, m, cooRowInd, idxBase);
        out->AddMarshal<int*>(cooRowInd);
    } catch (string e) {
        LOG4CPLUS_DEBUG(logger, e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger, "cusparseXcsr2coo Executed");
    return std::make_shared<Result>(cs, out);
}

CUSPARSE_ROUTINE_HANDLER(Csr2cscEx2_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Csr2cscEx2_bufferSize"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    int nnz = in->Get<int>();
    const int * csrRowPtr = in->GetFromMarshal<int*>();
    const int * csrColInd = in->GetFromMarshal<int*>();
    int * cscColPtr = in->GetFromMarshal<int*>();
    int * cscRowInd = in->GetFromMarshal<int*>();
    cudaDataType valType = in->GetFromMarshal<cudaDataType>();
    cusparseAction_t copyValues = in->GetFromMarshal<cusparseAction_t>();
    cusparseIndexBase_t idxBase = in->GetFromMarshal<cusparseIndexBase_t>();
    cusparseCsr2CscAlg_t alg = in->GetFromMarshal<cusparseCsr2CscAlg_t>();
    size_t * bufferSize = new size_t;
    void * csrVal;
    void * cscVal;
    if (valType == CUDA_R_32F) {
        // float
        csrVal = in->GetFromMarshal<float*>();
        cscVal = in->GetFromMarshal<float*>();
    } else if (valType == CUDA_R_64F) {
        // double
        csrVal = in->GetFromMarshal<double*>();
        cscVal = in->GetFromMarshal<double*>();
    } else if (valType == CUDA_C_32F) {
        // cuComplex
        csrVal = in->GetFromMarshal<cuComplex*>();
        cscVal = in->GetFromMarshal<cuComplex*>();
    } else if (valType == CUDA_C_64F) {
        // cuDoubleComplex
        csrVal = in->GetFromMarshal<cuDoubleComplex*>();
        cscVal = in->GetFromMarshal<cuDoubleComplex*>();
    } else {
        throw "Type not supported by GVirtus!";
    }
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseCsr2cscEx2_bufferSize(handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, cscVal, cscColPtr, cscRowInd, valType, copyValues, idxBase, alg, bufferSize);
        out->AddMarshal<size_t*>(bufferSize);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseCsr2cscEx2_bufferSize Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Csr2cscEx2){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Csr2cscEx2"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    int nnz = in->Get<int>();
    const int * csrRowPtr = in->GetFromMarshal<int*>();
    const int * csrColInd = in->GetFromMarshal<int*>();
    int * cscColPtr = in->GetFromMarshal<int*>();
    int * cscRowInd = in->GetFromMarshal<int*>();
    cudaDataType valType = in->GetFromMarshal<cudaDataType>();
    cusparseAction_t copyValues = in->GetFromMarshal<cusparseAction_t>();
    cusparseIndexBase_t idxBase = in->GetFromMarshal<cusparseIndexBase_t>();
    cusparseCsr2CscAlg_t alg = in->GetFromMarshal<cusparseCsr2CscAlg_t>();
    void* buffer = in->GetFromMarshal<void*>();
    void * csrVal;
    void * cscVal;
    if (valType == CUDA_R_32F) {
        // float
        csrVal = in->GetFromMarshal<float*>();
        cscVal = in->GetFromMarshal<float*>();
    } else if (valType == CUDA_R_64F) {
        // double
        csrVal = in->GetFromMarshal<double*>();
        cscVal = in->GetFromMarshal<double*>();
    } else if (valType == CUDA_C_32F) {
        // cuComplex
        csrVal = in->GetFromMarshal<cuComplex*>();
        cscVal = in->GetFromMarshal<cuComplex*>();
    } else if (valType == CUDA_C_64F) {
        // cuDoubleComplex
        csrVal = in->GetFromMarshal<cuDoubleComplex*>();
        cscVal = in->GetFromMarshal<cuDoubleComplex*>();
    } else {
        throw "Type not supported by GVirtus!";
    }
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseCsr2cscEx2(handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, cscVal, cscColPtr, cscRowInd, valType, copyValues, idxBase, alg, buffer);
        if (valType == CUDA_R_32F) {
            // float
            out->AddMarshal<float*>((float*)cscVal);
        } else if (valType == CUDA_R_64F) {
            // double
            out->AddMarshal<double*>((double*)cscVal);
        } else if (valType == CUDA_C_32F) {
            // cuComplex
            out->AddMarshal<cuComplex*>((cuComplex*)cscVal);
        } else if (valType == CUDA_C_64F) {
            // cuDoubleComplex
            out->AddMarshal<cuDoubleComplex*>((cuDoubleComplex*)cscVal);
        } else {
            throw "Type not supported by GVirtus!";
        }
        out->AddMarshal<int*>(cscColPtr);
        out->AddMarshal<int*>(cscRowInd);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseCsr2cscEx2 Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Scsr2dense){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Scsr2dense"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    const int m = in->Get<int>();
    const int n = in->Get<int>();
    const cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    float * csrValA = in->GetFromMarshal<float*>();
    int * csrRowPtrA = in->GetFromMarshal<int*>();
    int * csrColIndA = in->GetFromMarshal<int*>();
    float * A = in->GetFromMarshal<float*>();
    int lda = in->Get<int>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseScsr2dense(handle, m, n, descrA, csrValA, csrRowPtrA, csrColIndA, A, lda);
        out->AddMarshal<float*>(A);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseScsr2dense Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Dcsr2dense){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Dcsr2dense"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    const int m = in->Get<int>();
    const int n = in->Get<int>();
    const cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    double * csrValA = in->GetFromMarshal<double*>();
    int * csrRowPtrA = in->GetFromMarshal<int*>();
    int * csrColIndA = in->GetFromMarshal<int*>();
    double * A = in->GetFromMarshal<double*>();
    int lda = in->Get<int>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseDcsr2dense(handle, m, n, descrA, csrValA, csrRowPtrA, csrColIndA, A, lda);
        out->AddMarshal<double*>(A);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseDcsr2dense Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Ccsr2dense){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Ccsr2dense"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    const int m = in->Get<int>();
    const int n = in->Get<int>();
    const cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    cuComplex * csrValA = in->GetFromMarshal<cuComplex*>();
    int * csrRowPtrA = in->GetFromMarshal<int*>();
    int * csrColIndA = in->GetFromMarshal<int*>();
    cuComplex * A = in->GetFromMarshal<cuComplex*>();
    int lda = in->Get<int>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseCcsr2dense(handle, m, n, descrA, csrValA, csrRowPtrA, csrColIndA, A, lda);
        out->AddMarshal<cuComplex*>(A);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseCcsr2dense Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Zcsr2dense){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Zcsr2dense"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    const int m = in->Get<int>();
    const int n = in->Get<int>();
    const cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    cuDoubleComplex * csrValA = in->GetFromMarshal<cuDoubleComplex*>();
    int * csrRowPtrA = in->GetFromMarshal<int*>();
    int * csrColIndA = in->GetFromMarshal<int*>();
    cuDoubleComplex * A = in->GetFromMarshal<cuDoubleComplex*>();
    int lda = in->Get<int>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseZcsr2dense(handle, m, n, descrA, csrValA, csrRowPtrA, csrColIndA, A, lda);
        out->AddMarshal<cuDoubleComplex*>(A);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseZcsr2dense Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Scsr2csr_compress){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Scsr2csr_compress"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    const int m = in->Get<int>();
    const int n = in->Get<int>();
    const cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    float * csrValA = in->GetFromMarshal<float*>();
    int * csrColIndA = in->GetFromMarshal<int*>();
    int * csrRowPtrA = in->GetFromMarshal<int*>();
    int nnzA = in->Get<int>();
    const int* nnzPerRow = in->GetFromMarshal<int*>();
    float * csrValC = in->GetFromMarshal<float*>();
    int * csrColIndC = in->GetFromMarshal<int*>();
    int * csrRowPtrC = in->GetFromMarshal<int*>();
    float tol = in->Get<float>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseScsr2csr_compress(handle, m, n, descrA, csrValA, csrColIndA, csrRowPtrA, nnzA, nnzPerRow, csrValC, csrColIndC, csrRowPtrC, tol);
        out->AddMarshal<float*>(csrValC);
        out->AddMarshal<int*>(csrColIndC);
        out->AddMarshal<int*>(csrRowPtrC);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseScsr2csr_compress Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Dcsr2csr_compress){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Dcsr2csr_compress"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    const int m = in->Get<int>();
    const int n = in->Get<int>();
    const cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    double * csrValA = in->GetFromMarshal<double*>();
    int * csrColIndA = in->GetFromMarshal<int*>();
    int * csrRowPtrA = in->GetFromMarshal<int*>();
    int nnzA = in->Get<int>();
    const int* nnzPerRow = in->GetFromMarshal<int*>();
    double * csrValC = in->GetFromMarshal<double*>();
    int * csrColIndC = in->GetFromMarshal<int*>();
    int * csrRowPtrC = in->GetFromMarshal<int*>();
    double tol = in->Get<double>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseDcsr2csr_compress(handle, m, n, descrA, csrValA, csrColIndA, csrRowPtrA, nnzA, nnzPerRow, csrValC, csrColIndC, csrRowPtrC, tol);
        out->AddMarshal<double*>(csrValC);
        out->AddMarshal<int*>(csrColIndC);
        out->AddMarshal<int*>(csrRowPtrC);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseDcsr2csr_compress Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Ccsr2csr_compress){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Ccsr2csr_compress"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    const int m = in->Get<int>();
    const int n = in->Get<int>();
    const cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    cuComplex * csrValA = in->GetFromMarshal<cuComplex*>();
    int * csrColIndA = in->GetFromMarshal<int*>();
    int * csrRowPtrA = in->GetFromMarshal<int*>();
    int nnzA = in->Get<int>();
    const int* nnzPerRow = in->GetFromMarshal<int*>();
    cuComplex * csrValC = in->GetFromMarshal<cuComplex*>();
    int * csrColIndC = in->GetFromMarshal<int*>();
    int * csrRowPtrC = in->GetFromMarshal<int*>();
    cuComplex tol = in->Get<cuComplex>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseCcsr2csr_compress(handle, m, n, descrA, csrValA, csrColIndA, csrRowPtrA, nnzA, nnzPerRow, csrValC, csrColIndC, csrRowPtrC, tol);
        out->AddMarshal<cuComplex*>(csrValC);
        out->AddMarshal<int*>(csrColIndC);
        out->AddMarshal<int*>(csrRowPtrC);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseCcsr2csr_compress Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Zcsr2csr_compress){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Zcsr2csr_compress"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    const int m = in->Get<int>();
    const int n = in->Get<int>();
    const cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    cuDoubleComplex * csrValA = in->GetFromMarshal<cuDoubleComplex*>();
    int * csrColIndA = in->GetFromMarshal<int*>();
    int * csrRowPtrA = in->GetFromMarshal<int*>();
    int nnzA = in->Get<int>();
    const int* nnzPerRow = in->GetFromMarshal<int*>();
    cuDoubleComplex * csrValC = in->GetFromMarshal<cuDoubleComplex*>();
    int * csrColIndC = in->GetFromMarshal<int*>();
    int * csrRowPtrC = in->GetFromMarshal<int*>();
    cuDoubleComplex tol = in->Get<cuDoubleComplex>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseZcsr2csr_compress(handle, m, n, descrA, csrValA, csrColIndA, csrRowPtrA, nnzA, nnzPerRow, csrValC, csrColIndC, csrRowPtrC, tol);
        out->AddMarshal<cuDoubleComplex*>(csrValC);
        out->AddMarshal<int*>(csrColIndC);
        out->AddMarshal<int*>(csrRowPtrC);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseZcsr2csr_compress Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Sdense2csc){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Sdense2csc"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    const int m = in->Get<int>();
    const int n = in->Get<int>();
    const cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    const float * A = in->GetFromMarshal<float*>();
    int lda = in->Get<int>();
    const int * nnzPerCol = in->GetFromMarshal<int*>();
    float * cscValA = in->GetFromMarshal<float*>();
    int * cscRowIndA = in->GetFromMarshal<int*>();
    int * cscColPtrA = in->GetFromMarshal<int*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseSdense2csc(handle, m, n, descrA, A, lda, nnzPerCol, cscValA, cscRowIndA, cscColPtrA);
        out->AddMarshal<float*>(cscValA);
        out->AddMarshal<int*>(cscRowIndA);
        out->AddMarshal<int*>(cscColPtrA);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseSdense2csc Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Ddense2csc){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Ddense2csc"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    const int m = in->Get<int>();
    const int n = in->Get<int>();
    const cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    const double * A = in->GetFromMarshal<double*>();
    int lda = in->Get<int>();
    const int * nnzPerCol = in->GetFromMarshal<int*>();
    double * cscValA = in->GetFromMarshal<double*>();
    int * cscRowIndA = in->GetFromMarshal<int*>();
    int * cscColPtrA = in->GetFromMarshal<int*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseDdense2csc(handle, m, n, descrA, A, lda, nnzPerCol, cscValA, cscRowIndA, cscColPtrA);
        out->AddMarshal<double*>(cscValA);
        out->AddMarshal<int*>(cscRowIndA);
        out->AddMarshal<int*>(cscColPtrA);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseDdense2csc Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Cdense2csc){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Cdense2csc"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    const int m = in->Get<int>();
    const int n = in->Get<int>();
    const cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    const cuComplex * A = in->GetFromMarshal<cuComplex*>();
    int lda = in->Get<int>();
    const int * nnzPerCol = in->GetFromMarshal<int*>();
    cuComplex * cscValA = in->GetFromMarshal<cuComplex*>();
    int * cscRowIndA = in->GetFromMarshal<int*>();
    int * cscColPtrA = in->GetFromMarshal<int*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseCdense2csc(handle, m, n, descrA, A, lda, nnzPerCol, cscValA, cscRowIndA, cscColPtrA);
        out->AddMarshal<cuComplex*>(cscValA);
        out->AddMarshal<int*>(cscRowIndA);
        out->AddMarshal<int*>(cscColPtrA);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseCdense2csc Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Zdense2csc){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Zdense2csc"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    const int m = in->Get<int>();
    const int n = in->Get<int>();
    const cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    const cuDoubleComplex * A = in->GetFromMarshal<cuDoubleComplex*>();
    int lda = in->Get<int>();
    const int * nnzPerCol = in->GetFromMarshal<int*>();
    cuDoubleComplex * cscValA = in->GetFromMarshal<cuDoubleComplex*>();
    int * cscRowIndA = in->GetFromMarshal<int*>();
    int * cscColPtrA = in->GetFromMarshal<int*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseZdense2csc(handle, m, n, descrA, A, lda, nnzPerCol, cscValA, cscRowIndA, cscColPtrA);
        out->AddMarshal<cuDoubleComplex*>(cscValA);
        out->AddMarshal<int*>(cscRowIndA);
        out->AddMarshal<int*>(cscColPtrA);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseZdense2csc Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Sdense2csr){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Sdense2csr"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
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
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
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
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
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
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
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
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
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
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
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
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
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
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
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

CUSPARSE_ROUTINE_HANDLER(CreateIdentityPermutation){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("CreateIdentityPermutation"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    int n = in->Get<int>();
    int* p = in->GetFromMarshal<int*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseCreateIdentityPermutation(handle, n, p);
        out->AddMarshal<int*>(p);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseCreateIdentityPermutation Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Xcoosort_bufferSizeExt){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Xcoosort_bufferSizeExt"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    int nnz = in->Get<int>();
    int* cooRows = in->GetFromMarshal<int*>();
    int* cooCols = in->GetFromMarshal<int*>();
    size_t * pBufferSizeInBytes = new size_t;
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseXcoosort_bufferSizeExt(handle, m, n, nnz, cooRows, cooCols, pBufferSizeInBytes);
        out->AddMarshal<size_t*>(pBufferSizeInBytes);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseXcoosort_bufferSizeExt Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(XcoosortByRow){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("XcoosortByRow"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    int nnz = in->Get<int>();
    int* cooRows = in->GetFromMarshal<int*>();
    int* cooCols = in->GetFromMarshal<int*>();
    int* p = in->GetFromMarshal<int*>();
    void * pBuffer = in->GetFromMarshal<void*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseXcoosortByRow(handle, m, n, nnz, cooRows, cooCols, p, pBuffer);
        out->AddMarshal<int*>(cooRows);
        out->AddMarshal<int*>(cooCols);
        out->AddMarshal<int*>(p);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseXcoosortByRow Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(XcoosortByColumn){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("XcoosortByColumn"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    int nnz = in->Get<int>();
    int* cooRows = in->GetFromMarshal<int*>();
    int* cooCols = in->GetFromMarshal<int*>();
    int* p = in->GetFromMarshal<int*>();
    void * pBuffer = in->GetFromMarshal<void*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseXcoosortByColumn(handle, m, n, nnz, cooRows, cooCols, p, pBuffer);
        out->AddMarshal<int*>(cooRows);
        out->AddMarshal<int*>(cooCols);
        out->AddMarshal<int*>(p);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseXcoosortByColumn Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Snnz_compress){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Snnz_compress"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    const cusparseMatDescr_t descr = in->Get<cusparseMatDescr_t>();
    float * csrValA = in->GetFromMarshal<float*>();
    int* csrRowPtrA = in->GetFromMarshal<int*>();
    int* nnzPerRow = in->GetFromMarshal<int*>();
    int nnzC = 0;
    float tol = in->Get<float>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseSnnz_compress(handle, m, descr, csrValA, csrRowPtrA, nnzPerRow, &nnzC, tol);
        out->AddMarshal<int*>(nnzPerRow);
        out->Add<int>(nnzC);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseSnnz_compress Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Dnnz_compress){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Dnnz_compress"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    const cusparseMatDescr_t descr = in->Get<cusparseMatDescr_t>();
    double * csrValA = in->GetFromMarshal<double*>();
    int* csrRowPtrA = in->GetFromMarshal<int*>();
    int* nnzPerRow = in->GetFromMarshal<int*>();
    int nnzC = 0;
    double tol = in->Get<double>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseDnnz_compress(handle, m, descr, csrValA, csrRowPtrA, nnzPerRow, &nnzC, tol);
        out->AddMarshal<int*>(nnzPerRow);
        out->Add<int>(nnzC);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseDnnz_compress Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Cnnz_compress){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Cnnz_compress"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    const cusparseMatDescr_t descr = in->Get<cusparseMatDescr_t>();
    cuComplex * csrValA = in->GetFromMarshal<cuComplex*>();
    int* csrRowPtrA = in->GetFromMarshal<int*>();
    int* nnzPerRow = in->GetFromMarshal<int*>();
    int nnzC = 0;
    cuComplex tol = in->Get<cuComplex>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseCnnz_compress(handle, m, descr, csrValA, csrRowPtrA, nnzPerRow, &nnzC, tol);
        out->AddMarshal<int*>(nnzPerRow);
        out->Add<int>(nnzC);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseCnnz_compress Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Znnz_compress){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Znnz_compress"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    const cusparseMatDescr_t descr = in->Get<cusparseMatDescr_t>();
    cuDoubleComplex * csrValA = in->GetFromMarshal<cuDoubleComplex*>();
    int* csrRowPtrA = in->GetFromMarshal<int*>();
    int* nnzPerRow = in->GetFromMarshal<int*>();
    int nnzC = 0;
    cuDoubleComplex tol = in->Get<cuDoubleComplex>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseZnnz_compress(handle, m, descr, csrValA, csrRowPtrA, nnzPerRow, &nnzC, tol);
        out->AddMarshal<int*>(nnzPerRow);
        out->Add<int>(nnzC);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseZnnz_compress Executed");
    return std::make_shared<Result>(cs,out);
}

#ifndef CUSPARSE_VERSION
#error CUSPARSE_VERSION not defined
#endif
