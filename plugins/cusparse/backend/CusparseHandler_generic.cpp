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

CUSPARSE_ROUTINE_HANDLER(CreateCsr){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("CreateCsr"));
    CusparseHandler::setLogLevel(&logger);
    int64_t rows = in->Get<int64_t>();
    int64_t cols = in->Get<int64_t>();
    int64_t nnz = in->Get<int64_t>();
    void *csrRowOffsets = in->Get<void*>();
    void *csrColInd = in->Get<void*>();
    void *csrValues = in->Get<void*>();
    cusparseIndexType_t csrRowOffsetsType = in->Get<cusparseIndexType_t>();
    cusparseIndexType_t csrColIndType = in->Get<cusparseIndexType_t>();
    cusparseIndexBase_t idxBase = in->Get<cusparseIndexBase_t>();
    cudaDataType valueType = in->Get<cudaDataType>();
    cusparseSpMatDescr_t * spMatDescr = new cusparseSpMatDescr_t;
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseCreateCsr(spMatDescr, rows, cols, nnz, csrRowOffsets, csrColInd, csrValues, csrRowOffsetsType, csrColIndType, idxBase, valueType);
        out->Add<cusparseSpMatDescr_t>(spMatDescr);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseCreateCsr Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(DestroySpMat){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DestroySpMat"));
    CusparseHandler::setLogLevel(&logger);
    cusparseSpMatDescr_t spMatDescr = (cusparseSpMatDescr_t)in->Get<size_t>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseDestroySpMat(spMatDescr);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseDestroySpMat Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(CreateDnVec){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("CreateDnVec"));
    CusparseHandler::setLogLevel(&logger);
    int64_t size = in->Get<int64_t>();
    void* values = in->Get<void*>();
    cudaDataType valueType = in->Get<cudaDataType>();
    //cusparseDnVecDescr_t* dnVecDescr = in->Get<cusparseDnVecDescr_t*>();
    //cusparseDnVecDescr_t * dnVecDescr = in->Assign<cusparseDnVecDescr_t>();
    cusparseDnVecDescr_t * dnVecDescr = new cusparseDnVecDescr_t;
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseCreateDnVec(dnVecDescr, size, values, valueType);
        //out->Add<cusparseDnVecDescr_t*>(dnVecDescr);
        out->Add<cusparseDnVecDescr_t>(dnVecDescr);
        //out->AddMarshal<cusparseDnVecDescr_t*>(dnVecDescr);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseCreateDnVec Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(DestroyDnVec){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DestroyDnVec"));
    CusparseHandler::setLogLevel(&logger);
    cusparseDnVecDescr_t dnVecDescr = (cusparseDnVecDescr_t)in->Get<size_t>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseDestroyDnVec(dnVecDescr);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseDestroyDnVec Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(SpMV_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SpMV_bufferSize"));
    CusparseHandler::setLogLevel(&logger);

    //cusparseHandle_t handle = in->Get<cusparseHandle_t>();
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    cusparseOperation_t opA = in->Get<cusparseOperation_t>();
    void* alpha=in->Get<void *>();
    //printf("\nalpha address: %d\n", alpha);
    //cusparseSpMatDescr_t matA = in->Get<cusparseSpMatDescr_t>();
    cusparseSpMatDescr_t matA = (cusparseSpMatDescr_t)in->Get<size_t>();
    //cusparseDnVecDescr_t vecX = in->Get<cusparseDnVecDescr_t>();
    cusparseDnVecDescr_t vecX = (cusparseDnVecDescr_t)in->Get<size_t>();
    void* beta = in->Get<void*>();
    //cusparseDnVecDescr_t vecY = in->Get<cusparseDnVecDescr_t>();
    cusparseDnVecDescr_t vecY = (cusparseDnVecDescr_t)in->Get<size_t>();
    cudaDataType computeType = in->Get<cudaDataType>();
    cusparseSpMVAlg_t alg = in->Get<cusparseSpMVAlg_t>();
    //size_t * bufferSize = in->Assign<size_t>();
    size_t * bufferSize = new size_t;
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseSpMV_bufferSize(handle, opA, alpha, matA, vecX, beta, vecY, computeType, alg, bufferSize);
        out->Add<size_t>(bufferSize);
    } catch (string e){
        printf("\nexception\n");
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseSpMV_bufferSize Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(SpMV){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SpMV"));
    CusparseHandler::setLogLevel(&logger);

    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    cusparseOperation_t opA = in->Get<cusparseOperation_t>();
    void* alpha = in->Get<void*>();
    //printf("\nalpha address: %d\n", alpha);
    cusparseSpMatDescr_t matA = (cusparseSpMatDescr_t)in->Get<size_t>();
    cusparseDnVecDescr_t vecX = (cusparseDnVecDescr_t)in->Get<size_t>();
    //printf("\nvecX address: %d\n", vecX);
    void* beta = in->Get<void*>();
    cusparseDnVecDescr_t vecY = (cusparseDnVecDescr_t)in->Get<size_t>();
    //printf("\nvecY address: %d\n", vecY);
    cudaDataType computeType = in->Get<cudaDataType>();
    cusparseSpMVAlg_t alg = in->Get<cusparseSpMVAlg_t>();
    //void* externalBuffer = in->GetFromMarshal<void*>();
    void* externalBuffer = in->Get<void*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        //printf("\nvecY address: %d\n", vecY);
        cs = cusparseSpMV(handle, opA, alpha, matA, vecX, beta, vecY, computeType, alg, externalBuffer);
        out->Add<size_t>((size_t)vecY);
        //printf("\nvecY address: %d\n", vecY);
        //out->Add<cusparseDnVecDescr_t>(vecY);
        //out->AddMarshal<cusparseDnVecDescr_t>(vecY);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseSpMV Executed");
    return std::make_shared<Result>(cs,out);
}

#ifndef CUSPARSE_VERSION
#error CUSPARSE_VERSION not defined
#endif
