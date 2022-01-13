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
#include <cuda_runtime.h>

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
    /*
    float hV[4];
    cudaMemcpy(hV, values, 4 * sizeof(float), cudaMemcpyDeviceToHost);
    printf("\tCreateDnVec BEFORE - values: [");
    for (int i = 0; i < 4; i++) {
        printf("%f", hV[i]);
        if (i < 3) {
            printf(", ");
        }
    }
    printf("]\n");
    */
    cudaDataType valueType = in->Get<cudaDataType>();
    //cusparseDnVecDescr_t* dnVecDescr = in->Get<cusparseDnVecDescr_t*>();
    //cusparseDnVecDescr_t * dnVecDescr = in->Assign<cusparseDnVecDescr_t>();
    cusparseDnVecDescr_t * dnVecDescr = new cusparseDnVecDescr_t;
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseCreateDnVec(dnVecDescr, size, values, valueType);
        //printf("\n\tBE - CreateDnVec - dnVecDescr pointer: %p\n", *dnVecDescr);
        /*
        cudaMemcpy(hV, dnVecDescr, 4 * sizeof(float), cudaMemcpyDeviceToHost);
        printf("\tCreateDnVec AFTER - values: [");
        for (int i = 0; i < 4; i++) {
            printf("%f", hV[i]);
            if (i < 3) {
                printf(", ");
            }
        }
        printf("]\n");
        */
        //out->Add<cusparseDnVecDescr_t*>(dnVecDescr);
        //out->Add<cusparseDnVecDescr_t>(dnVecDescr);
        out->Add<size_t>((size_t)*dnVecDescr);
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
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    cusparseOperation_t opA = in->Get<cusparseOperation_t>();
    cusparseSpMatDescr_t matA = (cusparseSpMatDescr_t)in->Get<size_t>();
    cusparseDnVecDescr_t vecX = (cusparseDnVecDescr_t)in->Get<size_t>();
    //printf("\n\tBE - SpMV_bufferSize - vecX pointer: %p\n", vecX);
    cusparseDnVecDescr_t vecY = (cusparseDnVecDescr_t)in->Get<size_t>();
    //printf("\n\tBE - SpMV_bufferSize - vecY pointer: %p\n", vecY);
    cudaDataType computeType = in->Get<cudaDataType>();
    cusparseSpMVAlg_t alg = in->Get<cusparseSpMVAlg_t>();
    size_t * bufferSize = new size_t;
    void* alpha;
    void* beta;
    if (computeType == CUDA_R_32F) {
        // float
        float alphaFloat = in->Get<float>();
        float betaFloat = in->Get<float>();
        alpha = &alphaFloat;
        beta = &betaFloat;
    } else if (computeType == CUDA_R_64F) {
        // double
        double alphaDouble = in->Get<double>();
        double betaDouble = in->Get<double>();
        alpha = &alphaDouble;
        beta = &betaDouble;
    } else if (computeType == CUDA_C_32F) {
        // cuComplex
        cuComplex alphaCuComplex = in->Get<cuComplex>();
        cuComplex betaCuComplex = in->Get<cuComplex>();
        alpha = &alphaCuComplex;
        beta = &betaCuComplex;
    } else if (computeType == CUDA_C_64F) {
        // cuDoubleComplex
        cuDoubleComplex alphaCuDoubleComplex = in->Get<cuDoubleComplex>();
        cuDoubleComplex betaCuDoubleComplex = in->Get<cuDoubleComplex>();
        alpha = &alphaCuDoubleComplex;
        beta = &betaCuDoubleComplex;
    } else {
        throw "Type not supported by GVirtus!";
    }
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
    cusparseSpMatDescr_t matA = (cusparseSpMatDescr_t)in->Get<size_t>();
    cusparseDnVecDescr_t vecX = (cusparseDnVecDescr_t)in->Get<size_t>();
    //printf("\n\tBE - SpMV - vecX pointer: %p\n", vecX);
    cusparseDnVecDescr_t vecY = (cusparseDnVecDescr_t)in->Get<size_t>();
    //printf("\n\tBE - SpMV - vecY pointer: %p\n", vecY);
    cudaDataType computeType = in->Get<cudaDataType>();
    cusparseSpMVAlg_t alg = in->Get<cusparseSpMVAlg_t>();
    void* externalBuffer = in->Get<void*>();
    void* alpha;
    void* beta;
    if (computeType == CUDA_R_32F) {
        // float
        float alphaFloat = in->Get<float>();
        float betaFloat = in->Get<float>();
        alpha = &alphaFloat;
        beta = &betaFloat;
    } else if (computeType == CUDA_R_64F) {
        // double
        double alphaDouble = in->Get<double>();
        double betaDouble = in->Get<double>();
        alpha = &alphaDouble;
        beta = &betaDouble;
    } else if (computeType == CUDA_C_32F) {
        // cuComplex
        cuComplex alphaCuComplex = in->Get<cuComplex>();
        cuComplex betaCuComplex = in->Get<cuComplex>();
        alpha = &alphaCuComplex;
        beta = &betaCuComplex;
    } else if (computeType == CUDA_C_64F) {
        // cuDoubleComplex
        cuDoubleComplex alphaCuDoubleComplex = in->Get<cuDoubleComplex>();
        cuDoubleComplex betaCuDoubleComplex = in->Get<cuDoubleComplex>();
        alpha = &alphaCuDoubleComplex;
        beta = &betaCuDoubleComplex;
    } else {
        throw "Type not supported by GVirtus!";
    }
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseSpMV(handle, opA, alpha, matA, vecX, beta, vecY, computeType, alg, externalBuffer);
        out->Add<size_t>((size_t)vecY);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    } catch(const char *e) {
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseSpMV Executed");
    return std::make_shared<Result>(cs,out);
}

#ifndef CUSPARSE_VERSION
#error CUSPARSE_VERSION not defined
#endif
