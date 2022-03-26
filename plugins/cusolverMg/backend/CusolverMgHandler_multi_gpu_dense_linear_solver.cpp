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

#include "CusolverMgHandler.h"

using namespace log4cplus;

using gvirtus::communicators::Buffer;
using gvirtus::communicators::Result;

CUSOLVERMG_ROUTINE_HANDLER(MgPotrf_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("MgPotrf_bufferSize"));
    CusolverMgHandler::setLogLevel(&logger);
    cusolverMgHandle_t handle = (cusolverMgHandle_t)in->Get<size_t>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    int N = in->Get<int>();
    void **array_d_A = in->Assign<void*>(N * N);
    int IA = in->Get<int>();
    int JA = in->Get<int>();
    cudaLibMgMatrixDesc_t descrA = in->Get<cudaLibMgMatrixDesc_t>();
    cudaDataType computeType = in->Get<cudaDataType>();
    int64_t lwork;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverMgPotrf_bufferSize(handle, uplo, N, array_d_A, IA, JA, descrA, computeType, &lwork);
        out->Add<int64_t>(lwork);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverMgPotrf_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVERMG_ROUTINE_HANDLER(MgPotrf){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("MgPotrf"));
    CusolverMgHandler::setLogLevel(&logger);
    cusolverMgHandle_t handle = (cusolverMgHandle_t)in->Get<size_t>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    int N = in->Get<int>();
    void **array_d_A = in->Assign<void*>(N * N);
    int IA = in->Get<int>();
    int JA = in->Get<int>();
    cudaLibMgMatrixDesc_t descrA = in->Get<cudaLibMgMatrixDesc_t>();
    cudaDataType computeType = in->Get<cudaDataType>();
    int64_t lwork = in->Get<int64_t>();
    void **array_d_work = in->Assign<void*>(computeType * lwork);
    int info;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverMgPotrf(handle, uplo, N, array_d_A, IA, JA, descrA, computeType, array_d_work, lwork, &info);
        out->Add<int>(info);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverMgPotrf Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVERMG_ROUTINE_HANDLER(MgPotrs_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("MgPotrs_bufferSize"));
    CusolverMgHandler::setLogLevel(&logger);
    cusolverMgHandle_t handle = (cusolverMgHandle_t)in->Get<size_t>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    int n = in->Get<int>();
    int nrhs = in->Get<int>();
    void **array_d_A = in->Assign<void*>(n * n);
    int IA = in->Get<int>();
    int JA = in->Get<int>();
    cudaLibMgMatrixDesc_t descrA = in->Get<cudaLibMgMatrixDesc_t>();
    void **array_d_B = in->Assign<void*>(n * nrhs);
    int IB = in->Get<int>();
    int JB = in->Get<int>();
    cudaLibMgMatrixDesc_t descrB = in->Get<cudaLibMgMatrixDesc_t>();
    cudaDataType computeType = in->Get<cudaDataType>();
    int64_t lwork;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverMgPotrs_bufferSize(handle, uplo, n, nrhs, array_d_A, IA, JA, descrA, array_d_B, IB, JB, descrB, computeType, &lwork);
        out->Add<int64_t>(lwork);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverMgPotrs_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVERMG_ROUTINE_HANDLER(MgPotrs){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("MgPotrs"));
    CusolverMgHandler::setLogLevel(&logger);
    cusolverMgHandle_t handle = (cusolverMgHandle_t)in->Get<size_t>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    int n = in->Get<int>();
    int nrhs = in->Get<int>();
    void **array_d_A = in->Assign<void*>(n * n);
    int IA = in->Get<int>();
    int JA = in->Get<int>();
    cudaLibMgMatrixDesc_t descrA = in->Get<cudaLibMgMatrixDesc_t>();
    void **array_d_B = in->Assign<void*>(n * nrhs);
    int IB = in->Get<int>();
    int JB = in->Get<int>();
    cudaLibMgMatrixDesc_t descrB = in->Get<cudaLibMgMatrixDesc_t>();
    cudaDataType computeType = in->Get<cudaDataType>();
    int64_t lwork = in->Get<int64_t>();
    void **array_d_work = in->Assign<void*>(lwork);
    int info;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverMgPotrs(handle, uplo, n, nrhs, array_d_A, IA, JA, descrA, array_d_B, IB, JB, descrB, computeType, array_d_work, lwork, &info);
        out->Add<int>(info);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverMgPotrs Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVERMG_ROUTINE_HANDLER(MgPotri_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("MgPotri_bufferSize"));
    CusolverMgHandler::setLogLevel(&logger);
    cusolverMgHandle_t handle = (cusolverMgHandle_t)in->Get<size_t>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    int N = in->Get<int>();
    void **array_d_A = in->Assign<void*>(N * N);
    int IA = in->Get<int>();
    int JA = in->Get<int>();
    cudaLibMgMatrixDesc_t descrA = in->Get<cudaLibMgMatrixDesc_t>();
    cudaDataType computeType = in->Get<cudaDataType>();
    int64_t lwork;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverMgPotri_bufferSize(handle, uplo, N, array_d_A, IA, JA, descrA, computeType, &lwork);
        out->Add<int64_t>(lwork);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverMgPotri_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVERMG_ROUTINE_HANDLER(MgPotri){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("MgPotri"));
    CusolverMgHandler::setLogLevel(&logger);
    cusolverMgHandle_t handle = (cusolverMgHandle_t)in->Get<size_t>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    int N = in->Get<int>();
    void **array_d_A = in->Assign<void*>(N * N);
    int IA = in->Get<int>();
    int JA = in->Get<int>();
    cudaLibMgMatrixDesc_t descrA = in->Get<cudaLibMgMatrixDesc_t>();
    cudaDataType computeType = in->Get<cudaDataType>();
    int64_t lwork = in->Get<int64_t>();
    void **array_d_work = in->Assign<void*>(computeType * lwork);
    int info;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverMgPotri(handle, uplo, N, array_d_A, IA, JA, descrA, computeType, array_d_work, lwork, &info);
        out->Add<int>(info);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverMgPotri Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVERMG_ROUTINE_HANDLER(MgGetrf_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("MgGetrf_bufferSize"));
    CusolverMgHandler::setLogLevel(&logger);
    cusolverMgHandle_t handle = (cusolverMgHandle_t)in->Get<size_t>();
    int M = in->Get<int>();
    int N = in->Get<int>();
    void **array_d_A = in->Assign<void*>(N * N);
    int IA = in->Get<int>();
    int JA = in->Get<int>();
    cudaLibMgMatrixDesc_t descrA = in->Get<cudaLibMgMatrixDesc_t>();
    int **array_d_IPIV = in->Assign<int*>(min(M, N));
    cudaDataType computeType = in->Get<cudaDataType>();
    int64_t lwork;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverMgGetrf_bufferSize(handle, M, N, array_d_A, IA, JA, descrA, array_d_IPIV, computeType, &lwork);
        out->Add<int64_t>(lwork);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverMgGetrf_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVERMG_ROUTINE_HANDLER(MgGetrf){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("MgGetrf"));
    CusolverMgHandler::setLogLevel(&logger);
    cusolverMgHandle_t handle = (cusolverMgHandle_t)in->Get<size_t>();
    int M = in->Get<int>();
    int N = in->Get<int>();
    void **array_d_A = in->Assign<void*>(N * N);
    int IA = in->Get<int>();
    int JA = in->Get<int>();
    cudaLibMgMatrixDesc_t descrA = in->Get<cudaLibMgMatrixDesc_t>();
    int **array_d_IPIV = in->Assign<int*>(min(M, N));
    cudaDataType computeType = in->Get<cudaDataType>();
    int64_t lwork = in->Get<int64_t>();
    void **array_d_work = in->Assign<void*>(computeType * lwork);
    int info;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverMgGetrf(handle, M, N, array_d_A, IA, JA, descrA, array_d_IPIV, computeType, array_d_work, lwork, &info);
        out->Add<int>(info);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverMgGetrf Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVERMG_ROUTINE_HANDLER(MgGetrs_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("MgGetrs_bufferSize"));
    CusolverMgHandler::setLogLevel(&logger);
    cusolverMgHandle_t handle = (cusolverMgHandle_t)in->Get<size_t>();
    cublasOperation_t TRANS = in->Get<cublasOperation_t>();
    int n = in->Get<int>();
    int nrhs = in->Get<int>();
    void **array_d_A = in->Assign<void*>(n * n);
    int IA = in->Get<int>();
    int JA = in->Get<int>();
    cudaLibMgMatrixDesc_t descrA = in->Get<cudaLibMgMatrixDesc_t>();
    int **array_d_IPIV = in->Assign<int*>(min(n, nrhs));
    void **array_d_B = in->Assign<void*>(n * nrhs);
    int IB = in->Get<int>();
    int JB = in->Get<int>();
    cudaLibMgMatrixDesc_t descrB = in->Get<cudaLibMgMatrixDesc_t>();
    cudaDataType computeType = in->Get<cudaDataType>();
    int64_t lwork;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverMgGetrs_bufferSize(handle, TRANS, n, nrhs, array_d_A, IA, JA, descrA, array_d_IPIV, array_d_B, IB, JB, descrB, computeType, &lwork);
        out->Add<int64_t>(lwork);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverMgGetrs_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVERMG_ROUTINE_HANDLER(MgGetrs){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("MgGetrs"));
    CusolverMgHandler::setLogLevel(&logger);
    cusolverMgHandle_t handle = (cusolverMgHandle_t)in->Get<size_t>();
    cublasOperation_t TRANS = in->Get<cublasOperation_t>();
    int n = in->Get<int>();
    int nrhs = in->Get<int>();
    void **array_d_A = in->Assign<void*>(n * n);
    int IA = in->Get<int>();
    int JA = in->Get<int>();
    cudaLibMgMatrixDesc_t descrA = in->Get<cudaLibMgMatrixDesc_t>();
    int **array_d_IPIV = in->Assign<int*>(min(n, nrhs));
    void **array_d_B = in->Assign<void*>(n * nrhs);
    int IB = in->Get<int>();
    int JB = in->Get<int>();
    cudaLibMgMatrixDesc_t descrB = in->Get<cudaLibMgMatrixDesc_t>();
    cudaDataType computeType = in->Get<cudaDataType>();
    int64_t lwork = in->Get<int64_t>();
    void **array_d_work = in->Assign<void*>(lwork);
    int info;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverMgGetrs(handle, TRANS, n, nrhs, array_d_A, IA, JA, descrA, array_d_IPIV, array_d_B, IB, JB, descrB, computeType, array_d_work, lwork, &info);
        out->Add<int>(info);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverMgGetrs Executed");
    return std::make_shared<Result>(cs, out);
}