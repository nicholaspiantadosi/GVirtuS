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

#include "CusolverHandler.h"

using namespace log4cplus;

using gvirtus::communicators::Buffer;
using gvirtus::communicators::Result;

CUSOLVER_ROUTINE_HANDLER(DnSpotrf_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnSpotrf_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    int n = in->Get<int>();
    float* A = in->GetFromMarshal<float*>();
    int lda = in->Get<int>();
    int Lwork;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnSpotrf_bufferSize(handle, uplo, n, A, lda, &Lwork);
        out->AddMarshal<int>(Lwork);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnSpotrf_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnDpotrf_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnDpotrf_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    int n = in->Get<int>();
    double* A = in->GetFromMarshal<double*>();
    int lda = in->Get<int>();
    int Lwork;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnDpotrf_bufferSize(handle, uplo, n, A, lda, &Lwork);
        out->AddMarshal<int>(Lwork);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnDpotrf_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnCpotrf_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnCpotrf_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    int n = in->Get<int>();
    cuComplex* A = in->GetFromMarshal<cuComplex*>();
    int lda = in->Get<int>();
    int Lwork;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnCpotrf_bufferSize(handle, uplo, n, A, lda, &Lwork);
        out->AddMarshal<int>(Lwork);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnCpotrf_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnZpotrf_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnZpotrf_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    int n = in->Get<int>();
    cuDoubleComplex* A = in->GetFromMarshal<cuDoubleComplex*>();
    int lda = in->Get<int>();
    int Lwork;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnZpotrf_bufferSize(handle, uplo, n, A, lda, &Lwork);
        out->AddMarshal<int>(Lwork);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnZpotrf_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnSpotrf){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnSpotrf"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    int n = in->Get<int>();
    float* A = in->GetFromMarshal<float*>();
    int lda = in->Get<int>();
    float *Workspace = in->GetFromMarshal<float*>();
    int Lwork = in->Get<int>();
    int *devInfo = in->GetFromMarshal<int*>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnSpotrf(handle, uplo, n, A, lda, Workspace, Lwork, devInfo);
        out->Add<int*>(devInfo);
        out->Add<float*>(A);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnSpotrf Executed");
    return std::make_shared<Result>(cs,out);
}

CUSOLVER_ROUTINE_HANDLER(DnDpotrf){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnDpotrf"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    int n = in->Get<int>();
    double* A = in->GetFromMarshal<double*>();
    int lda = in->Get<int>();
    double *Workspace = in->GetFromMarshal<double*>();
    int Lwork = in->Get<int>();
    int *devInfo = in->GetFromMarshal<int*>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnDpotrf(handle, uplo, n, A, lda, Workspace, Lwork, devInfo);
        out->Add<int*>(devInfo);
        out->Add<double*>(A);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnDpotrf Executed");
    return std::make_shared<Result>(cs,out);
}

CUSOLVER_ROUTINE_HANDLER(DnCpotrf){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnCpotrf"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    int n = in->Get<int>();
    cuComplex* A = in->GetFromMarshal<cuComplex*>();
    int lda = in->Get<int>();
    cuComplex *Workspace = in->GetFromMarshal<cuComplex*>();
    int Lwork = in->Get<int>();
    int *devInfo = in->GetFromMarshal<int*>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnCpotrf(handle, uplo, n, A, lda, Workspace, Lwork, devInfo);
        out->Add<int*>(devInfo);
        out->Add<cuComplex*>(A);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnCpotrf Executed");
    return std::make_shared<Result>(cs,out);
}

CUSOLVER_ROUTINE_HANDLER(DnZpotrf){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnZpotrf"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    int n = in->Get<int>();
    cuDoubleComplex* A = in->GetFromMarshal<cuDoubleComplex*>();
    int lda = in->Get<int>();
    cuDoubleComplex *Workspace = in->GetFromMarshal<cuDoubleComplex*>();
    int Lwork = in->Get<int>();
    int *devInfo = in->GetFromMarshal<int*>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnZpotrf(handle, uplo, n, A, lda, Workspace, Lwork, devInfo);
        out->Add<int*>(devInfo);
        out->Add<cuDoubleComplex*>(A);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnZpotrf Executed");
    return std::make_shared<Result>(cs,out);
}

CUSOLVER_ROUTINE_HANDLER(DnPotrf_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnPotrf_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    cusolverDnParams_t params = (cusolverDnParams_t)in->Get<size_t>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    int64_t n = in->Get<int64_t>();
    cudaDataType dataTypeA = in->Get<cudaDataType_t>();
    int64_t lda = in->Get<int64_t>();
    cudaDataType computeType = in->Get<cudaDataType_t>();
    size_t * workspaceInBytes = new size_t;
    void* A;
    if (dataTypeA == CUDA_R_32F) {
        // float
        A = in->GetFromMarshal<float*>();
    } else if (dataTypeA == CUDA_R_64F) {
        // double
        A = in->GetFromMarshal<double*>();
    } else if (dataTypeA == CUDA_C_32F) {
        // cuComplex
        A = in->GetFromMarshal<cuComplex*>();
    } else if (dataTypeA == CUDA_C_64F) {
        // cuDoubleComplex
        A = in->GetFromMarshal<cuDoubleComplex*>();
    } else {
        throw "Type not supported by GVirtus!";
    }
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnPotrf_bufferSize(handle, params, uplo, n, dataTypeA, A, lda, computeType, workspaceInBytes);
        out->Add<size_t>(workspaceInBytes);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnPotrf_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnPotrf){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnPotrf"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    cusolverDnParams_t params = (cusolverDnParams_t)in->Get<size_t>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    int64_t n = in->Get<int64_t>();
    cudaDataType dataTypeA = in->Get<cudaDataType_t>();
    int64_t lda = in->Get<int64_t>();
    cudaDataType computeType = in->Get<cudaDataType_t>();
    void *pBuffer = in->Get<void*>();
    size_t workspaceInBytes = in->Get<size_t>();
    int *info = in->GetFromMarshal<int*>();
    void* A;
    if (dataTypeA == CUDA_R_32F) {
        // float
        A = in->GetFromMarshal<float*>();
    } else if (dataTypeA == CUDA_R_64F) {
        // double
        A = in->GetFromMarshal<double*>();
    } else if (dataTypeA == CUDA_C_32F) {
        // cuComplex
        A = in->GetFromMarshal<cuComplex*>();
    } else if (dataTypeA == CUDA_C_64F) {
        // cuDoubleComplex
        A = in->GetFromMarshal<cuDoubleComplex*>();
    } else {
        throw "Type not supported by GVirtus!";
    }
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnPotrf(handle, params, uplo, n, dataTypeA, A, lda, computeType, pBuffer, workspaceInBytes, info);
        out->Add<int*>(info);
        out->Add<void*>(A);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnPotrf Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnSpotrs){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnSpotrs"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    int n = in->Get<int>();
    int nrhs = in->Get<int>();
    float* A = in->GetFromMarshal<float*>();
    int lda = in->Get<int>();
    float* B = in->GetFromMarshal<float*>();
    int ldb = in->Get<int>();
    int *devInfo = in->GetFromMarshal<int*>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnSpotrs(handle, uplo, n, nrhs, A, lda, B, ldb, devInfo);
        out->Add<int*>(devInfo);
        out->Add<float*>(B);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnSpotrs Executed");
    return std::make_shared<Result>(cs,out);
}

CUSOLVER_ROUTINE_HANDLER(DnDpotrs){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnDpotrs"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    int n = in->Get<int>();
    int nrhs = in->Get<int>();
    double* A = in->GetFromMarshal<double*>();
    int lda = in->Get<int>();
    double* B = in->GetFromMarshal<double*>();
    int ldb = in->Get<int>();
    int *devInfo = in->GetFromMarshal<int*>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnDpotrs(handle, uplo, n, nrhs, A, lda, B, ldb, devInfo);
        out->Add<int*>(devInfo);
        out->Add<double*>(B);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnDpotrs Executed");
    return std::make_shared<Result>(cs,out);
}

CUSOLVER_ROUTINE_HANDLER(DnCpotrs){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnCpotrs"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    int n = in->Get<int>();
    int nrhs = in->Get<int>();
    cuComplex* A = in->GetFromMarshal<cuComplex*>();
    int lda = in->Get<int>();
    cuComplex* B = in->GetFromMarshal<cuComplex*>();
    int ldb = in->Get<int>();
    int *devInfo = in->GetFromMarshal<int*>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnCpotrs(handle, uplo, n, nrhs, A, lda, B, ldb, devInfo);
        out->Add<int*>(devInfo);
        out->Add<cuComplex*>(B);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnCpotrs Executed");
    return std::make_shared<Result>(cs,out);
}

CUSOLVER_ROUTINE_HANDLER(DnZpotrs){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnZpotrs"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    int n = in->Get<int>();
    int nrhs = in->Get<int>();
    cuDoubleComplex* A = in->GetFromMarshal<cuDoubleComplex*>();
    int lda = in->Get<int>();
    cuDoubleComplex* B = in->GetFromMarshal<cuDoubleComplex*>();
    int ldb = in->Get<int>();
    int *devInfo = in->GetFromMarshal<int*>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnZpotrs(handle, uplo, n, nrhs, A, lda, B, ldb, devInfo);
        out->Add<int*>(devInfo);
        out->Add<cuDoubleComplex*>(B);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnZpotrs Executed");
    return std::make_shared<Result>(cs,out);
}

CUSOLVER_ROUTINE_HANDLER(DnPotrs){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnPotrs"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    cusolverDnParams_t params = (cusolverDnParams_t)in->Get<size_t>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    int64_t n = in->Get<int64_t>();
    int64_t nrhs = in->Get<int64_t>();
    cudaDataType dataTypeA = in->Get<cudaDataType_t>();
    int64_t lda = in->Get<int64_t>();
    cudaDataType dataTypeB = in->Get<cudaDataType_t>();
    int64_t ldb = in->Get<int64_t>();
    int *info = in->GetFromMarshal<int*>();
    void* A;
    void* B;
    if (dataTypeA == CUDA_R_32F) {
        // float
        A = in->GetFromMarshal<float*>();
        B = in->GetFromMarshal<float*>();
    } else if (dataTypeA == CUDA_R_64F) {
        // double
        A = in->GetFromMarshal<double*>();
        B = in->GetFromMarshal<double*>();
    } else if (dataTypeA == CUDA_C_32F) {
        // cuComplex
        A = in->GetFromMarshal<cuComplex*>();
        B = in->GetFromMarshal<cuComplex*>();
    } else if (dataTypeA == CUDA_C_64F) {
        // cuDoubleComplex
        A = in->GetFromMarshal<cuDoubleComplex*>();
        B = in->GetFromMarshal<cuDoubleComplex*>();
    } else {
        throw "Type not supported by GVirtus!";
    }
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnPotrs(handle, params, uplo, n, nrhs, dataTypeA, A, lda, dataTypeB, B, ldb, info);
        out->Add<int*>(info);
        out->Add<void*>(B);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnPotrs Executed");
    return std::make_shared<Result>(cs, out);
}

