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

CUSOLVER_ROUTINE_HANDLER(DnSpotri_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnSpotri_bufferSize"));
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
        cs = cusolverDnSpotri_bufferSize(handle, uplo, n, A, lda, &Lwork);
        out->AddMarshal<int>(Lwork);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnSpotri_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnDpotri_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnDpotri_bufferSize"));
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
        cs = cusolverDnDpotri_bufferSize(handle, uplo, n, A, lda, &Lwork);
        out->AddMarshal<int>(Lwork);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnDpotri_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnCpotri_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnCpotri_bufferSize"));
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
        cs = cusolverDnCpotri_bufferSize(handle, uplo, n, A, lda, &Lwork);
        out->AddMarshal<int>(Lwork);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnCpotri_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnZpotri_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnZpotri_bufferSize"));
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
        cs = cusolverDnZpotri_bufferSize(handle, uplo, n, A, lda, &Lwork);
        out->AddMarshal<int>(Lwork);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnZpotri_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnSpotri){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnSpotri"));
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
        cs = cusolverDnSpotri(handle, uplo, n, A, lda, Workspace, Lwork, devInfo);
        out->Add<int*>(devInfo);
        out->Add<float*>(A);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnSpotri Executed");
    return std::make_shared<Result>(cs,out);
}

CUSOLVER_ROUTINE_HANDLER(DnDpotri){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnDpotri"));
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
        cs = cusolverDnDpotri(handle, uplo, n, A, lda, Workspace, Lwork, devInfo);
        out->Add<int*>(devInfo);
        out->Add<double*>(A);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnDpotri Executed");
    return std::make_shared<Result>(cs,out);
}

CUSOLVER_ROUTINE_HANDLER(DnCpotri){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnCpotri"));
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
        cs = cusolverDnCpotri(handle, uplo, n, A, lda, Workspace, Lwork, devInfo);
        out->Add<int*>(devInfo);
        out->Add<cuComplex*>(A);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnCpotri Executed");
    return std::make_shared<Result>(cs,out);
}

CUSOLVER_ROUTINE_HANDLER(DnZpotri){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnZpotri"));
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
        cs = cusolverDnZpotri(handle, uplo, n, A, lda, Workspace, Lwork, devInfo);
        out->Add<int*>(devInfo);
        out->Add<cuDoubleComplex*>(A);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnZpotri Executed");
    return std::make_shared<Result>(cs,out);
}

CUSOLVER_ROUTINE_HANDLER(DnSgetrf_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnSgetrf_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    float* A = in->GetFromMarshal<float*>();
    int lda = in->Get<int>();
    int Lwork;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnSgetrf_bufferSize(handle, m, n, A, lda, &Lwork);
        out->AddMarshal<int>(Lwork);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnSgetrf_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnDgetrf_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnDgetrf_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    double* A = in->GetFromMarshal<double*>();
    int lda = in->Get<int>();
    int Lwork;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnDgetrf_bufferSize(handle, m, n, A, lda, &Lwork);
        out->AddMarshal<int>(Lwork);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnDgetrf_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnCgetrf_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnCgetrf_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    cuComplex* A = in->GetFromMarshal<cuComplex*>();
    int lda = in->Get<int>();
    int Lwork;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnCgetrf_bufferSize(handle, m, n, A, lda, &Lwork);
        out->AddMarshal<int>(Lwork);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnCgetrf_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnZgetrf_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnZgetrf_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    cuDoubleComplex* A = in->GetFromMarshal<cuDoubleComplex*>();
    int lda = in->Get<int>();
    int Lwork;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnZgetrf_bufferSize(handle, m, n, A, lda, &Lwork);
        out->AddMarshal<int>(Lwork);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnZgetrf_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnSgetrf){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnSgetrf"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    float* A = in->GetFromMarshal<float*>();
    int lda = in->Get<int>();
    float *Workspace = in->GetFromMarshal<float*>();
    int *devIpiv = in->GetFromMarshal<int*>();
    int *devInfo = in->GetFromMarshal<int*>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnSgetrf(handle, m, n, A, lda, Workspace, devIpiv, devInfo);
        out->Add<int*>(devIpiv);
        out->Add<int*>(devInfo);
        out->Add<float*>(A);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnSgetrf Executed");
    return std::make_shared<Result>(cs,out);
}

CUSOLVER_ROUTINE_HANDLER(DnDgetrf){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnDgetrf"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    double* A = in->GetFromMarshal<double*>();
    int lda = in->Get<int>();
    double *Workspace = in->GetFromMarshal<double*>();
    int *devIpiv = in->GetFromMarshal<int*>();
    int *devInfo = in->GetFromMarshal<int*>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnDgetrf(handle, m, n, A, lda, Workspace, devIpiv, devInfo);
        out->Add<int*>(devIpiv);
        out->Add<int*>(devInfo);
        out->Add<double*>(A);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnDgetrf Executed");
    return std::make_shared<Result>(cs,out);
}

CUSOLVER_ROUTINE_HANDLER(DnCgetrf){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnCgetrf"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    cuComplex* A = in->GetFromMarshal<cuComplex*>();
    int lda = in->Get<int>();
    cuComplex *Workspace = in->GetFromMarshal<cuComplex*>();
    int *devIpiv = in->GetFromMarshal<int*>();
    int *devInfo = in->GetFromMarshal<int*>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnCgetrf(handle, m, n, A, lda, Workspace, devIpiv, devInfo);
        out->Add<int*>(devIpiv);
        out->Add<int*>(devInfo);
        out->Add<cuComplex*>(A);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnCgetrf Executed");
    return std::make_shared<Result>(cs,out);
}

CUSOLVER_ROUTINE_HANDLER(DnZgetrf){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnZgetrf"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    cuDoubleComplex* A = in->GetFromMarshal<cuDoubleComplex*>();
    int lda = in->Get<int>();
    cuDoubleComplex *Workspace = in->GetFromMarshal<cuDoubleComplex*>();
    int *devIpiv = in->GetFromMarshal<int*>();
    int *devInfo = in->GetFromMarshal<int*>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnZgetrf(handle, m, n, A, lda, Workspace, devIpiv, devInfo);
        out->Add<int*>(devIpiv);
        out->Add<int*>(devInfo);
        out->Add<cuDoubleComplex*>(A);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnZgetrf Executed");
    return std::make_shared<Result>(cs,out);
}

CUSOLVER_ROUTINE_HANDLER(DnGetrf_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnGetrf_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    cusolverDnParams_t params = (cusolverDnParams_t)in->Get<size_t>();
    int64_t m = in->Get<int64_t>();
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
        cs = cusolverDnGetrf_bufferSize(handle, params, m, n, dataTypeA, A, lda, computeType, workspaceInBytes);
        out->Add<size_t>(workspaceInBytes);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnGetrf_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnGetrf){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnGetrf"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    cusolverDnParams_t params = (cusolverDnParams_t)in->Get<size_t>();
    int64_t m = in->Get<int64_t>();
    int64_t n = in->Get<int64_t>();
    cudaDataType dataTypeA = in->Get<cudaDataType_t>();
    int64_t lda = in->Get<int64_t>();
    int64_t *ipiv = in->Get<int64_t*>();
    cudaDataType computeType = in->Get<cudaDataType_t>();
    void *pBuffer = in->Get<void*>();
    size_t workspaceInBytes = in->Get<size_t >();
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
        cs = cusolverDnGetrf(handle, params, m, n, dataTypeA, A, lda, ipiv, computeType, pBuffer, workspaceInBytes, info);
        out->Add<int64_t*>(ipiv);
        out->Add<int*>(info);
        out->Add<void*>(A);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnGetrf Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnSgetrs){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnSgetrs"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    cublasOperation_t trans = in->Get<cublasOperation_t>();
    int n = in->Get<int>();
    int nrhs = in->Get<int>();
    float* A = in->GetFromMarshal<float*>();
    int lda = in->Get<int>();
    int *devIpiv = in->GetFromMarshal<int*>();
    float* B = in->GetFromMarshal<float*>();
    int ldb = in->Get<int>();
    int *devInfo = in->GetFromMarshal<int*>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnSgetrs(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo);
        out->Add<int*>(devInfo);
        out->Add<float*>(B);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnSgetrs Executed");
    return std::make_shared<Result>(cs,out);
}

CUSOLVER_ROUTINE_HANDLER(DnDgetrs){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnDgetrs"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    cublasOperation_t trans = in->Get<cublasOperation_t>();
    int n = in->Get<int>();
    int nrhs = in->Get<int>();
    double* A = in->GetFromMarshal<double*>();
    int lda = in->Get<int>();
    int *devIpiv = in->GetFromMarshal<int*>();
    double* B = in->GetFromMarshal<double*>();
    int ldb = in->Get<int>();
    int *devInfo = in->GetFromMarshal<int*>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnDgetrs(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo);
        out->Add<int*>(devInfo);
        out->Add<double*>(B);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnDgetrs Executed");
    return std::make_shared<Result>(cs,out);
}

CUSOLVER_ROUTINE_HANDLER(DnCgetrs){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnCgetrs"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    cublasOperation_t trans = in->Get<cublasOperation_t>();
    int n = in->Get<int>();
    int nrhs = in->Get<int>();
    cuComplex* A = in->GetFromMarshal<cuComplex*>();
    int lda = in->Get<int>();
    int *devIpiv = in->GetFromMarshal<int*>();
    cuComplex* B = in->GetFromMarshal<cuComplex*>();
    int ldb = in->Get<int>();
    int *devInfo = in->GetFromMarshal<int*>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnCgetrs(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo);
        out->Add<int*>(devInfo);
        out->Add<cuComplex*>(B);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnCgetrs Executed");
    return std::make_shared<Result>(cs,out);
}

CUSOLVER_ROUTINE_HANDLER(DnZgetrs){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnZgetrs"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    cublasOperation_t trans = in->Get<cublasOperation_t>();
    int n = in->Get<int>();
    int nrhs = in->Get<int>();
    cuDoubleComplex* A = in->GetFromMarshal<cuDoubleComplex*>();
    int lda = in->Get<int>();
    int *devIpiv = in->GetFromMarshal<int*>();
    cuDoubleComplex* B = in->GetFromMarshal<cuDoubleComplex*>();
    int ldb = in->Get<int>();
    int *devInfo = in->GetFromMarshal<int*>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnZgetrs(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo);
        out->Add<int*>(devInfo);
        out->Add<cuDoubleComplex*>(B);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnZgetrs Executed");
    return std::make_shared<Result>(cs,out);
}

CUSOLVER_ROUTINE_HANDLER(DnGetrs){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnGetrs"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    cusolverDnParams_t params = (cusolverDnParams_t)in->Get<size_t>();
    cublasOperation_t trans = in->Get<cublasOperation_t>();
    int64_t n = in->Get<int64_t>();
    int64_t nrhs = in->Get<int64_t>();
    cudaDataType dataTypeA = in->Get<cudaDataType_t>();
    int64_t lda = in->Get<int64_t>();
    int64_t *ipiv = in->Get<int64_t*>();
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
        cs = cusolverDnGetrs(handle, params, trans, n, nrhs, dataTypeA, A, lda, ipiv, dataTypeB, B, ldb, info);
        out->Add<int*>(info);
        out->Add<void*>(B);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnGetrs Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnZZgesv_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnZZgesv_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int n = in->Get<int>();
    int nrhs = in->Get<int>();
    cuDoubleComplex *dA = in->GetFromMarshal<cuDoubleComplex*>();
    int ldda = in->Get<int>();
    int *dipiv = in->Get<int*>();
    cuDoubleComplex *dB = in->GetFromMarshal<cuDoubleComplex*>();
    int lddb = in->Get<int>();
    cuDoubleComplex *dX = in->GetFromMarshal<cuDoubleComplex*>();
    int lddx = in->Get<int>();
    void *dWork = in->Get<void*>();
    size_t * workspaceInBytes = new size_t;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnZZgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWork, workspaceInBytes);
        out->Add<size_t>(workspaceInBytes);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnZZgesv_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnZCgesv_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnZCgesv_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int n = in->Get<int>();
    int nrhs = in->Get<int>();
    cuDoubleComplex *dA = in->GetFromMarshal<cuDoubleComplex*>();
    int ldda = in->Get<int>();
    int *dipiv = in->Get<int*>();
    cuDoubleComplex *dB = in->GetFromMarshal<cuDoubleComplex*>();
    int lddb = in->Get<int>();
    cuDoubleComplex *dX = in->GetFromMarshal<cuDoubleComplex*>();
    int lddx = in->Get<int>();
    void *dWork = in->Get<void*>();
    size_t * workspaceInBytes = new size_t;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnZCgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWork, workspaceInBytes);
        out->Add<size_t>(workspaceInBytes);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnZCgesv_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnZKgesv_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnZKgesv_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int n = in->Get<int>();
    int nrhs = in->Get<int>();
    cuDoubleComplex *dA = in->GetFromMarshal<cuDoubleComplex*>();
    int ldda = in->Get<int>();
    int *dipiv = in->Get<int*>();
    cuDoubleComplex *dB = in->GetFromMarshal<cuDoubleComplex*>();
    int lddb = in->Get<int>();
    cuDoubleComplex *dX = in->GetFromMarshal<cuDoubleComplex*>();
    int lddx = in->Get<int>();
    void *dWork = in->Get<void*>();
    size_t * workspaceInBytes = new size_t;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnZKgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWork, workspaceInBytes);
        out->Add<size_t>(workspaceInBytes);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnZKgesv_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnZEgesv_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnZEgesv_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int n = in->Get<int>();
    int nrhs = in->Get<int>();
    cuDoubleComplex *dA = in->GetFromMarshal<cuDoubleComplex*>();
    int ldda = in->Get<int>();
    int *dipiv = in->Get<int*>();
    cuDoubleComplex *dB = in->GetFromMarshal<cuDoubleComplex*>();
    int lddb = in->Get<int>();
    cuDoubleComplex *dX = in->GetFromMarshal<cuDoubleComplex*>();
    int lddx = in->Get<int>();
    void *dWork = in->Get<void*>();
    size_t * workspaceInBytes = new size_t;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnZEgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWork, workspaceInBytes);
        out->Add<size_t>(workspaceInBytes);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnZEgesv_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnZYgesv_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnZYgesv_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int n = in->Get<int>();
    int nrhs = in->Get<int>();
    cuDoubleComplex *dA = in->GetFromMarshal<cuDoubleComplex*>();
    int ldda = in->Get<int>();
    int *dipiv = in->Get<int*>();
    cuDoubleComplex *dB = in->GetFromMarshal<cuDoubleComplex*>();
    int lddb = in->Get<int>();
    cuDoubleComplex *dX = in->GetFromMarshal<cuDoubleComplex*>();
    int lddx = in->Get<int>();
    void *dWork = in->Get<void*>();
    size_t * workspaceInBytes = new size_t;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnZYgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWork, workspaceInBytes);
        out->Add<size_t>(workspaceInBytes);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnZYgesv_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnCCgesv_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnCCgesv_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int n = in->Get<int>();
    int nrhs = in->Get<int>();
    cuComplex *dA = in->GetFromMarshal<cuComplex*>();
    int ldda = in->Get<int>();
    int *dipiv = in->Get<int*>();
    cuComplex *dB = in->GetFromMarshal<cuComplex*>();
    int lddb = in->Get<int>();
    cuComplex *dX = in->GetFromMarshal<cuComplex*>();
    int lddx = in->Get<int>();
    void *dWork = in->Get<void*>();
    size_t * workspaceInBytes = new size_t;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnCCgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWork, workspaceInBytes);
        out->Add<size_t>(workspaceInBytes);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnCCgesv_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnCKgesv_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnCKgesv_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int n = in->Get<int>();
    int nrhs = in->Get<int>();
    cuComplex *dA = in->GetFromMarshal<cuComplex*>();
    int ldda = in->Get<int>();
    int *dipiv = in->Get<int*>();
    cuComplex *dB = in->GetFromMarshal<cuComplex*>();
    int lddb = in->Get<int>();
    cuComplex *dX = in->GetFromMarshal<cuComplex*>();
    int lddx = in->Get<int>();
    void *dWork = in->Get<void*>();
    size_t * workspaceInBytes = new size_t;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnCKgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWork, workspaceInBytes);
        out->Add<size_t>(workspaceInBytes);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnCKgesv_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnCEgesv_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnCEgesv_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int n = in->Get<int>();
    int nrhs = in->Get<int>();
    cuComplex *dA = in->GetFromMarshal<cuComplex*>();
    int ldda = in->Get<int>();
    int *dipiv = in->Get<int*>();
    cuComplex *dB = in->GetFromMarshal<cuComplex*>();
    int lddb = in->Get<int>();
    cuComplex *dX = in->GetFromMarshal<cuComplex*>();
    int lddx = in->Get<int>();
    void *dWork = in->Get<void*>();
    size_t * workspaceInBytes = new size_t;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnCEgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWork, workspaceInBytes);
        out->Add<size_t>(workspaceInBytes);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnCEgesv_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnCYgesv_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnCYgesv_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int n = in->Get<int>();
    int nrhs = in->Get<int>();
    cuComplex *dA = in->GetFromMarshal<cuComplex*>();
    int ldda = in->Get<int>();
    int *dipiv = in->Get<int*>();
    cuComplex *dB = in->GetFromMarshal<cuComplex*>();
    int lddb = in->Get<int>();
    cuComplex *dX = in->GetFromMarshal<cuComplex*>();
    int lddx = in->Get<int>();
    void *dWork = in->Get<void*>();
    size_t * workspaceInBytes = new size_t;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnCYgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWork, workspaceInBytes);
        out->Add<size_t>(workspaceInBytes);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnCYgesv_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnDDgesv_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnDDgesv_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int n = in->Get<int>();
    int nrhs = in->Get<int>();
    double *dA = in->GetFromMarshal<double*>();
    int ldda = in->Get<int>();
    int *dipiv = in->Get<int*>();
    double *dB = in->GetFromMarshal<double*>();
    int lddb = in->Get<int>();
    double *dX = in->GetFromMarshal<double*>();
    int lddx = in->Get<int>();
    void *dWork = in->Get<void*>();
    size_t * workspaceInBytes = new size_t;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnDDgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWork, workspaceInBytes);
        out->Add<size_t>(workspaceInBytes);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnDDgesv_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnDSgesv_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnDSgesv_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int n = in->Get<int>();
    int nrhs = in->Get<int>();
    double *dA = in->GetFromMarshal<double*>();
    int ldda = in->Get<int>();
    int *dipiv = in->Get<int*>();
    double *dB = in->GetFromMarshal<double*>();
    int lddb = in->Get<int>();
    double *dX = in->GetFromMarshal<double*>();
    int lddx = in->Get<int>();
    void *dWork = in->Get<void*>();
    size_t * workspaceInBytes = new size_t;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnDSgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWork, workspaceInBytes);
        out->Add<size_t>(workspaceInBytes);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnDSgesv_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnDHgesv_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnDHgesv_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int n = in->Get<int>();
    int nrhs = in->Get<int>();
    double *dA = in->GetFromMarshal<double*>();
    int ldda = in->Get<int>();
    int *dipiv = in->Get<int*>();
    double *dB = in->GetFromMarshal<double*>();
    int lddb = in->Get<int>();
    double *dX = in->GetFromMarshal<double*>();
    int lddx = in->Get<int>();
    void *dWork = in->Get<void*>();
    size_t * workspaceInBytes = new size_t;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnDHgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWork, workspaceInBytes);
        out->Add<size_t>(workspaceInBytes);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnDHgesv_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnDBgesv_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnDBgesv_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int n = in->Get<int>();
    int nrhs = in->Get<int>();
    double *dA = in->GetFromMarshal<double*>();
    int ldda = in->Get<int>();
    int *dipiv = in->Get<int*>();
    double *dB = in->GetFromMarshal<double*>();
    int lddb = in->Get<int>();
    double *dX = in->GetFromMarshal<double*>();
    int lddx = in->Get<int>();
    void *dWork = in->Get<void*>();
    size_t * workspaceInBytes = new size_t;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnDBgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWork, workspaceInBytes);
        out->Add<size_t>(workspaceInBytes);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnDBgesv_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnDXgesv_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnDXgesv_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int n = in->Get<int>();
    int nrhs = in->Get<int>();
    double *dA = in->GetFromMarshal<double*>();
    int ldda = in->Get<int>();
    int *dipiv = in->Get<int*>();
    double *dB = in->GetFromMarshal<double*>();
    int lddb = in->Get<int>();
    double *dX = in->GetFromMarshal<double*>();
    int lddx = in->Get<int>();
    void *dWork = in->Get<void*>();
    size_t * workspaceInBytes = new size_t;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnDXgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWork, workspaceInBytes);
        out->Add<size_t>(workspaceInBytes);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnDXgesv_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnSSgesv_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnSSgesv_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int n = in->Get<int>();
    int nrhs = in->Get<int>();
    float *dA = in->GetFromMarshal<float*>();
    int ldda = in->Get<int>();
    int *dipiv = in->Get<int*>();
    float *dB = in->GetFromMarshal<float*>();
    int lddb = in->Get<int>();
    float *dX = in->GetFromMarshal<float*>();
    int lddx = in->Get<int>();
    void *dWork = in->Get<void*>();
    size_t * workspaceInBytes = new size_t;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnSSgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWork, workspaceInBytes);
        out->Add<size_t>(workspaceInBytes);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnSSgesv_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnSHgesv_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnSHgesv_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int n = in->Get<int>();
    int nrhs = in->Get<int>();
    float *dA = in->GetFromMarshal<float*>();
    int ldda = in->Get<int>();
    int *dipiv = in->Get<int*>();
    float *dB = in->GetFromMarshal<float*>();
    int lddb = in->Get<int>();
    float *dX = in->GetFromMarshal<float*>();
    int lddx = in->Get<int>();
    void *dWork = in->Get<void*>();
    size_t * workspaceInBytes = new size_t;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnSHgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWork, workspaceInBytes);
        out->Add<size_t>(workspaceInBytes);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnSHgesv_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnSBgesv_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnSBgesv_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int n = in->Get<int>();
    int nrhs = in->Get<int>();
    float *dA = in->GetFromMarshal<float*>();
    int ldda = in->Get<int>();
    int *dipiv = in->Get<int*>();
    float *dB = in->GetFromMarshal<float*>();
    int lddb = in->Get<int>();
    float *dX = in->GetFromMarshal<float*>();
    int lddx = in->Get<int>();
    void *dWork = in->Get<void*>();
    size_t * workspaceInBytes = new size_t;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnSBgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWork, workspaceInBytes);
        out->Add<size_t>(workspaceInBytes);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnSBgesv_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnSXgesv_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnSXgesv_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int n = in->Get<int>();
    int nrhs = in->Get<int>();
    float *dA = in->GetFromMarshal<float*>();
    int ldda = in->Get<int>();
    int *dipiv = in->Get<int*>();
    float *dB = in->GetFromMarshal<float*>();
    int lddb = in->Get<int>();
    float *dX = in->GetFromMarshal<float*>();
    int lddx = in->Get<int>();
    void *dWork = in->Get<void*>();
    size_t * workspaceInBytes = new size_t;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnSXgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWork, workspaceInBytes);
        out->Add<size_t>(workspaceInBytes);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnSXgesv_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnZZgesv){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnZZgesv"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int n = in->Get<int>();
    int nrhs = in->Get<int>();
    cuDoubleComplex *dA = in->GetFromMarshal<cuDoubleComplex*>();
    int ldda = in->Get<int>();
    int *dipiv = in->Get<int*>();
    cuDoubleComplex *dB = in->GetFromMarshal<cuDoubleComplex*>();
    int lddb = in->Get<int>();
    cuDoubleComplex *dX = in->GetFromMarshal<cuDoubleComplex*>();
    int lddx = in->Get<int>();
    void *dWorkspace = in->Get<void*>();
    size_t lwork_bytes = in->Get<size_t>();
    int niter;
    int *dinfo = in->Get<int*>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnZZgesv(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, &niter, dinfo);
        out->Add<cuDoubleComplex*>(dA);
        out->Add<int*>(dipiv);
        out->Add<cuDoubleComplex*>(dX);
        out->Add<int>(niter);
        out->Add<int*>(dinfo);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnZZgesv Executed");
    return std::make_shared<Result>(cs,out);
}

CUSOLVER_ROUTINE_HANDLER(DnZCgesv){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnZCgesv"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int n = in->Get<int>();
    int nrhs = in->Get<int>();
    cuDoubleComplex *dA = in->GetFromMarshal<cuDoubleComplex*>();
    int ldda = in->Get<int>();
    int *dipiv = in->Get<int*>();
    cuDoubleComplex *dB = in->GetFromMarshal<cuDoubleComplex*>();
    int lddb = in->Get<int>();
    cuDoubleComplex *dX = in->GetFromMarshal<cuDoubleComplex*>();
    int lddx = in->Get<int>();
    void *dWorkspace = in->Get<void*>();
    size_t lwork_bytes = in->Get<size_t>();
    int niter;
    int *dinfo = in->Get<int*>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnZCgesv(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, &niter, dinfo);
        out->Add<cuDoubleComplex*>(dA);
        out->Add<int*>(dipiv);
        out->Add<cuDoubleComplex*>(dX);
        out->Add<int>(niter);
        out->Add<int*>(dinfo);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnZCgesv Executed");
    return std::make_shared<Result>(cs,out);
}

CUSOLVER_ROUTINE_HANDLER(DnZKgesv){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnZKgesv"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int n = in->Get<int>();
    int nrhs = in->Get<int>();
    cuDoubleComplex *dA = in->GetFromMarshal<cuDoubleComplex*>();
    int ldda = in->Get<int>();
    int *dipiv = in->Get<int*>();
    cuDoubleComplex *dB = in->GetFromMarshal<cuDoubleComplex*>();
    int lddb = in->Get<int>();
    cuDoubleComplex *dX = in->GetFromMarshal<cuDoubleComplex*>();
    int lddx = in->Get<int>();
    void *dWorkspace = in->Get<void*>();
    size_t lwork_bytes = in->Get<size_t>();
    int niter;
    int *dinfo = in->Get<int*>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnZKgesv(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, &niter, dinfo);
        out->Add<cuDoubleComplex*>(dA);
        out->Add<int*>(dipiv);
        out->Add<cuDoubleComplex*>(dX);
        out->Add<int>(niter);
        out->Add<int*>(dinfo);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnZKgesv Executed");
    return std::make_shared<Result>(cs,out);
}

CUSOLVER_ROUTINE_HANDLER(DnZEgesv){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnZEgesv"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int n = in->Get<int>();
    int nrhs = in->Get<int>();
    cuDoubleComplex *dA = in->GetFromMarshal<cuDoubleComplex*>();
    int ldda = in->Get<int>();
    int *dipiv = in->Get<int*>();
    cuDoubleComplex *dB = in->GetFromMarshal<cuDoubleComplex*>();
    int lddb = in->Get<int>();
    cuDoubleComplex *dX = in->GetFromMarshal<cuDoubleComplex*>();
    int lddx = in->Get<int>();
    void *dWorkspace = in->Get<void*>();
    size_t lwork_bytes = in->Get<size_t>();
    int niter;
    int *dinfo = in->Get<int*>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnZEgesv(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, &niter, dinfo);
        out->Add<cuDoubleComplex*>(dA);
        out->Add<int*>(dipiv);
        out->Add<cuDoubleComplex*>(dX);
        out->Add<int>(niter);
        out->Add<int*>(dinfo);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnZEgesv Executed");
    return std::make_shared<Result>(cs,out);
}

CUSOLVER_ROUTINE_HANDLER(DnZYgesv){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnZYgesv"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int n = in->Get<int>();
    int nrhs = in->Get<int>();
    cuDoubleComplex *dA = in->GetFromMarshal<cuDoubleComplex*>();
    int ldda = in->Get<int>();
    int *dipiv = in->Get<int*>();
    cuDoubleComplex *dB = in->GetFromMarshal<cuDoubleComplex*>();
    int lddb = in->Get<int>();
    cuDoubleComplex *dX = in->GetFromMarshal<cuDoubleComplex*>();
    int lddx = in->Get<int>();
    void *dWorkspace = in->Get<void*>();
    size_t lwork_bytes = in->Get<size_t>();
    int niter;
    int *dinfo = in->Get<int*>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnZYgesv(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, &niter, dinfo);
        out->Add<cuDoubleComplex*>(dA);
        out->Add<int*>(dipiv);
        out->Add<cuDoubleComplex*>(dX);
        out->Add<int>(niter);
        out->Add<int*>(dinfo);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnZYgesv Executed");
    return std::make_shared<Result>(cs,out);
}

CUSOLVER_ROUTINE_HANDLER(DnCCgesv){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnCCgesv"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int n = in->Get<int>();
    int nrhs = in->Get<int>();
    cuComplex *dA = in->GetFromMarshal<cuComplex*>();
    int ldda = in->Get<int>();
    int *dipiv = in->Get<int*>();
    cuComplex *dB = in->GetFromMarshal<cuComplex*>();
    int lddb = in->Get<int>();
    cuComplex *dX = in->GetFromMarshal<cuComplex*>();
    int lddx = in->Get<int>();
    void *dWorkspace = in->Get<void*>();
    size_t lwork_bytes = in->Get<size_t>();
    int niter;
    int *dinfo = in->Get<int*>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnCCgesv(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, &niter, dinfo);
        out->Add<cuComplex*>(dA);
        out->Add<int*>(dipiv);
        out->Add<cuComplex*>(dX);
        out->Add<int>(niter);
        out->Add<int*>(dinfo);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnCCgesv Executed");
    return std::make_shared<Result>(cs,out);
}

CUSOLVER_ROUTINE_HANDLER(DnCKgesv){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnCKgesv"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int n = in->Get<int>();
    int nrhs = in->Get<int>();
    cuComplex *dA = in->GetFromMarshal<cuComplex*>();
    int ldda = in->Get<int>();
    int *dipiv = in->Get<int*>();
    cuComplex *dB = in->GetFromMarshal<cuComplex*>();
    int lddb = in->Get<int>();
    cuComplex *dX = in->GetFromMarshal<cuComplex*>();
    int lddx = in->Get<int>();
    void *dWorkspace = in->Get<void*>();
    size_t lwork_bytes = in->Get<size_t>();
    int niter;
    int *dinfo = in->Get<int*>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnCKgesv(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, &niter, dinfo);
        out->Add<cuComplex*>(dA);
        out->Add<int*>(dipiv);
        out->Add<cuComplex*>(dX);
        out->Add<int>(niter);
        out->Add<int*>(dinfo);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnCKgesv Executed");
    return std::make_shared<Result>(cs,out);
}

CUSOLVER_ROUTINE_HANDLER(DnCEgesv){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnCEgesv"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int n = in->Get<int>();
    int nrhs = in->Get<int>();
    cuComplex *dA = in->GetFromMarshal<cuComplex*>();
    int ldda = in->Get<int>();
    int *dipiv = in->Get<int*>();
    cuComplex *dB = in->GetFromMarshal<cuComplex*>();
    int lddb = in->Get<int>();
    cuComplex *dX = in->GetFromMarshal<cuComplex*>();
    int lddx = in->Get<int>();
    void *dWorkspace = in->Get<void*>();
    size_t lwork_bytes = in->Get<size_t>();
    int niter;
    int *dinfo = in->Get<int*>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnCEgesv(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, &niter, dinfo);
        out->Add<cuComplex*>(dA);
        out->Add<int*>(dipiv);
        out->Add<cuComplex*>(dX);
        out->Add<int>(niter);
        out->Add<int*>(dinfo);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnCEgesv Executed");
    return std::make_shared<Result>(cs,out);
}

CUSOLVER_ROUTINE_HANDLER(DnCYgesv){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnCYgesv"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int n = in->Get<int>();
    int nrhs = in->Get<int>();
    cuComplex *dA = in->GetFromMarshal<cuComplex*>();
    int ldda = in->Get<int>();
    int *dipiv = in->Get<int*>();
    cuComplex *dB = in->GetFromMarshal<cuComplex*>();
    int lddb = in->Get<int>();
    cuComplex *dX = in->GetFromMarshal<cuComplex*>();
    int lddx = in->Get<int>();
    void *dWorkspace = in->Get<void*>();
    size_t lwork_bytes = in->Get<size_t>();
    int niter;
    int *dinfo = in->Get<int*>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnCYgesv(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, &niter, dinfo);
        out->Add<cuComplex*>(dA);
        out->Add<int*>(dipiv);
        out->Add<cuComplex*>(dX);
        out->Add<int>(niter);
        out->Add<int*>(dinfo);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnCYgesv Executed");
    return std::make_shared<Result>(cs,out);
}

CUSOLVER_ROUTINE_HANDLER(DnDDgesv){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnDDgesv"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int n = in->Get<int>();
    int nrhs = in->Get<int>();
    double *dA = in->GetFromMarshal<double*>();
    int ldda = in->Get<int>();
    int *dipiv = in->Get<int*>();
    double *dB = in->GetFromMarshal<double*>();
    int lddb = in->Get<int>();
    double *dX = in->GetFromMarshal<double*>();
    int lddx = in->Get<int>();
    void *dWorkspace = in->Get<void*>();
    size_t lwork_bytes = in->Get<size_t>();
    int niter;
    int *dinfo = in->Get<int*>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnDDgesv(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, &niter, dinfo);
        out->Add<double*>(dA);
        out->Add<int*>(dipiv);
        out->Add<double*>(dX);
        out->Add<int>(niter);
        out->Add<int*>(dinfo);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnDDgesv Executed");
    return std::make_shared<Result>(cs,out);
}

CUSOLVER_ROUTINE_HANDLER(DnDSgesv){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnDSgesv"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int n = in->Get<int>();
    int nrhs = in->Get<int>();
    double *dA = in->GetFromMarshal<double*>();
    int ldda = in->Get<int>();
    int *dipiv = in->Get<int*>();
    double *dB = in->GetFromMarshal<double*>();
    int lddb = in->Get<int>();
    double *dX = in->GetFromMarshal<double*>();
    int lddx = in->Get<int>();
    void *dWorkspace = in->Get<void*>();
    size_t lwork_bytes = in->Get<size_t>();
    int niter;
    int *dinfo = in->Get<int*>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnDSgesv(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, &niter, dinfo);
        out->Add<double*>(dA);
        out->Add<int*>(dipiv);
        out->Add<double*>(dX);
        out->Add<int>(niter);
        out->Add<int*>(dinfo);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnDSgesv Executed");
    return std::make_shared<Result>(cs,out);
}

CUSOLVER_ROUTINE_HANDLER(DnDHgesv){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnDHgesv"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int n = in->Get<int>();
    int nrhs = in->Get<int>();
    double *dA = in->GetFromMarshal<double*>();
    int ldda = in->Get<int>();
    int *dipiv = in->Get<int*>();
    double *dB = in->GetFromMarshal<double*>();
    int lddb = in->Get<int>();
    double *dX = in->GetFromMarshal<double*>();
    int lddx = in->Get<int>();
    void *dWorkspace = in->Get<void*>();
    size_t lwork_bytes = in->Get<size_t>();
    int niter;
    int *dinfo = in->Get<int*>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnDHgesv(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, &niter, dinfo);
        out->Add<double*>(dA);
        out->Add<int*>(dipiv);
        out->Add<double*>(dX);
        out->Add<int>(niter);
        out->Add<int*>(dinfo);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnDHgesv Executed");
    return std::make_shared<Result>(cs,out);
}

CUSOLVER_ROUTINE_HANDLER(DnDBgesv){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnDBgesv"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int n = in->Get<int>();
    int nrhs = in->Get<int>();
    double *dA = in->GetFromMarshal<double*>();
    int ldda = in->Get<int>();
    int *dipiv = in->Get<int*>();
    double *dB = in->GetFromMarshal<double*>();
    int lddb = in->Get<int>();
    double *dX = in->GetFromMarshal<double*>();
    int lddx = in->Get<int>();
    void *dWorkspace = in->Get<void*>();
    size_t lwork_bytes = in->Get<size_t>();
    int niter;
    int *dinfo = in->Get<int*>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnDBgesv(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, &niter, dinfo);
        out->Add<double*>(dA);
        out->Add<int*>(dipiv);
        out->Add<double*>(dX);
        out->Add<int>(niter);
        out->Add<int*>(dinfo);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnDBgesv Executed");
    return std::make_shared<Result>(cs,out);
}

CUSOLVER_ROUTINE_HANDLER(DnDXgesv){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnDXgesv"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int n = in->Get<int>();
    int nrhs = in->Get<int>();
    double *dA = in->GetFromMarshal<double*>();
    int ldda = in->Get<int>();
    int *dipiv = in->Get<int*>();
    double *dB = in->GetFromMarshal<double*>();
    int lddb = in->Get<int>();
    double *dX = in->GetFromMarshal<double*>();
    int lddx = in->Get<int>();
    void *dWorkspace = in->Get<void*>();
    size_t lwork_bytes = in->Get<size_t>();
    int niter;
    int *dinfo = in->Get<int*>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnDXgesv(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, &niter, dinfo);
        out->Add<double*>(dA);
        out->Add<int*>(dipiv);
        out->Add<double*>(dX);
        out->Add<int>(niter);
        out->Add<int*>(dinfo);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnDXgesv Executed");
    return std::make_shared<Result>(cs,out);
}

CUSOLVER_ROUTINE_HANDLER(DnSSgesv){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnSSgesv"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int n = in->Get<int>();
    int nrhs = in->Get<int>();
    float *dA = in->GetFromMarshal<float*>();
    int ldda = in->Get<int>();
    int *dipiv = in->Get<int*>();
    float *dB = in->GetFromMarshal<float*>();
    int lddb = in->Get<int>();
    float *dX = in->GetFromMarshal<float*>();
    int lddx = in->Get<int>();
    void *dWorkspace = in->Get<void*>();
    size_t lwork_bytes = in->Get<size_t>();
    int niter;
    int *dinfo = in->Get<int*>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnSSgesv(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, &niter, dinfo);
        out->Add<float*>(dA);
        out->Add<int*>(dipiv);
        out->Add<float*>(dX);
        out->Add<int>(niter);
        out->Add<int*>(dinfo);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnSSgesv Executed");
    return std::make_shared<Result>(cs,out);
}

CUSOLVER_ROUTINE_HANDLER(DnSHgesv){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnSHgesv"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int n = in->Get<int>();
    int nrhs = in->Get<int>();
    float *dA = in->GetFromMarshal<float*>();
    int ldda = in->Get<int>();
    int *dipiv = in->Get<int*>();
    float *dB = in->GetFromMarshal<float*>();
    int lddb = in->Get<int>();
    float *dX = in->GetFromMarshal<float*>();
    int lddx = in->Get<int>();
    void *dWorkspace = in->Get<void*>();
    size_t lwork_bytes = in->Get<size_t>();
    int niter;
    int *dinfo = in->Get<int*>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnSHgesv(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, &niter, dinfo);
        out->Add<float*>(dA);
        out->Add<int*>(dipiv);
        out->Add<float*>(dX);
        out->Add<int>(niter);
        out->Add<int*>(dinfo);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnSHgesv Executed");
    return std::make_shared<Result>(cs,out);
}

CUSOLVER_ROUTINE_HANDLER(DnSBgesv){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnSBgesv"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int n = in->Get<int>();
    int nrhs = in->Get<int>();
    float *dA = in->GetFromMarshal<float*>();
    int ldda = in->Get<int>();
    int *dipiv = in->Get<int*>();
    float *dB = in->GetFromMarshal<float*>();
    int lddb = in->Get<int>();
    float *dX = in->GetFromMarshal<float*>();
    int lddx = in->Get<int>();
    void *dWorkspace = in->Get<void*>();
    size_t lwork_bytes = in->Get<size_t>();
    int niter;
    int *dinfo = in->Get<int*>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnSBgesv(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, &niter, dinfo);
        out->Add<float*>(dA);
        out->Add<int*>(dipiv);
        out->Add<float*>(dX);
        out->Add<int>(niter);
        out->Add<int*>(dinfo);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnSBgesv Executed");
    return std::make_shared<Result>(cs,out);
}

CUSOLVER_ROUTINE_HANDLER(DnSXgesv){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnSXgesv"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int n = in->Get<int>();
    int nrhs = in->Get<int>();
    float *dA = in->GetFromMarshal<float*>();
    int ldda = in->Get<int>();
    int *dipiv = in->Get<int*>();
    float *dB = in->GetFromMarshal<float*>();
    int lddb = in->Get<int>();
    float *dX = in->GetFromMarshal<float*>();
    int lddx = in->Get<int>();
    void *dWorkspace = in->Get<void*>();
    size_t lwork_bytes = in->Get<size_t>();
    int niter;
    int *dinfo = in->Get<int*>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnSXgesv(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, &niter, dinfo);
        out->Add<float*>(dA);
        out->Add<int*>(dipiv);
        out->Add<float*>(dX);
        out->Add<int>(niter);
        out->Add<int*>(dinfo);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnSXgesv Executed");
    return std::make_shared<Result>(cs,out);
}

CUSOLVER_ROUTINE_HANDLER(DnIRSXgesv_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnIRSXgesv_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    cusolverDnIRSParams_t gesv_irs_params = (cusolverDnIRSParams_t)in->Get<size_t>();
    cusolver_int_t n = in->Get<cusolver_int_t>();
    cusolver_int_t nrhs = in->Get<cusolver_int_t>();
    size_t * lwork_bytes = new size_t;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnIRSXgesv_bufferSize(handle, gesv_irs_params, n, nrhs, lwork_bytes);
        out->Add<size_t>(lwork_bytes);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnIRSXgesv_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnIRSXgesv){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnIRSXgesv"));
    CusolverHandler::setLogLevel(&logger);

    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    cusolverDnIRSParams_t gesv_irs_params = (cusolverDnIRSParams_t)in->Get<size_t>();
    cusolverDnIRSInfos_t gesv_irs_infos = (cusolverDnIRSInfos_t)in->Get<size_t>();
    cusolver_int_t n = in->Get<cusolver_int_t>();
    cusolver_int_t nrhs = in->Get<cusolver_int_t>();
    void *dA = in->GetFromMarshal<void*>();
    cusolver_int_t ldda = in->Get<cusolver_int_t>();
    void *dB = in->GetFromMarshal<void*>();
    cusolver_int_t lddb = in->Get<cusolver_int_t>();
    void *dX = in->GetFromMarshal<void*>();
    cusolver_int_t lddx = in->Get<cusolver_int_t>();
    void *dWorkspace = in->Get<void*>();
    size_t lwork_bytes = in->Get<size_t>();
    cusolver_int_t niters;
    cusolver_int_t *dinfo = in->Get<cusolver_int_t*>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnIRSXgesv(handle, gesv_irs_params, gesv_irs_infos, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, &niters, dinfo);
        out->Add<void*>(dA);
        out->Add<void*>(dX);
        out->Add<cusolver_int_t>(niters);
        out->Add<cusolver_int_t*>(dinfo);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnIRSXgesv Executed");
    return std::make_shared<Result>(cs,out);
}

CUSOLVER_ROUTINE_HANDLER(DnSgeqrf_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnSgeqrf_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    float *A = in->Get<float*>();
    int lda = in->Get<int>();
    int Lwork;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnSgeqrf_bufferSize(handle, m, n, A, lda, &Lwork);
        out->Add<int>(Lwork);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnSgeqrf_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnDgeqrf_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnDgeqrf_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    double *A = in->Get<double*>();
    int lda = in->Get<int>();
    int Lwork;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnDgeqrf_bufferSize(handle, m, n, A, lda, &Lwork);
        out->Add<int>(Lwork);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnDgeqrf_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnCgeqrf_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnCgeqrf_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    cuComplex *A = in->Get<cuComplex*>();
    int lda = in->Get<int>();
    int Lwork;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnCgeqrf_bufferSize(handle, m, n, A, lda, &Lwork);
        out->Add<int>(Lwork);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnCgeqrf_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnZgeqrf_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnZgeqrf_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    cuDoubleComplex *A = in->Get<cuDoubleComplex*>();
    int lda = in->Get<int>();
    int Lwork;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnZgeqrf_bufferSize(handle, m, n, A, lda, &Lwork);
        out->Add<int>(Lwork);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnZgeqrf_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnSgeqrf){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnSgeqrf_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    float *A = in->Get<float*>();
    int lda = in->Get<int>();
    float *TAU = in->Get<float*>();
    float *Workspace = in->Get<float*>();
    int Lwork = in->Get<int>();
    int *devInfo = in->Get<int*>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnSgeqrf(handle, m, n, A, lda, TAU, Workspace, Lwork, devInfo);
        out->Add<float*>(A);
        out->Add<float*>(TAU);
        out->Add<int*>(devInfo);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnSgeqrf Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnDgeqrf){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnDgeqrf_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    double *A = in->Get<double*>();
    int lda = in->Get<int>();
    double *TAU = in->Get<double*>();
    double *Workspace = in->Get<double*>();
    int Lwork = in->Get<int>();
    int *devInfo = in->Get<int*>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnDgeqrf(handle, m, n, A, lda, TAU, Workspace, Lwork, devInfo);
        out->Add<double*>(A);
        out->Add<double*>(TAU);
        out->Add<int*>(devInfo);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnDgeqrf Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnCgeqrf){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnCgeqrf_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    cuComplex *A = in->Get<cuComplex*>();
    int lda = in->Get<int>();
    cuComplex *TAU = in->Get<cuComplex*>();
    cuComplex *Workspace = in->Get<cuComplex*>();
    int Lwork = in->Get<int>();
    int *devInfo = in->Get<int*>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnCgeqrf(handle, m, n, A, lda, TAU, Workspace, Lwork, devInfo);
        out->Add<cuComplex*>(A);
        out->Add<cuComplex*>(TAU);
        out->Add<int*>(devInfo);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnCgeqrf Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnZgeqrf){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnZgeqrf_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    cuDoubleComplex *A = in->Get<cuDoubleComplex*>();
    int lda = in->Get<int>();
    cuDoubleComplex *TAU = in->Get<cuDoubleComplex*>();
    cuDoubleComplex *Workspace = in->Get<cuDoubleComplex*>();
    int Lwork = in->Get<int>();
    int *devInfo = in->Get<int*>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnZgeqrf(handle, m, n, A, lda, TAU, Workspace, Lwork, devInfo);
        out->Add<cuDoubleComplex*>(A);
        out->Add<cuDoubleComplex*>(TAU);
        out->Add<int*>(devInfo);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnZgeqrf Executed");
    return std::make_shared<Result>(cs, out);
}
