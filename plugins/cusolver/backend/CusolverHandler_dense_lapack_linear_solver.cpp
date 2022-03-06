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
    size_t workspaceInBytes;
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
        cs = cusolverDnPotrf_bufferSize(handle, params, uplo, n, dataTypeA, A, lda, computeType, &workspaceInBytes);
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
    size_t workspaceInBytes;
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
        cs = cusolverDnGetrf_bufferSize(handle, params, m, n, dataTypeA, A, lda, computeType, &workspaceInBytes);
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
    size_t workspaceInBytes;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnZZgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWork, &workspaceInBytes);
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
    size_t workspaceInBytes;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnZCgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWork, &workspaceInBytes);
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
    size_t workspaceInBytes;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnZKgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWork, &workspaceInBytes);
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
    size_t workspaceInBytes;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnZEgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWork, &workspaceInBytes);
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
    size_t workspaceInBytes;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnZYgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWork, &workspaceInBytes);
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
    size_t workspaceInBytes;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnCCgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWork, &workspaceInBytes);
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
    size_t workspaceInBytes;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnCKgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWork, &workspaceInBytes);
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
    size_t workspaceInBytes;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnCEgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWork, &workspaceInBytes);
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
    size_t workspaceInBytes;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnCYgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWork, &workspaceInBytes);
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
    size_t workspaceInBytes;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnDDgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWork, &workspaceInBytes);
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
    size_t workspaceInBytes;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnDSgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWork, &workspaceInBytes);
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
    size_t workspaceInBytes;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnDHgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWork, &workspaceInBytes);
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
    size_t workspaceInBytes;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnDBgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWork, &workspaceInBytes);
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
    size_t workspaceInBytes;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnDXgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWork, &workspaceInBytes);
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
    size_t workspaceInBytes;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnSSgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWork, &workspaceInBytes);
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
    size_t workspaceInBytes;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnSHgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWork, &workspaceInBytes);
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
    size_t workspaceInBytes;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnSBgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWork, &workspaceInBytes);
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
    size_t workspaceInBytes;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnSXgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWork, &workspaceInBytes);
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
    size_t lwork_bytes;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnIRSXgesv_bufferSize(handle, gesv_irs_params, n, nrhs, &lwork_bytes);
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
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnSgeqrf"));
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
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnDgeqrf"));
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
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnCgeqrf"));
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
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnZgeqrf"));
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

CUSOLVER_ROUTINE_HANDLER(DnGeqrf_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnGeqrf_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    cusolverDnParams_t params = (cusolverDnParams_t)in->Get<size_t>();
    int64_t m = in->Get<int64_t>();
    int64_t n = in->Get<int64_t>();
    cudaDataType dataTypeA = in->Get<cudaDataType_t>();
    int64_t lda = in->Get<int64_t>();
    cudaDataType dataTypeTau = in->Get<cudaDataType_t>();
    cudaDataType computeType = in->Get<cudaDataType_t>();
    size_t workspaceInBytes;
    void* A;
    void* TAU;
    if (dataTypeA == CUDA_R_32F) {
        // float
        A = in->GetFromMarshal<float*>();
        TAU = in->GetFromMarshal<float*>();
    } else if (dataTypeA == CUDA_R_64F) {
        // double
        A = in->GetFromMarshal<double*>();
        TAU = in->GetFromMarshal<double*>();
    } else if (dataTypeA == CUDA_C_32F) {
        // cuComplex
        A = in->GetFromMarshal<cuComplex*>();
        TAU = in->GetFromMarshal<cuComplex*>();
    } else if (dataTypeA == CUDA_C_64F) {
        // cuDoubleComplex
        A = in->GetFromMarshal<cuDoubleComplex*>();
        TAU = in->GetFromMarshal<cuDoubleComplex*>();
    } else {
        throw "Type not supported by GVirtus!";
    }
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnGeqrf_bufferSize(handle, params, m, n, dataTypeA, A, lda, dataTypeTau, TAU, computeType, &workspaceInBytes);
        out->Add<size_t>(workspaceInBytes);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnGeqrf_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnGeqrf){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnGeqrf"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    cusolverDnParams_t params = (cusolverDnParams_t)in->Get<size_t>();
    int64_t m = in->Get<int64_t>();
    int64_t n = in->Get<int64_t>();
    cudaDataType dataTypeA = in->Get<cudaDataType_t>();
    int64_t lda = in->Get<int64_t>();
    cudaDataType dataTypeTau = in->Get<cudaDataType_t>();
    cudaDataType computeType = in->Get<cudaDataType_t>();
    void *pBuffer = in->Get<void*>();
    size_t workspaceInBytes = in->Get<size_t >();
    int *info = in->GetFromMarshal<int*>();
    void* A;
    void* TAU;
    if (dataTypeA == CUDA_R_32F) {
        // float
        A = in->GetFromMarshal<float*>();
        TAU = in->GetFromMarshal<float*>();
    } else if (dataTypeA == CUDA_R_64F) {
        // double
        A = in->GetFromMarshal<double*>();
        TAU = in->GetFromMarshal<double*>();
    } else if (dataTypeA == CUDA_C_32F) {
        // cuComplex
        A = in->GetFromMarshal<cuComplex*>();
        TAU = in->GetFromMarshal<cuComplex*>();
    } else if (dataTypeA == CUDA_C_64F) {
        // cuDoubleComplex
        A = in->GetFromMarshal<cuDoubleComplex*>();
        TAU = in->GetFromMarshal<cuDoubleComplex*>();
    } else {
        throw "Type not supported by GVirtus!";
    }
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnGeqrf(handle, params, m, n, dataTypeA, A, lda, dataTypeTau, TAU, computeType, pBuffer, workspaceInBytes, info);
        out->Add<int*>(info);
        out->Add<void*>(TAU);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnGeqrf Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnZZgels_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnZZgels_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    int nrhs = in->Get<int>();
    cuDoubleComplex *dA = in->Get<cuDoubleComplex*>();
    int ldda = in->Get<int>();
    cuDoubleComplex *dB = in->Get<cuDoubleComplex*>();
    int lddb = in->Get<int>();
    cuDoubleComplex *dX = in->Get<cuDoubleComplex*>();
    int lddx = in->Get<int>();
    void *dwork = in->Get<void*>();
    size_t lwork_bytes;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnZZgels_bufferSize(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dwork, &lwork_bytes);
        out->Add<size_t>(lwork_bytes);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnZZgels_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnZCgels_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnZCgels_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    int nrhs = in->Get<int>();
    cuDoubleComplex *dA = in->Get<cuDoubleComplex*>();
    int ldda = in->Get<int>();
    cuDoubleComplex *dB = in->Get<cuDoubleComplex*>();
    int lddb = in->Get<int>();
    cuDoubleComplex *dX = in->Get<cuDoubleComplex*>();
    int lddx = in->Get<int>();
    void *dwork = in->Get<void*>();
    size_t lwork_bytes;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnZCgels_bufferSize(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dwork, &lwork_bytes);
        out->Add<size_t>(lwork_bytes);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnZCgels_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnZKgels_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnZKgels_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    int nrhs = in->Get<int>();
    cuDoubleComplex *dA = in->Get<cuDoubleComplex*>();
    int ldda = in->Get<int>();
    cuDoubleComplex *dB = in->Get<cuDoubleComplex*>();
    int lddb = in->Get<int>();
    cuDoubleComplex *dX = in->Get<cuDoubleComplex*>();
    int lddx = in->Get<int>();
    void *dwork = in->Get<void*>();
    size_t lwork_bytes;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnZKgels_bufferSize(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dwork, &lwork_bytes);
        out->Add<size_t>(lwork_bytes);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnZKgels_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnZEgels_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnZEgels_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    int nrhs = in->Get<int>();
    cuDoubleComplex *dA = in->Get<cuDoubleComplex*>();
    int ldda = in->Get<int>();
    cuDoubleComplex *dB = in->Get<cuDoubleComplex*>();
    int lddb = in->Get<int>();
    cuDoubleComplex *dX = in->Get<cuDoubleComplex*>();
    int lddx = in->Get<int>();
    void *dwork = in->Get<void*>();
    size_t lwork_bytes;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnZEgels_bufferSize(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dwork, &lwork_bytes);
        out->Add<size_t>(lwork_bytes);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnZEgels_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnZYgels_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnZYgels_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    int nrhs = in->Get<int>();
    cuDoubleComplex *dA = in->Get<cuDoubleComplex*>();
    int ldda = in->Get<int>();
    cuDoubleComplex *dB = in->Get<cuDoubleComplex*>();
    int lddb = in->Get<int>();
    cuDoubleComplex *dX = in->Get<cuDoubleComplex*>();
    int lddx = in->Get<int>();
    void *dwork = in->Get<void*>();
    size_t lwork_bytes;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnZYgels_bufferSize(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dwork, &lwork_bytes);
        out->Add<size_t>(lwork_bytes);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnZYgels_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnCCgels_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnCCgels_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    int nrhs = in->Get<int>();
    cuComplex *dA = in->Get<cuComplex*>();
    int ldda = in->Get<int>();
    cuComplex *dB = in->Get<cuComplex*>();
    int lddb = in->Get<int>();
    cuComplex *dX = in->Get<cuComplex*>();
    int lddx = in->Get<int>();
    void *dwork = in->Get<void*>();
    size_t lwork_bytes;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnCCgels_bufferSize(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dwork, &lwork_bytes);
        out->Add<size_t>(lwork_bytes);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnCCgels_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnCKgels_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnCKgels_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    int nrhs = in->Get<int>();
    cuComplex *dA = in->Get<cuComplex*>();
    int ldda = in->Get<int>();
    cuComplex *dB = in->Get<cuComplex*>();
    int lddb = in->Get<int>();
    cuComplex *dX = in->Get<cuComplex*>();
    int lddx = in->Get<int>();
    void *dwork = in->Get<void*>();
    size_t lwork_bytes;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnCKgels_bufferSize(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dwork, &lwork_bytes);
        out->Add<size_t>(lwork_bytes);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnCKgels_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnCEgels_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnCEgels_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    int nrhs = in->Get<int>();
    cuComplex *dA = in->Get<cuComplex*>();
    int ldda = in->Get<int>();
    cuComplex *dB = in->Get<cuComplex*>();
    int lddb = in->Get<int>();
    cuComplex *dX = in->Get<cuComplex*>();
    int lddx = in->Get<int>();
    void *dwork = in->Get<void*>();
    size_t lwork_bytes;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnCEgels_bufferSize(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dwork, &lwork_bytes);
        out->Add<size_t>(lwork_bytes);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnCEgels_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnCYgels_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnCYgels_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    int nrhs = in->Get<int>();
    cuComplex *dA = in->Get<cuComplex*>();
    int ldda = in->Get<int>();
    cuComplex *dB = in->Get<cuComplex*>();
    int lddb = in->Get<int>();
    cuComplex *dX = in->Get<cuComplex*>();
    int lddx = in->Get<int>();
    void *dwork = in->Get<void*>();
    size_t lwork_bytes;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnCYgels_bufferSize(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dwork, &lwork_bytes);
        out->Add<size_t>(lwork_bytes);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnCYgels_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnDDgels_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnDDgels_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    int nrhs = in->Get<int>();
    double *dA = in->Get<double*>();
    int ldda = in->Get<int>();
    double *dB = in->Get<double*>();
    int lddb = in->Get<int>();
    double *dX = in->Get<double*>();
    int lddx = in->Get<int>();
    void *dwork = in->Get<void*>();
    size_t lwork_bytes;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnDDgels_bufferSize(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dwork, &lwork_bytes);
        out->Add<size_t>(lwork_bytes);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnDDgels_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnDSgels_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnDSgels_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    int nrhs = in->Get<int>();
    double *dA = in->Get<double*>();
    int ldda = in->Get<int>();
    double *dB = in->Get<double*>();
    int lddb = in->Get<int>();
    double *dX = in->Get<double*>();
    int lddx = in->Get<int>();
    void *dwork = in->Get<void*>();
    size_t lwork_bytes;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnDSgels_bufferSize(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dwork, &lwork_bytes);
        out->Add<size_t>(lwork_bytes);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnDSgels_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnDHgels_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnDHgels_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    int nrhs = in->Get<int>();
    double *dA = in->Get<double*>();
    int ldda = in->Get<int>();
    double *dB = in->Get<double*>();
    int lddb = in->Get<int>();
    double *dX = in->Get<double*>();
    int lddx = in->Get<int>();
    void *dwork = in->Get<void*>();
    size_t lwork_bytes;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnDHgels_bufferSize(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dwork, &lwork_bytes);
        out->Add<size_t>(lwork_bytes);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnDHgels_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnDBgels_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnDBgels_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    int nrhs = in->Get<int>();
    double *dA = in->Get<double*>();
    int ldda = in->Get<int>();
    double *dB = in->Get<double*>();
    int lddb = in->Get<int>();
    double *dX = in->Get<double*>();
    int lddx = in->Get<int>();
    void *dwork = in->Get<void*>();
    size_t lwork_bytes;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnDBgels_bufferSize(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dwork, &lwork_bytes);
        out->Add<size_t>(lwork_bytes);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnDBgels_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnDXgels_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnDXgels_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    int nrhs = in->Get<int>();
    double *dA = in->Get<double*>();
    int ldda = in->Get<int>();
    double *dB = in->Get<double*>();
    int lddb = in->Get<int>();
    double *dX = in->Get<double*>();
    int lddx = in->Get<int>();
    void *dwork = in->Get<void*>();
    size_t lwork_bytes;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnDXgels_bufferSize(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dwork, &lwork_bytes);
        out->Add<size_t>(lwork_bytes);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnDXgels_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnSSgels_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnSSgels_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    int nrhs = in->Get<int>();
    float *dA = in->Get<float*>();
    int ldda = in->Get<int>();
    float *dB = in->Get<float*>();
    int lddb = in->Get<int>();
    float *dX = in->Get<float*>();
    int lddx = in->Get<int>();
    void *dwork = in->Get<void*>();
    size_t lwork_bytes;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnSSgels_bufferSize(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dwork, &lwork_bytes);
        out->Add<size_t>(lwork_bytes);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnSSgels_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnSHgels_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnSHgels_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    int nrhs = in->Get<int>();
    float *dA = in->Get<float*>();
    int ldda = in->Get<int>();
    float *dB = in->Get<float*>();
    int lddb = in->Get<int>();
    float *dX = in->Get<float*>();
    int lddx = in->Get<int>();
    void *dwork = in->Get<void*>();
    size_t lwork_bytes;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnSHgels_bufferSize(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dwork, &lwork_bytes);
        out->Add<size_t>(lwork_bytes);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnSHgels_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnSBgels_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnSBgels_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    int nrhs = in->Get<int>();
    float *dA = in->Get<float*>();
    int ldda = in->Get<int>();
    float *dB = in->Get<float*>();
    int lddb = in->Get<int>();
    float *dX = in->Get<float*>();
    int lddx = in->Get<int>();
    void *dwork = in->Get<void*>();
    size_t lwork_bytes;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnSBgels_bufferSize(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dwork, &lwork_bytes);
        out->Add<size_t>(lwork_bytes);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnSBgels_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnSXgels_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnSXgels_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    int nrhs = in->Get<int>();
    float *dA = in->Get<float*>();
    int ldda = in->Get<int>();
    float *dB = in->Get<float*>();
    int lddb = in->Get<int>();
    float *dX = in->Get<float*>();
    int lddx = in->Get<int>();
    void *dwork = in->Get<void*>();
    size_t lwork_bytes;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnSXgels_bufferSize(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dwork, &lwork_bytes);
        out->Add<size_t>(lwork_bytes);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnSXgels_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnZZgels){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnZZgels"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    int nrhs = in->Get<int>();
    cuDoubleComplex *dA = in->Get<cuDoubleComplex*>();
    int ldda = in->Get<int>();
    cuDoubleComplex *dB = in->Get<cuDoubleComplex*>();
    int lddb = in->Get<int>();
    cuDoubleComplex *dX = in->Get<cuDoubleComplex*>();
    int lddx = in->Get<int>();
    void *dWorkspace = in->Get<void*>();
    int lwork_bytes = in->Get<int>();
    int niters;
    int *dinfo = in->Get<int*>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnZZgels(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, &niters, dinfo);
        out->Add<cuDoubleComplex*>(dX);
        out->Add<int>(niters);
        out->Add<int*>(dinfo);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnZZgels Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnZCgels){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnZCgels"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    int nrhs = in->Get<int>();
    cuDoubleComplex *dA = in->Get<cuDoubleComplex*>();
    int ldda = in->Get<int>();
    cuDoubleComplex *dB = in->Get<cuDoubleComplex*>();
    int lddb = in->Get<int>();
    cuDoubleComplex *dX = in->Get<cuDoubleComplex*>();
    int lddx = in->Get<int>();
    void *dWorkspace = in->Get<void*>();
    int lwork_bytes = in->Get<int>();
    int niters;
    int *dinfo = in->Get<int*>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnZCgels(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, &niters, dinfo);
        out->Add<cuDoubleComplex*>(dX);
        out->Add<int>(niters);
        out->Add<int*>(dinfo);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnZCgels Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnZKgels){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnZKgels"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    int nrhs = in->Get<int>();
    cuDoubleComplex *dA = in->Get<cuDoubleComplex*>();
    int ldda = in->Get<int>();
    cuDoubleComplex *dB = in->Get<cuDoubleComplex*>();
    int lddb = in->Get<int>();
    cuDoubleComplex *dX = in->Get<cuDoubleComplex*>();
    int lddx = in->Get<int>();
    void *dWorkspace = in->Get<void*>();
    int lwork_bytes = in->Get<int>();
    int niters;
    int *dinfo = in->Get<int*>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnZKgels(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, &niters, dinfo);
        out->Add<cuDoubleComplex*>(dX);
        out->Add<int>(niters);
        out->Add<int*>(dinfo);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnZKgels Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnZEgels){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnZEgels"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    int nrhs = in->Get<int>();
    cuDoubleComplex *dA = in->Get<cuDoubleComplex*>();
    int ldda = in->Get<int>();
    cuDoubleComplex *dB = in->Get<cuDoubleComplex*>();
    int lddb = in->Get<int>();
    cuDoubleComplex *dX = in->Get<cuDoubleComplex*>();
    int lddx = in->Get<int>();
    void *dWorkspace = in->Get<void*>();
    int lwork_bytes = in->Get<int>();
    int niters;
    int *dinfo = in->Get<int*>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnZEgels(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, &niters, dinfo);
        out->Add<cuDoubleComplex*>(dX);
        out->Add<int>(niters);
        out->Add<int*>(dinfo);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnZEgels Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnZYgels){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnZYgels"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    int nrhs = in->Get<int>();
    cuDoubleComplex *dA = in->Get<cuDoubleComplex*>();
    int ldda = in->Get<int>();
    cuDoubleComplex *dB = in->Get<cuDoubleComplex*>();
    int lddb = in->Get<int>();
    cuDoubleComplex *dX = in->Get<cuDoubleComplex*>();
    int lddx = in->Get<int>();
    void *dWorkspace = in->Get<void*>();
    int lwork_bytes = in->Get<int>();
    int niters;
    int *dinfo = in->Get<int*>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnZYgels(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, &niters, dinfo);
        out->Add<cuDoubleComplex*>(dX);
        out->Add<int>(niters);
        out->Add<int*>(dinfo);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnZYgels Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnCCgels){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnCCgels"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    int nrhs = in->Get<int>();
    cuComplex *dA = in->Get<cuComplex*>();
    int ldda = in->Get<int>();
    cuComplex *dB = in->Get<cuComplex*>();
    int lddb = in->Get<int>();
    cuComplex *dX = in->Get<cuComplex*>();
    int lddx = in->Get<int>();
    void *dWorkspace = in->Get<void*>();
    int lwork_bytes = in->Get<int>();
    int niters;
    int *dinfo = in->Get<int*>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnCCgels(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, &niters, dinfo);
        out->Add<cuComplex*>(dX);
        out->Add<int>(niters);
        out->Add<int*>(dinfo);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnCCgels Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnCKgels){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnCKgels"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    int nrhs = in->Get<int>();
    cuComplex *dA = in->Get<cuComplex*>();
    int ldda = in->Get<int>();
    cuComplex *dB = in->Get<cuComplex*>();
    int lddb = in->Get<int>();
    cuComplex *dX = in->Get<cuComplex*>();
    int lddx = in->Get<int>();
    void *dWorkspace = in->Get<void*>();
    int lwork_bytes = in->Get<int>();
    int niters;
    int *dinfo = in->Get<int*>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnCKgels(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, &niters, dinfo);
        out->Add<cuComplex*>(dX);
        out->Add<int>(niters);
        out->Add<int*>(dinfo);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnCKgels Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnCEgels){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnCEgels"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    int nrhs = in->Get<int>();
    cuComplex *dA = in->Get<cuComplex*>();
    int ldda = in->Get<int>();
    cuComplex *dB = in->Get<cuComplex*>();
    int lddb = in->Get<int>();
    cuComplex *dX = in->Get<cuComplex*>();
    int lddx = in->Get<int>();
    void *dWorkspace = in->Get<void*>();
    int lwork_bytes = in->Get<int>();
    int niters;
    int *dinfo = in->Get<int*>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnCEgels(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, &niters, dinfo);
        out->Add<cuComplex*>(dX);
        out->Add<int>(niters);
        out->Add<int*>(dinfo);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnCEgels Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnCYgels){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnCYgels"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    int nrhs = in->Get<int>();
    cuComplex *dA = in->Get<cuComplex*>();
    int ldda = in->Get<int>();
    cuComplex *dB = in->Get<cuComplex*>();
    int lddb = in->Get<int>();
    cuComplex *dX = in->Get<cuComplex*>();
    int lddx = in->Get<int>();
    void *dWorkspace = in->Get<void*>();
    int lwork_bytes = in->Get<int>();
    int niters;
    int *dinfo = in->Get<int*>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnCYgels(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, &niters, dinfo);
        out->Add<cuComplex*>(dX);
        out->Add<int>(niters);
        out->Add<int*>(dinfo);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnCYgels Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnDDgels){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnDDgels"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    int nrhs = in->Get<int>();
    double *dA = in->Get<double*>();
    int ldda = in->Get<int>();
    double *dB = in->Get<double*>();
    int lddb = in->Get<int>();
    double *dX = in->Get<double*>();
    int lddx = in->Get<int>();
    void *dWorkspace = in->Get<void*>();
    int lwork_bytes = in->Get<int>();
    int niters;
    int *dinfo = in->Get<int*>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnDDgels(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, &niters, dinfo);
        out->Add<double*>(dX);
        out->Add<int>(niters);
        out->Add<int*>(dinfo);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnDDgels Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnDSgels){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnDSgels"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    int nrhs = in->Get<int>();
    double *dA = in->Get<double*>();
    int ldda = in->Get<int>();
    double *dB = in->Get<double*>();
    int lddb = in->Get<int>();
    double *dX = in->Get<double*>();
    int lddx = in->Get<int>();
    void *dWorkspace = in->Get<void*>();
    int lwork_bytes = in->Get<int>();
    int niters;
    int *dinfo = in->Get<int*>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnDSgels(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, &niters, dinfo);
        out->Add<double*>(dX);
        out->Add<int>(niters);
        out->Add<int*>(dinfo);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnDSgels Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnDHgels){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnDHgels"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    int nrhs = in->Get<int>();
    double *dA = in->Get<double*>();
    int ldda = in->Get<int>();
    double *dB = in->Get<double*>();
    int lddb = in->Get<int>();
    double *dX = in->Get<double*>();
    int lddx = in->Get<int>();
    void *dWorkspace = in->Get<void*>();
    int lwork_bytes = in->Get<int>();
    int niters;
    int *dinfo = in->Get<int*>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnDHgels(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, &niters, dinfo);
        out->Add<double*>(dX);
        out->Add<int>(niters);
        out->Add<int*>(dinfo);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnDHgels Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnDBgels){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnDBgels"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    int nrhs = in->Get<int>();
    double *dA = in->Get<double*>();
    int ldda = in->Get<int>();
    double *dB = in->Get<double*>();
    int lddb = in->Get<int>();
    double *dX = in->Get<double*>();
    int lddx = in->Get<int>();
    void *dWorkspace = in->Get<void*>();
    int lwork_bytes = in->Get<int>();
    int niters;
    int *dinfo = in->Get<int*>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnDBgels(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, &niters, dinfo);
        out->Add<double*>(dX);
        out->Add<int>(niters);
        out->Add<int*>(dinfo);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnDBgels Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnDXgels){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnDXgels"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    int nrhs = in->Get<int>();
    double *dA = in->Get<double*>();
    int ldda = in->Get<int>();
    double *dB = in->Get<double*>();
    int lddb = in->Get<int>();
    double *dX = in->Get<double*>();
    int lddx = in->Get<int>();
    void *dWorkspace = in->Get<void*>();
    int lwork_bytes = in->Get<int>();
    int niters;
    int *dinfo = in->Get<int*>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnDXgels(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, &niters, dinfo);
        out->Add<double*>(dX);
        out->Add<int>(niters);
        out->Add<int*>(dinfo);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnDXgels Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnSSgels){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnSSgels"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    int nrhs = in->Get<int>();
    float *dA = in->Get<float*>();
    int ldda = in->Get<int>();
    float *dB = in->Get<float*>();
    int lddb = in->Get<int>();
    float *dX = in->Get<float*>();
    int lddx = in->Get<int>();
    void *dWorkspace = in->Get<void*>();
    int lwork_bytes = in->Get<int>();
    int niters;
    int *dinfo = in->Get<int*>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnSSgels(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, &niters, dinfo);
        out->Add<float*>(dX);
        out->Add<int>(niters);
        out->Add<int*>(dinfo);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnSSgels Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnSHgels){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnSHgels"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    int nrhs = in->Get<int>();
    float *dA = in->Get<float*>();
    int ldda = in->Get<int>();
    float *dB = in->Get<float*>();
    int lddb = in->Get<int>();
    float *dX = in->Get<float*>();
    int lddx = in->Get<int>();
    void *dWorkspace = in->Get<void*>();
    int lwork_bytes = in->Get<int>();
    int niters;
    int *dinfo = in->Get<int*>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnSHgels(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, &niters, dinfo);
        out->Add<float*>(dX);
        out->Add<int>(niters);
        out->Add<int*>(dinfo);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnSHgels Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnSBgels){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnSBgels"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    int nrhs = in->Get<int>();
    float *dA = in->Get<float*>();
    int ldda = in->Get<int>();
    float *dB = in->Get<float*>();
    int lddb = in->Get<int>();
    float *dX = in->Get<float*>();
    int lddx = in->Get<int>();
    void *dWorkspace = in->Get<void*>();
    int lwork_bytes = in->Get<int>();
    int niters;
    int *dinfo = in->Get<int*>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnSBgels(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, &niters, dinfo);
        out->Add<float*>(dX);
        out->Add<int>(niters);
        out->Add<int*>(dinfo);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnSBgels Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnSXgels){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnSXgels"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    int nrhs = in->Get<int>();
    float *dA = in->Get<float*>();
    int ldda = in->Get<int>();
    float *dB = in->Get<float*>();
    int lddb = in->Get<int>();
    float *dX = in->Get<float*>();
    int lddx = in->Get<int>();
    void *dWorkspace = in->Get<void*>();
    int lwork_bytes = in->Get<int>();
    int niters;
    int *dinfo = in->Get<int*>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnSXgels(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, &niters, dinfo);
        out->Add<float*>(dX);
        out->Add<int>(niters);
        out->Add<int*>(dinfo);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnSXgels Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnIRSXgels_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnIRSXgels_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    cusolverDnIRSParams_t gesl_irs_params = (cusolverDnIRSParams_t)in->Get<size_t>();
    cusolver_int_t m = in->Get<cusolver_int_t>();
    cusolver_int_t n = in->Get<cusolver_int_t>();
    cusolver_int_t nrhs = in->Get<cusolver_int_t>();
    size_t lwork_bytes;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnIRSXgels_bufferSize(handle, gesl_irs_params, m, n, nrhs, &lwork_bytes);
        out->Add<size_t>(lwork_bytes);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnIRSXgels_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnIRSXgels){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnIRSXgels"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    cusolverDnIRSParams_t gesl_irs_params = (cusolverDnIRSParams_t)in->Get<size_t>();
    cusolverDnIRSInfos_t gesl_irs_infos = (cusolverDnIRSInfos_t)in->Get<size_t>();
    cusolver_int_t m = in->Get<cusolver_int_t>();
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
        cs = cusolverDnIRSXgels(handle, gesl_irs_params, gesl_irs_infos, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, &niters, dinfo);
        out->Add<void*>(dA);
        out->Add<void*>(dX);
        out->Add<cusolver_int_t>(niters);
        out->Add<cusolver_int_t*>(dinfo);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnIRSXgels Executed");
    return std::make_shared<Result>(cs,out);
}

CUSOLVER_ROUTINE_HANDLER(DnSormqr_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnSormqr_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    cublasSideMode_t side = in->Get<cublasSideMode_t>();
    cublasOperation_t trans = in->Get<cublasOperation_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    int k = in->Get<int>();
    float *A = in->Get<float*>();
    int lda = in->Get<int>();
    float *tau = in->Get<float*>();
    float *C = in->Get<float*>();
    int ldc = in->Get<int>();
    int lwork;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnSormqr_bufferSize(handle, side, trans, m, n, k, A, lda, tau, C, ldc, &lwork);
        out->Add<int>(lwork);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnSormqr_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnDormqr_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnDormqr_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    cublasSideMode_t side = in->Get<cublasSideMode_t>();
    cublasOperation_t trans = in->Get<cublasOperation_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    int k = in->Get<int>();
    double *A = in->Get<double*>();
    int lda = in->Get<int>();
    double *tau = in->Get<double*>();
    double *C = in->Get<double*>();
    int ldc = in->Get<int>();
    int lwork;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnDormqr_bufferSize(handle, side, trans, m, n, k, A, lda, tau, C, ldc, &lwork);
        out->Add<int>(lwork);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnDormqr_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnCunmqr_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnCunmqr_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    cublasSideMode_t side = in->Get<cublasSideMode_t>();
    cublasOperation_t trans = in->Get<cublasOperation_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    int k = in->Get<int>();
    cuComplex *A = in->Get<cuComplex*>();
    int lda = in->Get<int>();
    cuComplex *tau = in->Get<cuComplex*>();
    cuComplex *C = in->Get<cuComplex*>();
    int ldc = in->Get<int>();
    int lwork;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnCunmqr_bufferSize(handle, side, trans, m, n, k, A, lda, tau, C, ldc, &lwork);
        out->Add<int>(lwork);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnCunmqr_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnZunmqr_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnZunmqr_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    cublasSideMode_t side = in->Get<cublasSideMode_t>();
    cublasOperation_t trans = in->Get<cublasOperation_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    int k = in->Get<int>();
    cuDoubleComplex *A = in->Get<cuDoubleComplex*>();
    int lda = in->Get<int>();
    cuDoubleComplex *tau = in->Get<cuDoubleComplex*>();
    cuDoubleComplex *C = in->Get<cuDoubleComplex*>();
    int ldc = in->Get<int>();
    int lwork;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnZunmqr_bufferSize(handle, side, trans, m, n, k, A, lda, tau, C, ldc, &lwork);
        out->Add<int>(lwork);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnZunmqr_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnSormqr){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnSormqr"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    cublasSideMode_t side = in->Get<cublasSideMode_t>();
    cublasOperation_t trans = in->Get<cublasOperation_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    int k = in->Get<int>();
    float *A = in->Get<float*>();
    int lda = in->Get<int>();
    float *tau = in->Get<float*>();
    float *C = in->Get<float*>();
    int ldc = in->Get<int>();
    float *work = in->Get<float*>();
    int lwork = in->Get<int>();
    int *devInfo = in->Get<int*>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnSormqr(handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, devInfo);
        out->Add<float*>(tau);
        out->Add<int*>(devInfo);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnSormqr Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnDormqr){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnDormqr"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    cublasSideMode_t side = in->Get<cublasSideMode_t>();
    cublasOperation_t trans = in->Get<cublasOperation_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    int k = in->Get<int>();
    double *A = in->Get<double*>();
    int lda = in->Get<int>();
    double *tau = in->Get<double*>();
    double *C = in->Get<double*>();
    int ldc = in->Get<int>();
    double *work = in->Get<double*>();
    int lwork = in->Get<int>();
    int *devInfo = in->Get<int*>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnDormqr(handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, devInfo);
        out->Add<double*>(tau);
        out->Add<int*>(devInfo);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnDormqr Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnCunmqr){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnCunmqr"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    cublasSideMode_t side = in->Get<cublasSideMode_t>();
    cublasOperation_t trans = in->Get<cublasOperation_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    int k = in->Get<int>();
    cuComplex *A = in->Get<cuComplex*>();
    int lda = in->Get<int>();
    cuComplex *tau = in->Get<cuComplex*>();
    cuComplex *C = in->Get<cuComplex*>();
    int ldc = in->Get<int>();
    cuComplex *work = in->Get<cuComplex*>();
    int lwork = in->Get<int>();
    int *devInfo = in->Get<int*>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnCunmqr(handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, devInfo);
        out->Add<cuComplex*>(tau);
        out->Add<int*>(devInfo);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnCunmqr Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnZunmqr){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnZunmqr"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    cublasSideMode_t side = in->Get<cublasSideMode_t>();
    cublasOperation_t trans = in->Get<cublasOperation_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    int k = in->Get<int>();
    cuDoubleComplex *A = in->Get<cuDoubleComplex*>();
    int lda = in->Get<int>();
    cuDoubleComplex *tau = in->Get<cuDoubleComplex*>();
    cuDoubleComplex *C = in->Get<cuDoubleComplex*>();
    int ldc = in->Get<int>();
    cuDoubleComplex *work = in->Get<cuDoubleComplex*>();
    int lwork = in->Get<int>();
    int *devInfo = in->Get<int*>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnZunmqr(handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, devInfo);
        out->Add<cuDoubleComplex*>(tau);
        out->Add<int*>(devInfo);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnZunmqr Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnSorgqr_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnSorgqr_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    int k = in->Get<int>();
    float *A = in->Get<float*>();
    int lda = in->Get<int>();
    float *tau = in->Get<float*>();
    int lwork;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnSorgqr_bufferSize(handle, m, n, k, A, lda, tau, &lwork);
        out->Add<int>(lwork);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnSorgqr_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnDorgqr_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnDorgqr_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    int k = in->Get<int>();
    double *A = in->Get<double*>();
    int lda = in->Get<int>();
    double *tau = in->Get<double*>();
    int lwork;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnDorgqr_bufferSize(handle, m, n, k, A, lda, tau, &lwork);
        out->Add<int>(lwork);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnDorgqr_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnCungqr_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnCungqr_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    int k = in->Get<int>();
    cuComplex *A = in->Get<cuComplex*>();
    int lda = in->Get<int>();
    cuComplex *tau = in->Get<cuComplex*>();
    int lwork;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnCungqr_bufferSize(handle, m, n, k, A, lda, tau, &lwork);
        out->Add<int>(lwork);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnCungqr_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnZungqr_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnZungqr_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    int k = in->Get<int>();
    cuDoubleComplex *A = in->Get<cuDoubleComplex*>();
    int lda = in->Get<int>();
    cuDoubleComplex *tau = in->Get<cuDoubleComplex*>();
    int lwork;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnZungqr_bufferSize(handle, m, n, k, A, lda, tau, &lwork);
        out->Add<int>(lwork);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnZungqr_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnSorgqr){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnSorgqr"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    int k = in->Get<int>();
    float *A = in->Get<float*>();
    int lda = in->Get<int>();
    float *tau = in->Get<float*>();
    float *work = in->Get<float*>();
    int lwork = in->Get<int>();
    int *devInfo = in->Get<int*>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnSorgqr(handle, m, n, k, A, lda, tau, work, lwork, devInfo);
        out->Add<float*>(tau);
        out->Add<int*>(devInfo);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnSorgqr Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnDorgqr){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnDorgqr"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    int k = in->Get<int>();
    double *A = in->Get<double*>();
    int lda = in->Get<int>();
    double *tau = in->Get<double*>();
    double *work = in->Get<double*>();
    int lwork = in->Get<int>();
    int *devInfo = in->Get<int*>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnDorgqr(handle, m, n, k, A, lda, tau, work, lwork, devInfo);
        out->Add<double*>(tau);
        out->Add<int*>(devInfo);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnDorgqr Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnCungqr){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnCungqr"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    int k = in->Get<int>();
    cuComplex *A = in->Get<cuComplex*>();
    int lda = in->Get<int>();
    cuComplex *tau = in->Get<cuComplex*>();
    cuComplex *work = in->Get<cuComplex*>();
    int lwork = in->Get<int>();
    int *devInfo = in->Get<int*>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnCungqr(handle, m, n, k, A, lda, tau, work, lwork, devInfo);
        out->Add<cuComplex*>(tau);
        out->Add<int*>(devInfo);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnCungqr Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnZungqr){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnZungqr"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    int k = in->Get<int>();
    cuDoubleComplex *A = in->Get<cuDoubleComplex*>();
    int lda = in->Get<int>();
    cuDoubleComplex *tau = in->Get<cuDoubleComplex*>();
    cuDoubleComplex *work = in->Get<cuDoubleComplex*>();
    int lwork = in->Get<int>();
    int *devInfo = in->Get<int*>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnZungqr(handle, m, n, k, A, lda, tau, work, lwork, devInfo);
        out->Add<cuDoubleComplex*>(tau);
        out->Add<int*>(devInfo);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnZungqr Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnSsytrf_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnSsytrf_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int n = in->Get<int>();
    float *A = in->Get<float*>();
    int lda = in->Get<int>();
    int lwork;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnSsytrf_bufferSize(handle, n, A, lda, &lwork);
        out->Add<int>(lwork);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnSsytrf_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnDsytrf_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnDsytrf_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int n = in->Get<int>();
    double *A = in->Get<double*>();
    int lda = in->Get<int>();
    int lwork;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnDsytrf_bufferSize(handle, n, A, lda, &lwork);
        out->Add<int>(lwork);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnDsytrf_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnCsytrf_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnCsytrf_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int n = in->Get<int>();
    cuComplex *A = in->Get<cuComplex*>();
    int lda = in->Get<int>();
    int lwork;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnCsytrf_bufferSize(handle, n, A, lda, &lwork);
        out->Add<int>(lwork);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnCsytrf_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnZsytrf_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnZsytrf_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int n = in->Get<int>();
    cuDoubleComplex *A = in->Get<cuDoubleComplex*>();
    int lda = in->Get<int>();
    int lwork;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnZsytrf_bufferSize(handle, n, A, lda, &lwork);
        out->Add<int>(lwork);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnZsytrf_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnSsytrf){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnSsytrf"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    int n = in->Get<int>();
    float *A = in->Get<float*>();
    int lda = in->Get<int>();
    int *ipiv = in->Get<int*>();
    float *work = in->Get<float*>();
    int lwork = in->Get<int>();
    int *devInfo = in->Get<int*>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnSsytrf(handle, uplo, n, A, lda, ipiv, work, lwork, devInfo);
        out->Add<int*>(ipiv);
        out->Add<int*>(devInfo);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnSsytrf Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnDsytrf){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnDsytrf"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    int n = in->Get<int>();
    double *A = in->Get<double*>();
    int lda = in->Get<int>();
    int *ipiv = in->Get<int*>();
    double *work = in->Get<double*>();
    int lwork = in->Get<int>();
    int *devInfo = in->Get<int*>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnDsytrf(handle, uplo, n, A, lda, ipiv, work, lwork, devInfo);
        out->Add<int*>(ipiv);
        out->Add<int*>(devInfo);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnDsytrf Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnCsytrf){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnCsytrf"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    int n = in->Get<int>();
    cuComplex *A = in->Get<cuComplex*>();
    int lda = in->Get<int>();
    int *ipiv = in->Get<int*>();
    cuComplex *work = in->Get<cuComplex*>();
    int lwork = in->Get<int>();
    int *devInfo = in->Get<int*>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnCsytrf(handle, uplo, n, A, lda, ipiv, work, lwork, devInfo);
        out->Add<int*>(ipiv);
        out->Add<int*>(devInfo);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnCsytrf Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnZsytrf){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnZsytrf"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    int n = in->Get<int>();
    cuDoubleComplex *A = in->Get<cuDoubleComplex*>();
    int lda = in->Get<int>();
    int *ipiv = in->Get<int*>();
    cuDoubleComplex *work = in->Get<cuDoubleComplex*>();
    int lwork = in->Get<int>();
    int *devInfo = in->Get<int*>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnZsytrf(handle, uplo, n, A, lda, ipiv, work, lwork, devInfo);
        out->Add<int*>(ipiv);
        out->Add<int*>(devInfo);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnZsytrf Executed");
    return std::make_shared<Result>(cs, out);
}