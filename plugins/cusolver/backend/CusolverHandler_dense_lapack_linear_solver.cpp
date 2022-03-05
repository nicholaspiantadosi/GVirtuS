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

