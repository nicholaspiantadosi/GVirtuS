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

CUSOLVER_ROUTINE_HANDLER(DnSgebrd_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnSgebrd_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    int Lwork;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnSgebrd_bufferSize(handle, m, n, &Lwork);
        out->AddMarshal<int>(Lwork);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnSgebrd_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnDgebrd_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnDgebrd_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    int Lwork;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnDgebrd_bufferSize(handle, m, n, &Lwork);
        out->AddMarshal<int>(Lwork);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnDgebrd_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnCgebrd_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnCgebrd_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    int Lwork;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnCgebrd_bufferSize(handle, m, n, &Lwork);
        out->AddMarshal<int>(Lwork);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnCgebrd_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnZgebrd_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnZgebrd_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    int Lwork;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnZgebrd_bufferSize(handle, m, n, &Lwork);
        out->AddMarshal<int>(Lwork);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnZgebrd_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnSgebrd){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnSgebrd"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    float* A = in->GetFromMarshal<float*>();
    int lda = in->Get<int>();
    float* D = in->GetFromMarshal<float*>();
    float* E = in->GetFromMarshal<float*>();
    float* TAUQ = in->GetFromMarshal<float*>();
    float* TAUP = in->GetFromMarshal<float*>();
    float *Work = in->GetFromMarshal<float*>();
    int Lwork = in->Get<int>();
    int *devInfo = in->GetFromMarshal<int*>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnSgebrd(handle, m, n, A, lda, D, E, TAUQ, TAUP, Work, Lwork, devInfo);
        out->Add<float*>(D);
        out->Add<float*>(E);
        out->Add<float*>(TAUQ);
        out->Add<float*>(TAUP);
        out->Add<int*>(devInfo);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnSgebrd Executed");
    return std::make_shared<Result>(cs,out);
}

CUSOLVER_ROUTINE_HANDLER(DnDgebrd){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnDgebrd"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    double* A = in->GetFromMarshal<double*>();
    int lda = in->Get<int>();
    double* D = in->GetFromMarshal<double*>();
    double* E = in->GetFromMarshal<double*>();
    double* TAUQ = in->GetFromMarshal<double*>();
    double* TAUP = in->GetFromMarshal<double*>();
    double *Work = in->GetFromMarshal<double*>();
    int Lwork = in->Get<int>();
    int *devInfo = in->GetFromMarshal<int*>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnDgebrd(handle, m, n, A, lda, D, E, TAUQ, TAUP, Work, Lwork, devInfo);
        out->Add<double*>(D);
        out->Add<double*>(E);
        out->Add<double*>(TAUQ);
        out->Add<double*>(TAUP);
        out->Add<int*>(devInfo);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnDgebrd Executed");
    return std::make_shared<Result>(cs,out);
}

CUSOLVER_ROUTINE_HANDLER(DnCgebrd){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnCgebrd"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    cuComplex* A = in->GetFromMarshal<cuComplex*>();
    int lda = in->Get<int>();
    float* D = in->GetFromMarshal<float*>();
    float* E = in->GetFromMarshal<float*>();
    cuComplex* TAUQ = in->GetFromMarshal<cuComplex*>();
    cuComplex* TAUP = in->GetFromMarshal<cuComplex*>();
    cuComplex *Work = in->GetFromMarshal<cuComplex*>();
    int Lwork = in->Get<int>();
    int *devInfo = in->GetFromMarshal<int*>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnCgebrd(handle, m, n, A, lda, D, E, TAUQ, TAUP, Work, Lwork, devInfo);
        out->Add<float*>(D);
        out->Add<float*>(E);
        out->Add<cuComplex*>(TAUQ);
        out->Add<cuComplex*>(TAUP);
        out->Add<int*>(devInfo);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnCgebrd Executed");
    return std::make_shared<Result>(cs,out);
}

CUSOLVER_ROUTINE_HANDLER(DnZgebrd){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnZgebrd"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    cuDoubleComplex* A = in->GetFromMarshal<cuDoubleComplex*>();
    int lda = in->Get<int>();
    double* D = in->GetFromMarshal<double*>();
    double* E = in->GetFromMarshal<double*>();
    cuDoubleComplex* TAUQ = in->GetFromMarshal<cuDoubleComplex*>();
    cuDoubleComplex* TAUP = in->GetFromMarshal<cuDoubleComplex*>();
    cuDoubleComplex *Work = in->GetFromMarshal<cuDoubleComplex*>();
    int Lwork = in->Get<int>();
    int *devInfo = in->GetFromMarshal<int*>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnZgebrd(handle, m, n, A, lda, D, E, TAUQ, TAUP, Work, Lwork, devInfo);
        out->Add<double*>(D);
        out->Add<double*>(E);
        out->Add<cuDoubleComplex*>(TAUQ);
        out->Add<cuDoubleComplex*>(TAUP);
        out->Add<int*>(devInfo);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnZgebrd Executed");
    return std::make_shared<Result>(cs,out);
}

CUSOLVER_ROUTINE_HANDLER(DnSorgbr_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnSorgbr_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    cublasSideMode_t side = in->Get<cublasSideMode_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    int k = in->Get<int>();
    const float *A = in->Get<float*>();
    int lda = in->Get<int>();
    const float *tau = in->Get<float*>();
    int Lwork;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnSorgbr_bufferSize(handle, side, m, n, k, A, lda, tau, &Lwork);
        out->AddMarshal<int>(Lwork);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnSorgbr_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnDorgbr_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnDorgbr_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    cublasSideMode_t side = in->Get<cublasSideMode_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    int k = in->Get<int>();
    const double *A = in->Get<double*>();
    int lda = in->Get<int>();
    const double *tau = in->Get<double*>();
    int Lwork;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnDorgbr_bufferSize(handle, side, m, n, k, A, lda, tau, &Lwork);
        out->AddMarshal<int>(Lwork);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnDorgbr_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnCungbr_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnCungbr_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    cublasSideMode_t side = in->Get<cublasSideMode_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    int k = in->Get<int>();
    const cuComplex *A = in->Get<cuComplex*>();
    int lda = in->Get<int>();
    const cuComplex *tau = in->Get<cuComplex*>();
    int Lwork;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnCungbr_bufferSize(handle, side, m, n, k, A, lda, tau, &Lwork);
        out->AddMarshal<int>(Lwork);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnCungbr_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnZungbr_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnZungbr_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    cublasSideMode_t side = in->Get<cublasSideMode_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    int k = in->Get<int>();
    const cuDoubleComplex *A = in->Get<cuDoubleComplex*>();
    int lda = in->Get<int>();
    const cuDoubleComplex *tau = in->Get<cuDoubleComplex*>();
    int Lwork;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnZungbr_bufferSize(handle, side, m, n, k, A, lda, tau, &Lwork);
        out->AddMarshal<int>(Lwork);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnZungbr_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnSorgbr){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnSorgbr"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    cublasSideMode_t side = in->Get<cublasSideMode_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    int k = in->Get<int>();
    float* A = in->GetFromMarshal<float*>();
    int lda = in->Get<int>();
    float* tau = in->GetFromMarshal<float*>();
    float *work = in->GetFromMarshal<float*>();
    int lwork = in->Get<int>();
    int *devInfo = in->GetFromMarshal<int*>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnSorgbr(handle, side, m, n, k, A, lda, tau, work, lwork, devInfo);
        out->Add<float*>(tau);
        out->Add<int*>(devInfo);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnSorgbr Executed");
    return std::make_shared<Result>(cs,out);
}

CUSOLVER_ROUTINE_HANDLER(DnDorgbr){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnDorgbr"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    cublasSideMode_t side = in->Get<cublasSideMode_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    int k = in->Get<int>();
    double* A = in->GetFromMarshal<double*>();
    int lda = in->Get<int>();
    double* tau = in->GetFromMarshal<double*>();
    double *work = in->GetFromMarshal<double*>();
    int lwork = in->Get<int>();
    int *devInfo = in->GetFromMarshal<int*>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnDorgbr(handle, side, m, n, k, A, lda, tau, work, lwork, devInfo);
        out->Add<double*>(tau);
        out->Add<int*>(devInfo);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnDorgbr Executed");
    return std::make_shared<Result>(cs,out);
}

CUSOLVER_ROUTINE_HANDLER(DnCungbr){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnCungbr"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    cublasSideMode_t side = in->Get<cublasSideMode_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    int k = in->Get<int>();
    cuComplex* A = in->GetFromMarshal<cuComplex*>();
    int lda = in->Get<int>();
    cuComplex* tau = in->GetFromMarshal<cuComplex*>();
    cuComplex *work = in->GetFromMarshal<cuComplex*>();
    int lwork = in->Get<int>();
    int *devInfo = in->GetFromMarshal<int*>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnCungbr(handle, side, m, n, k, A, lda, tau, work, lwork, devInfo);
        out->Add<cuComplex*>(tau);
        out->Add<int*>(devInfo);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnCungbr Executed");
    return std::make_shared<Result>(cs,out);
}

CUSOLVER_ROUTINE_HANDLER(DnZungbr){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnZungbr"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    cublasSideMode_t side = in->Get<cublasSideMode_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    int k = in->Get<int>();
    cuDoubleComplex* A = in->GetFromMarshal<cuDoubleComplex*>();
    int lda = in->Get<int>();
    cuDoubleComplex* tau = in->GetFromMarshal<cuDoubleComplex*>();
    cuDoubleComplex *work = in->GetFromMarshal<cuDoubleComplex*>();
    int lwork = in->Get<int>();
    int *devInfo = in->GetFromMarshal<int*>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnZungbr(handle, side, m, n, k, A, lda, tau, work, lwork, devInfo);
        out->Add<cuDoubleComplex*>(tau);
        out->Add<int*>(devInfo);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnZungbr Executed");
    return std::make_shared<Result>(cs,out);
}