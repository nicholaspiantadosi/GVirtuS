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

CUSOLVER_ROUTINE_HANDLER(DnSsytrd_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnSsytrd_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    int n = in->Get<int>();
    const float *A = in->Get<float*>();
    int lda = in->Get<int>();
    const float *d = in->Get<float*>();
    const float *e = in->Get<float*>();
    const float *tau = in->Get<float*>();
    int lwork;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnSsytrd_bufferSize(handle, uplo, n, A, lda, d, e, tau, &lwork);
        out->AddMarshal<int>(lwork);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnSsytrd_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnDsytrd_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnDsytrd_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    int n = in->Get<int>();
    const double *A = in->Get<double*>();
    int lda = in->Get<int>();
    const double *d = in->Get<double*>();
    const double *e = in->Get<double*>();
    const double *tau = in->Get<double*>();
    int lwork;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnDsytrd_bufferSize(handle, uplo, n, A, lda, d, e, tau, &lwork);
        out->AddMarshal<int>(lwork);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnDsytrd_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnChetrd_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnChetrd_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    int n = in->Get<int>();
    const cuComplex *A = in->Get<cuComplex*>();
    int lda = in->Get<int>();
    const float *d = in->Get<float*>();
    const float *e = in->Get<float*>();
    const cuComplex *tau = in->Get<cuComplex*>();
    int lwork;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnChetrd_bufferSize(handle, uplo, n, A, lda, d, e, tau, &lwork);
        out->AddMarshal<int>(lwork);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnChetrd_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnZhetrd_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnZhetrd_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    int n = in->Get<int>();
    const cuDoubleComplex *A = in->Get<cuDoubleComplex*>();
    int lda = in->Get<int>();
    const double *d = in->Get<double*>();
    const double *e = in->Get<double*>();
    const cuDoubleComplex *tau = in->Get<cuDoubleComplex*>();
    int lwork;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnZhetrd_bufferSize(handle, uplo, n, A, lda, d, e, tau, &lwork);
        out->AddMarshal<int>(lwork);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnZhetrd_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnSsytrd){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnSsytrd"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    int n = in->Get<int>();
    float* A = in->GetFromMarshal<float*>();
    int lda = in->Get<int>();
    float* d = in->GetFromMarshal<float*>();
    float* e = in->GetFromMarshal<float*>();
    float* tau = in->GetFromMarshal<float*>();
    float *work = in->GetFromMarshal<float*>();
    int lwork = in->Get<int>();
    int *devInfo = in->GetFromMarshal<int*>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnSsytrd(handle, uplo, n, A, lda, d, e, tau, work, lwork, devInfo);
        out->Add<float*>(d);
        out->Add<float*>(e);
        out->Add<float*>(tau);
        out->Add<int*>(devInfo);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnSsytrd Executed");
    return std::make_shared<Result>(cs,out);
}

CUSOLVER_ROUTINE_HANDLER(DnDsytrd){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnDsytrd"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    int n = in->Get<int>();
    double* A = in->GetFromMarshal<double*>();
    int lda = in->Get<int>();
    double* d = in->GetFromMarshal<double*>();
    double* e = in->GetFromMarshal<double*>();
    double* tau = in->GetFromMarshal<double*>();
    double *work = in->GetFromMarshal<double*>();
    int lwork = in->Get<int>();
    int *devInfo = in->GetFromMarshal<int*>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnDsytrd(handle, uplo, n, A, lda, d, e, tau, work, lwork, devInfo);
        out->Add<double*>(d);
        out->Add<double*>(e);
        out->Add<double*>(tau);
        out->Add<int*>(devInfo);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnDsytrd Executed");
    return std::make_shared<Result>(cs,out);
}

CUSOLVER_ROUTINE_HANDLER(DnChetrd){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnChetrd"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    int n = in->Get<int>();
    cuComplex* A = in->GetFromMarshal<cuComplex*>();
    int lda = in->Get<int>();
    float* d = in->GetFromMarshal<float*>();
    float* e = in->GetFromMarshal<float*>();
    cuComplex* tau = in->GetFromMarshal<cuComplex*>();
    cuComplex *work = in->GetFromMarshal<cuComplex*>();
    int lwork = in->Get<int>();
    int *devInfo = in->GetFromMarshal<int*>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnChetrd(handle, uplo, n, A, lda, d, e, tau, work, lwork, devInfo);
        out->Add<float*>(d);
        out->Add<float*>(e);
        out->Add<cuComplex*>(tau);
        out->Add<int*>(devInfo);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnChetrd Executed");
    return std::make_shared<Result>(cs,out);
}

CUSOLVER_ROUTINE_HANDLER(DnZhetrd){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnZhetrd"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    int n = in->Get<int>();
    cuDoubleComplex* A = in->GetFromMarshal<cuDoubleComplex*>();
    int lda = in->Get<int>();
    double* d = in->GetFromMarshal<double*>();
    double* e = in->GetFromMarshal<double*>();
    cuDoubleComplex* tau = in->GetFromMarshal<cuDoubleComplex*>();
    cuDoubleComplex *work = in->GetFromMarshal<cuDoubleComplex*>();
    int lwork = in->Get<int>();
    int *devInfo = in->GetFromMarshal<int*>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnZhetrd(handle, uplo, n, A, lda, d, e, tau, work, lwork, devInfo);
        out->Add<double*>(d);
        out->Add<double*>(e);
        out->Add<cuDoubleComplex*>(tau);
        out->Add<int*>(devInfo);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnZhetrd Executed");
    return std::make_shared<Result>(cs,out);
}

CUSOLVER_ROUTINE_HANDLER(DnSormtr_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnSormtr_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    cublasSideMode_t side = in->Get<cublasSideMode_t>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    cublasOperation_t trans = in->Get<cublasOperation_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    const float *A = in->Get<float*>();
    int lda = in->Get<int>();
    const float *tau = in->Get<float*>();
    const float *C = in->Get<float*>();
    int ldc = in->Get<int>();
    int lwork;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnSormtr_bufferSize(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, &lwork);
        out->AddMarshal<int>(lwork);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnSormtr_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnDormtr_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnDormtr_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    cublasSideMode_t side = in->Get<cublasSideMode_t>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    cublasOperation_t trans = in->Get<cublasOperation_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    const double *A = in->Get<double*>();
    int lda = in->Get<int>();
    const double *tau = in->Get<double*>();
    const double *C = in->Get<double*>();
    int ldc = in->Get<int>();
    int lwork;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnDormtr_bufferSize(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, &lwork);
        out->AddMarshal<int>(lwork);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnDormtr_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnCunmtr_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnCunmtr_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    cublasSideMode_t side = in->Get<cublasSideMode_t>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    cublasOperation_t trans = in->Get<cublasOperation_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    const cuComplex *A = in->Get<cuComplex*>();
    int lda = in->Get<int>();
    const cuComplex *tau = in->Get<cuComplex*>();
    const cuComplex *C = in->Get<cuComplex*>();
    int ldc = in->Get<int>();
    int lwork;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnCunmtr_bufferSize(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, &lwork);
        out->AddMarshal<int>(lwork);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnCunmtr_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnZunmtr_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnZunmtr_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    cublasSideMode_t side = in->Get<cublasSideMode_t>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    cublasOperation_t trans = in->Get<cublasOperation_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    const cuDoubleComplex *A = in->Get<cuDoubleComplex*>();
    int lda = in->Get<int>();
    const cuDoubleComplex *tau = in->Get<cuDoubleComplex*>();
    const cuDoubleComplex *C = in->Get<cuDoubleComplex*>();
    int ldc = in->Get<int>();
    int lwork;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnZunmtr_bufferSize(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, &lwork);
        out->AddMarshal<int>(lwork);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnZunmtr_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnSormtr){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnSormtr"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    cublasSideMode_t side = in->Get<cublasSideMode_t>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    cublasOperation_t trans = in->Get<cublasOperation_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    float *A = in->Get<float*>();
    int lda = in->Get<int>();
    float *tau = in->Get<float*>();
    float *C = in->Get<float*>();
    int ldc = in->Get<int>();
    float *work = in->GetFromMarshal<float*>();
    int lwork = in->Get<int>();
    int *devInfo = in->GetFromMarshal<int*>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnSormtr(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, work, lwork, devInfo);
        out->Add<float*>(tau);
        out->Add<int*>(devInfo);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnSormtr Executed");
    return std::make_shared<Result>(cs,out);
}

CUSOLVER_ROUTINE_HANDLER(DnDormtr){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnDormtr"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    cublasSideMode_t side = in->Get<cublasSideMode_t>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    cublasOperation_t trans = in->Get<cublasOperation_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    double *A = in->Get<double*>();
    int lda = in->Get<int>();
    double *tau = in->Get<double*>();
    double *C = in->Get<double*>();
    int ldc = in->Get<int>();
    double *work = in->GetFromMarshal<double*>();
    int lwork = in->Get<int>();
    int *devInfo = in->GetFromMarshal<int*>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnDormtr(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, work, lwork, devInfo);
        out->Add<double*>(tau);
        out->Add<int*>(devInfo);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnDormtr Executed");
    return std::make_shared<Result>(cs,out);
}

CUSOLVER_ROUTINE_HANDLER(DnCunmtr){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnCunmtr"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    cublasSideMode_t side = in->Get<cublasSideMode_t>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    cublasOperation_t trans = in->Get<cublasOperation_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    cuComplex *A = in->Get<cuComplex*>();
    int lda = in->Get<int>();
    cuComplex *tau = in->Get<cuComplex*>();
    cuComplex *C = in->Get<cuComplex*>();
    int ldc = in->Get<int>();
    cuComplex *work = in->GetFromMarshal<cuComplex*>();
    int lwork = in->Get<int>();
    int *devInfo = in->GetFromMarshal<int*>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnCunmtr(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, work, lwork, devInfo);
        out->Add<cuComplex*>(tau);
        out->Add<int*>(devInfo);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnCunmtr Executed");
    return std::make_shared<Result>(cs,out);
}

CUSOLVER_ROUTINE_HANDLER(DnZunmtr){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnZunmtr"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    cublasSideMode_t side = in->Get<cublasSideMode_t>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    cublasOperation_t trans = in->Get<cublasOperation_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    cuDoubleComplex *A = in->Get<cuDoubleComplex*>();
    int lda = in->Get<int>();
    cuDoubleComplex *tau = in->Get<cuDoubleComplex*>();
    cuDoubleComplex *C = in->Get<cuDoubleComplex*>();
    int ldc = in->Get<int>();
    cuDoubleComplex *work = in->GetFromMarshal<cuDoubleComplex*>();
    int lwork = in->Get<int>();
    int *devInfo = in->GetFromMarshal<int*>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnZunmtr(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, work, lwork, devInfo);
        out->Add<cuDoubleComplex*>(tau);
        out->Add<int*>(devInfo);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnZunmtr Executed");
    return std::make_shared<Result>(cs,out);
}

CUSOLVER_ROUTINE_HANDLER(DnSorgtr_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnSorgtr_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    int n = in->Get<int>();
    const float *A = in->Get<float*>();
    int lda = in->Get<int>();
    const float *tau = in->Get<float*>();
    int lwork;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnSorgtr_bufferSize(handle, uplo, n, A, lda, tau, &lwork);
        out->AddMarshal<int>(lwork);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnSorgtr_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnDorgtr_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnDorgtr_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    int n = in->Get<int>();
    const double *A = in->Get<double*>();
    int lda = in->Get<int>();
    const double *tau = in->Get<double*>();
    int lwork;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnDorgtr_bufferSize(handle, uplo, n, A, lda, tau, &lwork);
        out->AddMarshal<int>(lwork);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnDorgtr_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnCungtr_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnCungtr_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    int n = in->Get<int>();
    const cuComplex *A = in->Get<cuComplex*>();
    int lda = in->Get<int>();
    const cuComplex *tau = in->Get<cuComplex*>();
    int lwork;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnCungtr_bufferSize(handle, uplo, n, A, lda, tau, &lwork);
        out->AddMarshal<int>(lwork);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnCungtr_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnZungtr_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnZungtr_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    int n = in->Get<int>();
    const cuDoubleComplex *A = in->Get<cuDoubleComplex*>();
    int lda = in->Get<int>();
    const cuDoubleComplex *tau = in->Get<cuDoubleComplex*>();
    int lwork;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnZungtr_bufferSize(handle, uplo, n, A, lda, tau, &lwork);
        out->AddMarshal<int>(lwork);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnZungtr_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnSorgtr){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnSorgtr"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    int n = in->Get<int>();
    float *A = in->Get<float*>();
    int lda = in->Get<int>();
    float *tau = in->Get<float*>();
    float *work = in->GetFromMarshal<float*>();
    int lwork = in->Get<int>();
    int *devInfo = in->GetFromMarshal<int*>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnSorgtr(handle, uplo, n, A, lda, tau, work, lwork, devInfo);
        out->Add<float*>(tau);
        out->Add<int*>(devInfo);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnSorgtr Executed");
    return std::make_shared<Result>(cs,out);
}

CUSOLVER_ROUTINE_HANDLER(DnDorgtr){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnDorgtr"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    int n = in->Get<int>();
    double *A = in->Get<double*>();
    int lda = in->Get<int>();
    double *tau = in->Get<double*>();
    double *work = in->GetFromMarshal<double*>();
    int lwork = in->Get<int>();
    int *devInfo = in->GetFromMarshal<int*>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnDorgtr(handle, uplo, n, A, lda, tau, work, lwork, devInfo);
        out->Add<double*>(tau);
        out->Add<int*>(devInfo);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnDorgtr Executed");
    return std::make_shared<Result>(cs,out);
}

CUSOLVER_ROUTINE_HANDLER(DnCungtr){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnCungtr"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    int n = in->Get<int>();
    cuComplex *A = in->Get<cuComplex*>();
    int lda = in->Get<int>();
    cuComplex *tau = in->Get<cuComplex*>();
    cuComplex *work = in->GetFromMarshal<cuComplex*>();
    int lwork = in->Get<int>();
    int *devInfo = in->GetFromMarshal<int*>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnCungtr(handle, uplo, n, A, lda, tau, work, lwork, devInfo);
        out->Add<cuComplex*>(tau);
        out->Add<int*>(devInfo);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnCungtr Executed");
    return std::make_shared<Result>(cs,out);
}

CUSOLVER_ROUTINE_HANDLER(DnZungtr){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnZungtr"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    int n = in->Get<int>();
    cuDoubleComplex *A = in->Get<cuDoubleComplex*>();
    int lda = in->Get<int>();
    cuDoubleComplex *tau = in->Get<cuDoubleComplex*>();
    cuDoubleComplex *work = in->GetFromMarshal<cuDoubleComplex*>();
    int lwork = in->Get<int>();
    int *devInfo = in->GetFromMarshal<int*>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnZungtr(handle, uplo, n, A, lda, tau, work, lwork, devInfo);
        out->Add<cuDoubleComplex*>(tau);
        out->Add<int*>(devInfo);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnZungtr Executed");
    return std::make_shared<Result>(cs,out);
}

CUSOLVER_ROUTINE_HANDLER(DnSgesvd_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnSgesvd_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    int lwork;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnSgesvd_bufferSize(handle, m, n, &lwork);
        out->AddMarshal<int>(lwork);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnSgesvd_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnDgesvd_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnDgesvd_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    int lwork;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnDgesvd_bufferSize(handle, m, n, &lwork);
        out->AddMarshal<int>(lwork);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnDgesvd_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnCgesvd_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnCgesvd_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    int lwork;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnCgesvd_bufferSize(handle, m, n, &lwork);
        out->AddMarshal<int>(lwork);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnCgesvd_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnZgesvd_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnZgesvd_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    int lwork;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnZgesvd_bufferSize(handle, m, n, &lwork);
        out->AddMarshal<int>(lwork);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnZgesvd_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnSgesvd){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnSgesvd"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    signed char jobu = in->Get<signed char>();
    signed char jobvt = in->Get<signed char>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    float *A = in->Get<float*>();
    int lda = in->Get<int>();
    float *S = in->Get<float*>();
    float *U = in->Get<float*>();
    int ldu = in->Get<int>();
    float *VT = in->Get<float*>();
    int ldvt = in->Get<int>();
    float *work = in->GetFromMarshal<float*>();
    int lwork = in->Get<int>();
    float *rwork = in->GetFromMarshal<float*>();
    int *devInfo = in->GetFromMarshal<int*>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnSgesvd(handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt, work, lwork, rwork, devInfo);
        out->Add<float*>(S);
        out->Add<float*>(U);
        out->Add<float*>(VT);
        out->Add<int*>(devInfo);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnSgesvd Executed");
    return std::make_shared<Result>(cs,out);
}

CUSOLVER_ROUTINE_HANDLER(DnDgesvd){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnDgesvd"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    signed char jobu = in->Get<signed char>();
    signed char jobvt = in->Get<signed char>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    double *A = in->Get<double*>();
    int lda = in->Get<int>();
    double *S = in->Get<double*>();
    double *U = in->Get<double*>();
    int ldu = in->Get<int>();
    double *VT = in->Get<double*>();
    int ldvt = in->Get<int>();
    double *work = in->GetFromMarshal<double*>();
    int lwork = in->Get<int>();
    double *rwork = in->GetFromMarshal<double*>();
    int *devInfo = in->GetFromMarshal<int*>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnDgesvd(handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt, work, lwork, rwork, devInfo);
        out->Add<double*>(S);
        out->Add<double*>(U);
        out->Add<double*>(VT);
        out->Add<int*>(devInfo);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnDgesvd Executed");
    return std::make_shared<Result>(cs,out);
}

CUSOLVER_ROUTINE_HANDLER(DnCgesvd){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnCgesvd"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    signed char jobu = in->Get<signed char>();
    signed char jobvt = in->Get<signed char>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    cuComplex *A = in->Get<cuComplex*>();
    int lda = in->Get<int>();
    float *S = in->Get<float*>();
    cuComplex *U = in->Get<cuComplex*>();
    int ldu = in->Get<int>();
    cuComplex *VT = in->Get<cuComplex*>();
    int ldvt = in->Get<int>();
    cuComplex *work = in->GetFromMarshal<cuComplex*>();
    int lwork = in->Get<int>();
    float *rwork = in->GetFromMarshal<float*>();
    int *devInfo = in->GetFromMarshal<int*>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnCgesvd(handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt, work, lwork, rwork, devInfo);
        out->Add<float*>(S);
        out->Add<cuComplex*>(U);
        out->Add<cuComplex*>(VT);
        out->Add<int*>(devInfo);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnCgesvd Executed");
    return std::make_shared<Result>(cs,out);
}

CUSOLVER_ROUTINE_HANDLER(DnZgesvd){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnZgesvd"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    signed char jobu = in->Get<signed char>();
    signed char jobvt = in->Get<signed char>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    cuDoubleComplex *A = in->Get<cuDoubleComplex*>();
    int lda = in->Get<int>();
    double *S = in->Get<double*>();
    cuDoubleComplex *U = in->Get<cuDoubleComplex*>();
    int ldu = in->Get<int>();
    cuDoubleComplex *VT = in->Get<cuDoubleComplex*>();
    int ldvt = in->Get<int>();
    cuDoubleComplex *work = in->GetFromMarshal<cuDoubleComplex*>();
    int lwork = in->Get<int>();
    double *rwork = in->GetFromMarshal<double*>();
    int *devInfo = in->GetFromMarshal<int*>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnZgesvd(handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt, work, lwork, rwork, devInfo);
        out->Add<double*>(S);
        out->Add<cuDoubleComplex*>(U);
        out->Add<cuDoubleComplex*>(VT);
        out->Add<int*>(devInfo);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnZgesvd Executed");
    return std::make_shared<Result>(cs,out);
}

CUSOLVER_ROUTINE_HANDLER(DnGesvd_bufferSize){
        Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnGesvd_bufferSize"));
        CusolverHandler::setLogLevel(&logger);

        cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
        cusolverDnParams_t params = in->Get<cusolverDnParams_t>();
        signed char jobu = in->Get<signed char>();
        signed char jobvt = in->Get<signed char>();
        int64_t m = in->Get<int64_t>();
        int64_t n = in->Get<int64_t>();
        cudaDataType dataTypeA = in->Get<cudaDataType>();
        int64_t lda = in->Get<int64_t>();
        cudaDataType dataTypeS = in->Get<cudaDataType>();
        cudaDataType dataTypeU = in->Get<cudaDataType>();
        int64_t ldu = in->Get<int64_t>();
        cudaDataType dataTypeVT = in->Get<cudaDataType>();
        int64_t ldvt = in->Get<int64_t>();
        cudaDataType computeType = in->Get<cudaDataType>();
        size_t workspaceInBytes;
        void *A;
        void *S;
        void *U;
        void *VT;
        if (dataTypeA == CUDA_R_32F) {
            // float
            A = in->GetFromMarshal<float*>();
            S = in->GetFromMarshal<float*>();
            U = in->GetFromMarshal<float*>();
            VT = in->GetFromMarshal<float*>();
        } else if (dataTypeA == CUDA_R_64F) {
            // double
            A = in->GetFromMarshal<double*>();
            S = in->GetFromMarshal<double*>();
            U = in->GetFromMarshal<double*>();
            VT = in->GetFromMarshal<double*>();
        } else if (dataTypeA == CUDA_C_32F) {
            // cuComplex
            A = in->GetFromMarshal<cuComplex*>();
            S = in->GetFromMarshal<float*>();
            U = in->GetFromMarshal<cuComplex*>();
            VT = in->GetFromMarshal<cuComplex*>();
        } else if (dataTypeA == CUDA_C_64F) {
            // cuDoubleComplex
            A = in->GetFromMarshal<cuDoubleComplex*>();
            S = in->GetFromMarshal<double*>();
            U = in->GetFromMarshal<cuDoubleComplex*>();
            VT = in->GetFromMarshal<cuDoubleComplex*>();
        } else {
            throw "Type not supported by GVirtus!";
        }
        cusolverStatus_t cs;
        std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
        try{
            cs = cusolverDnGesvd_bufferSize(handle, params, jobu, jobvt, m, n, dataTypeA, A, lda, dataTypeS, S, dataTypeU, U, ldu, dataTypeVT, VT, ldvt, computeType, &workspaceInBytes);
            out->AddMarshal<size_t>(workspaceInBytes);
        } catch (string e){
            LOG4CPLUS_DEBUG(logger,e);
            return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
        }
        LOG4CPLUS_DEBUG(logger,"cusolverDnGesvd_bufferSize Executed");
        return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnGesvd){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnGesvd"));
    CusolverHandler::setLogLevel(&logger);

    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    cusolverDnParams_t params = in->Get<cusolverDnParams_t>();
    signed char jobu = in->Get<signed char>();
    signed char jobvt = in->Get<signed char>();
    int64_t m = in->Get<int64_t>();
    int64_t n = in->Get<int64_t>();
    cudaDataType dataTypeA = in->Get<cudaDataType>();
    int64_t lda = in->Get<int64_t>();
    cudaDataType dataTypeS = in->Get<cudaDataType>();
    cudaDataType dataTypeU = in->Get<cudaDataType>();
    int64_t ldu = in->Get<int64_t>();
    cudaDataType dataTypeVT = in->Get<cudaDataType>();
    int64_t ldvt = in->Get<int64_t>();
    cudaDataType computeType = in->Get<cudaDataType>();
    void *pBuffer = in->Get<void*>();
    size_t workspaceInBytes = in->Get<size_t>();
    int *info = in->Get<int*>();
    void *A;
    void *S;
    void *U;
    void *VT;
    if (dataTypeA == CUDA_R_32F) {
        // float
        A = in->GetFromMarshal<float*>();
        S = in->GetFromMarshal<float*>();
        U = in->GetFromMarshal<float*>();
        VT = in->GetFromMarshal<float*>();
    } else if (dataTypeA == CUDA_R_64F) {
        // double
        A = in->GetFromMarshal<double*>();
        S = in->GetFromMarshal<double*>();
        U = in->GetFromMarshal<double*>();
        VT = in->GetFromMarshal<double*>();
    } else if (dataTypeA == CUDA_C_32F) {
        // cuComplex
        A = in->GetFromMarshal<cuComplex*>();
        S = in->GetFromMarshal<float*>();
        U = in->GetFromMarshal<cuComplex*>();
        VT = in->GetFromMarshal<cuComplex*>();
    } else if (dataTypeA == CUDA_C_64F) {
        // cuDoubleComplex
        A = in->GetFromMarshal<cuDoubleComplex*>();
        S = in->GetFromMarshal<double*>();
        U = in->GetFromMarshal<cuDoubleComplex*>();
        VT = in->GetFromMarshal<cuDoubleComplex*>();
    } else {
        throw "Type not supported by GVirtus!";
    }
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnGesvd(handle, params, jobu, jobvt, m, n, dataTypeA, A, lda, dataTypeS, S, dataTypeU, U, ldu, dataTypeVT, VT, ldvt, computeType, pBuffer, workspaceInBytes, info);
        out->Add<void*>(S);
        out->Add<void*>(U);
        out->Add<void*>(VT);
        out->Add<int*>(info);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnGesvd Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnSgesvdj_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnSgesvdj_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    cusolverEigMode_t jobz = in->Get<cusolverEigMode_t>();
    int econ = in->Get<int>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    float *A = in->Get<float*>();
    int lda = in->Get<int>();
    float *S = in->Get<float*>();
    float *U = in->Get<float*>();
    int ldu = in->Get<int>();
    float *V = in->Get<float*>();
    int ldv = in->Get<int>();
    int lwork;
    gesvdjInfo_t params = in->Get<gesvdjInfo_t>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnSgesvdj_bufferSize(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, &lwork, params);
        out->AddMarshal<int>(lwork);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnSgesvdj_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnDgesvdj_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnDgesvdj_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    cusolverEigMode_t jobz = in->Get<cusolverEigMode_t>();
    int econ = in->Get<int>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    double *A = in->Get<double*>();
    int lda = in->Get<int>();
    double *S = in->Get<double*>();
    double *U = in->Get<double*>();
    int ldu = in->Get<int>();
    double *V = in->Get<double*>();
    int ldv = in->Get<int>();
    int lwork;
    gesvdjInfo_t params = in->Get<gesvdjInfo_t>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnDgesvdj_bufferSize(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, &lwork, params);
        out->AddMarshal<int>(lwork);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnDgesvdj_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnCgesvdj_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnCgesvdj_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    cusolverEigMode_t jobz = in->Get<cusolverEigMode_t>();
    int econ = in->Get<int>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    cuComplex *A = in->Get<cuComplex*>();
    int lda = in->Get<int>();
    float *S = in->Get<float*>();
    cuComplex *U = in->Get<cuComplex*>();
    int ldu = in->Get<int>();
    cuComplex *V = in->Get<cuComplex*>();
    int ldv = in->Get<int>();
    int lwork;
    gesvdjInfo_t params = in->Get<gesvdjInfo_t>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnCgesvdj_bufferSize(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, &lwork, params);
        out->AddMarshal<int>(lwork);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnCgesvdj_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnZgesvdj_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnZgesvdj_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    cusolverEigMode_t jobz = in->Get<cusolverEigMode_t>();
    int econ = in->Get<int>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    cuDoubleComplex *A = in->Get<cuDoubleComplex*>();
    int lda = in->Get<int>();
    double *S = in->Get<double*>();
    cuDoubleComplex *U = in->Get<cuDoubleComplex*>();
    int ldu = in->Get<int>();
    cuDoubleComplex *V = in->Get<cuDoubleComplex*>();
    int ldv = in->Get<int>();
    int lwork;
    gesvdjInfo_t params = in->Get<gesvdjInfo_t>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnZgesvdj_bufferSize(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, &lwork, params);
        out->AddMarshal<int>(lwork);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnZgesvdj_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnSgesvdj){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnSgesvdj"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    cusolverEigMode_t jobz = in->Get<cusolverEigMode_t>();
    int econ = in->Get<int>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    float *A = in->Get<float*>();
    int lda = in->Get<int>();
    float *S = in->Get<float*>();
    float *U = in->Get<float*>();
    int ldu = in->Get<int>();
    float *V = in->Get<float*>();
    int ldv = in->Get<int>();
    float *work = in->GetFromMarshal<float*>();
    int lwork = in->Get<int>();
    int *info = in->GetFromMarshal<int*>();
    gesvdjInfo_t params = in->Get<gesvdjInfo_t>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnSgesvdj(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info, params);
        out->Add<float*>(S);
        out->Add<float*>(U);
        out->Add<float*>(V);
        out->Add<int*>(info);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnSgesvdj Executed");
    return std::make_shared<Result>(cs,out);
}

CUSOLVER_ROUTINE_HANDLER(DnDgesvdj){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnDgesvdj"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    cusolverEigMode_t jobz = in->Get<cusolverEigMode_t>();
    int econ = in->Get<int>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    double *A = in->Get<double*>();
    int lda = in->Get<int>();
    double *S = in->Get<double*>();
    double *U = in->Get<double*>();
    int ldu = in->Get<int>();
    double *V = in->Get<double*>();
    int ldv = in->Get<int>();
    double *work = in->GetFromMarshal<double*>();
    int lwork = in->Get<int>();
    int *info = in->GetFromMarshal<int*>();
    gesvdjInfo_t params = in->Get<gesvdjInfo_t>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnDgesvdj(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info, params);
        out->Add<double*>(S);
        out->Add<double*>(U);
        out->Add<double*>(V);
        out->Add<int*>(info);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnDgesvdj Executed");
    return std::make_shared<Result>(cs,out);
}

CUSOLVER_ROUTINE_HANDLER(DnCgesvdj){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnCgesvdj"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    cusolverEigMode_t jobz = in->Get<cusolverEigMode_t>();
    int econ = in->Get<int>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    cuComplex *A = in->Get<cuComplex*>();
    int lda = in->Get<int>();
    float *S = in->Get<float*>();
    cuComplex *U = in->Get<cuComplex*>();
    int ldu = in->Get<int>();
    cuComplex *V = in->Get<cuComplex*>();
    int ldv = in->Get<int>();
    cuComplex *work = in->GetFromMarshal<cuComplex*>();
    int lwork = in->Get<int>();
    int *info = in->GetFromMarshal<int*>();
    gesvdjInfo_t params = in->Get<gesvdjInfo_t>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnCgesvdj(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info, params);
        out->Add<float*>(S);
        out->Add<cuComplex*>(U);
        out->Add<cuComplex*>(V);
        out->Add<int*>(info);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnCgesvdj Executed");
    return std::make_shared<Result>(cs,out);
}

CUSOLVER_ROUTINE_HANDLER(DnZgesvdj){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnZgesvdj"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    cusolverEigMode_t jobz = in->Get<cusolverEigMode_t>();
    int econ = in->Get<int>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    cuDoubleComplex *A = in->Get<cuDoubleComplex*>();
    int lda = in->Get<int>();
    double *S = in->Get<double*>();
    cuDoubleComplex *U = in->Get<cuDoubleComplex*>();
    int ldu = in->Get<int>();
    cuDoubleComplex *V = in->Get<cuDoubleComplex*>();
    int ldv = in->Get<int>();
    cuDoubleComplex *work = in->GetFromMarshal<cuDoubleComplex*>();
    int lwork = in->Get<int>();
    int *info = in->GetFromMarshal<int*>();
    gesvdjInfo_t params = in->Get<gesvdjInfo_t>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnZgesvdj(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info, params);
        out->Add<double*>(S);
        out->Add<cuDoubleComplex*>(U);
        out->Add<cuDoubleComplex*>(V);
        out->Add<int*>(info);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnZgesvdj Executed");
    return std::make_shared<Result>(cs,out);
}

CUSOLVER_ROUTINE_HANDLER(DnSgesvdjBatched_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnSgesvdjBatched_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    cusolverEigMode_t jobz = in->Get<cusolverEigMode_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    float *A = in->Get<float*>();
    int lda = in->Get<int>();
    float *S = in->Get<float*>();
    float *U = in->Get<float*>();
    int ldu = in->Get<int>();
    float *V = in->Get<float*>();
    int ldv = in->Get<int>();
    int lwork;
    gesvdjInfo_t params = in->Get<gesvdjInfo_t>();
    int batchSize = in->Get<int>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnSgesvdjBatched_bufferSize(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, &lwork, params, batchSize);
        out->AddMarshal<int>(lwork);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnSgesvdjBatched_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnDgesvdjBatched_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnDgesvdjBatched_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    cusolverEigMode_t jobz = in->Get<cusolverEigMode_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    double *A = in->Get<double*>();
    int lda = in->Get<int>();
    double *S = in->Get<double*>();
    double *U = in->Get<double*>();
    int ldu = in->Get<int>();
    double *V = in->Get<double*>();
    int ldv = in->Get<int>();
    int lwork;
    gesvdjInfo_t params = in->Get<gesvdjInfo_t>();
    int batchSize = in->Get<int>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnDgesvdjBatched_bufferSize(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, &lwork, params, batchSize);
        out->AddMarshal<int>(lwork);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnDgesvdjBatched_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnCgesvdjBatched_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnCgesvdjBatched_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    cusolverEigMode_t jobz = in->Get<cusolverEigMode_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    cuComplex *A = in->Get<cuComplex*>();
    int lda = in->Get<int>();
    float *S = in->Get<float*>();
    cuComplex *U = in->Get<cuComplex*>();
    int ldu = in->Get<int>();
    cuComplex *V = in->Get<cuComplex*>();
    int ldv = in->Get<int>();
    int lwork;
    gesvdjInfo_t params = in->Get<gesvdjInfo_t>();
    int batchSize = in->Get<int>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnCgesvdjBatched_bufferSize(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, &lwork, params, batchSize);
        out->AddMarshal<int>(lwork);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnCgesvdjBatched_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnZgesvdjBatched_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnZgesvdjBatched_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    cusolverEigMode_t jobz = in->Get<cusolverEigMode_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    cuDoubleComplex *A = in->Get<cuDoubleComplex*>();
    int lda = in->Get<int>();
    double *S = in->Get<double*>();
    cuDoubleComplex *U = in->Get<cuDoubleComplex*>();
    int ldu = in->Get<int>();
    cuDoubleComplex *V = in->Get<cuDoubleComplex*>();
    int ldv = in->Get<int>();
    int lwork;
    gesvdjInfo_t params = in->Get<gesvdjInfo_t>();
    int batchSize = in->Get<int>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnZgesvdjBatched_bufferSize(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, &lwork, params, batchSize);
        out->AddMarshal<int>(lwork);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnZgesvdjBatched_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnSgesvdjBatched){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnSgesvdjBatched"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    cusolverEigMode_t jobz = in->Get<cusolverEigMode_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    float *A = in->Get<float*>();
    int lda = in->Get<int>();
    float *S = in->Get<float*>();
    float *U = in->Get<float*>();
    int ldu = in->Get<int>();
    float *V = in->Get<float*>();
    int ldv = in->Get<int>();
    float *work = in->GetFromMarshal<float*>();
    int lwork = in->Get<int>();
    int *info = in->GetFromMarshal<int*>();
    gesvdjInfo_t params = in->Get<gesvdjInfo_t>();
    int batchSize = in->Get<int>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnSgesvdjBatched(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info, params, batchSize);
        out->Add<float*>(S);
        out->Add<float*>(U);
        out->Add<float*>(V);
        out->Add<int*>(info);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnSgesvdjBatched Executed");
    return std::make_shared<Result>(cs,out);
}

CUSOLVER_ROUTINE_HANDLER(DnDgesvdjBatched){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnDgesvdjBatched"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    cusolverEigMode_t jobz = in->Get<cusolverEigMode_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    double *A = in->Get<double*>();
    int lda = in->Get<int>();
    double *S = in->Get<double*>();
    double *U = in->Get<double*>();
    int ldu = in->Get<int>();
    double *V = in->Get<double*>();
    int ldv = in->Get<int>();
    double *work = in->GetFromMarshal<double*>();
    int lwork = in->Get<int>();
    int *info = in->GetFromMarshal<int*>();
    gesvdjInfo_t params = in->Get<gesvdjInfo_t>();
    int batchSize = in->Get<int>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnDgesvdjBatched(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info, params, batchSize);
        out->Add<double*>(S);
        out->Add<double*>(U);
        out->Add<double*>(V);
        out->Add<int*>(info);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnDgesvdjBatched Executed");
    return std::make_shared<Result>(cs,out);
}

CUSOLVER_ROUTINE_HANDLER(DnCgesvdjBatched){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnCgesvdjBatched"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    cusolverEigMode_t jobz = in->Get<cusolverEigMode_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    cuComplex *A = in->Get<cuComplex*>();
    int lda = in->Get<int>();
    float *S = in->Get<float*>();
    cuComplex *U = in->Get<cuComplex*>();
    int ldu = in->Get<int>();
    cuComplex *V = in->Get<cuComplex*>();
    int ldv = in->Get<int>();
    cuComplex *work = in->GetFromMarshal<cuComplex*>();
    int lwork = in->Get<int>();
    int *info = in->GetFromMarshal<int*>();
    gesvdjInfo_t params = in->Get<gesvdjInfo_t>();
    int batchSize = in->Get<int>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnCgesvdjBatched(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info, params, batchSize);
        out->Add<float*>(S);
        out->Add<cuComplex*>(U);
        out->Add<cuComplex*>(V);
        out->Add<int*>(info);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnCgesvdjBatched Executed");
    return std::make_shared<Result>(cs,out);
}

CUSOLVER_ROUTINE_HANDLER(DnZgesvdjBatched){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnZgesvdjBatched"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    cusolverEigMode_t jobz = in->Get<cusolverEigMode_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    cuDoubleComplex *A = in->Get<cuDoubleComplex*>();
    int lda = in->Get<int>();
    double *S = in->Get<double*>();
    cuDoubleComplex *U = in->Get<cuDoubleComplex*>();
    int ldu = in->Get<int>();
    cuDoubleComplex *V = in->Get<cuDoubleComplex*>();
    int ldv = in->Get<int>();
    cuDoubleComplex *work = in->GetFromMarshal<cuDoubleComplex*>();
    int lwork = in->Get<int>();
    int *info = in->GetFromMarshal<int*>();
    gesvdjInfo_t params = in->Get<gesvdjInfo_t>();
    int batchSize = in->Get<int>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnZgesvdjBatched(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info, params, batchSize);
        out->Add<double*>(S);
        out->Add<cuDoubleComplex*>(U);
        out->Add<cuDoubleComplex*>(V);
        out->Add<int*>(info);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnZgesvdjBatched Executed");
    return std::make_shared<Result>(cs,out);
}

CUSOLVER_ROUTINE_HANDLER(DnSgesvdaStridedBatched_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnSgesvdaStridedBatched_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    cusolverEigMode_t jobz = in->Get<cusolverEigMode_t>();
    int rank = in->Get<int>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    float *A = in->Get<float*>();
    int lda = in->Get<int>();
    size_t strideA = in->Get<size_t>();
    float *S = in->Get<float*>();
    size_t strideS = in->Get<size_t>();
    float *U = in->Get<float*>();
    int ldu = in->Get<int>();
    size_t strideU = in->Get<size_t>();
    float *V = in->Get<float*>();
    int ldv = in->Get<int>();
    size_t strideV = in->Get<size_t>();
    int lwork;
    int batchSize = in->Get<int>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnSgesvdaStridedBatched_bufferSize(handle, jobz, rank, m, n, A, lda, strideA, S, strideS, U, ldu, strideU, V, ldv, strideV, &lwork, batchSize);
        out->AddMarshal<int>(lwork);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnSgesvdaStridedBatched_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnDgesvdaStridedBatched_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnDgesvdaStridedBatched_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    cusolverEigMode_t jobz = in->Get<cusolverEigMode_t>();
    int rank = in->Get<int>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    double *A = in->Get<double*>();
    int lda = in->Get<int>();
    size_t strideA = in->Get<size_t>();
    double *S = in->Get<double*>();
    size_t strideS = in->Get<size_t>();
    double *U = in->Get<double*>();
    int ldu = in->Get<int>();
    size_t strideU = in->Get<size_t>();
    double *V = in->Get<double*>();
    int ldv = in->Get<int>();
    size_t strideV = in->Get<size_t>();
    int lwork;
    int batchSize = in->Get<int>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnDgesvdaStridedBatched_bufferSize(handle, jobz, rank, m, n, A, lda, strideA, S, strideS, U, ldu, strideU, V, ldv, strideV, &lwork, batchSize);
        out->AddMarshal<int>(lwork);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnDgesvdaStridedBatched_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnCgesvdaStridedBatched_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnCgesvdaStridedBatched_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    cusolverEigMode_t jobz = in->Get<cusolverEigMode_t>();
    int rank = in->Get<int>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    cuComplex *A = in->Get<cuComplex*>();
    int lda = in->Get<int>();
    size_t strideA = in->Get<size_t>();
    float *S = in->Get<float*>();
    size_t strideS = in->Get<size_t>();
    cuComplex *U = in->Get<cuComplex*>();
    int ldu = in->Get<int>();
    size_t strideU = in->Get<size_t>();
    cuComplex *V = in->Get<cuComplex*>();
    int ldv = in->Get<int>();
    size_t strideV = in->Get<size_t>();
    int lwork;
    int batchSize = in->Get<int>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnCgesvdaStridedBatched_bufferSize(handle, jobz, rank, m, n, A, lda, strideA, S, strideS, U, ldu, strideU, V, ldv, strideV, &lwork, batchSize);
        out->AddMarshal<int>(lwork);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnCgesvdaStridedBatched_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnZgesvdaStridedBatched_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnZgesvdaStridedBatched_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    cusolverEigMode_t jobz = in->Get<cusolverEigMode_t>();
    int rank = in->Get<int>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    cuDoubleComplex *A = in->Get<cuDoubleComplex*>();
    int lda = in->Get<int>();
    size_t strideA = in->Get<size_t>();
    double *S = in->Get<double*>();
    size_t strideS = in->Get<size_t>();
    cuDoubleComplex *U = in->Get<cuDoubleComplex*>();
    int ldu = in->Get<int>();
    size_t strideU = in->Get<size_t>();
    cuDoubleComplex *V = in->Get<cuDoubleComplex*>();
    int ldv = in->Get<int>();
    size_t strideV = in->Get<size_t>();
    int lwork;
    int batchSize = in->Get<int>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnZgesvdaStridedBatched_bufferSize(handle, jobz, rank, m, n, A, lda, strideA, S, strideS, U, ldu, strideU, V, ldv, strideV, &lwork, batchSize);
        out->AddMarshal<int>(lwork);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnZgesvdaStridedBatched_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnSgesvdaStridedBatched){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnSgesvdaStridedBatched"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    cusolverEigMode_t jobz = in->Get<cusolverEigMode_t>();
    int rank = in->Get<int>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    float *A = in->Get<float*>();
    int lda = in->Get<int>();
    size_t strideA = in->Get<size_t>();
    float *S = in->Get<float*>();
    size_t strideS = in->Get<size_t>();
    float *U = in->Get<float*>();
    int ldu = in->Get<int>();
    size_t strideU = in->Get<size_t>();
    float *V = in->Get<float*>();
    int ldv = in->Get<int>();
    size_t strideV = in->Get<size_t>();
    float *work = in->GetFromMarshal<float*>();
    int lwork = in->Get<int>();
    int *info = in->GetFromMarshal<int*>();
    int batchSize = in->Get<int>();
    double *h_R_nrmF = in->Assign<double>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnSgesvdaStridedBatched(handle, jobz, rank, m, n, A, lda, strideA, S, strideS, U, ldu, strideU, V, ldv, strideV, work, lwork, info, h_R_nrmF, batchSize);
        out->Add<float*>(S);
        out->Add<float*>(U);
        out->Add<float*>(V);
        out->Add<int*>(info);
        out->Add(h_R_nrmF);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnSgesvdaStridedBatched Executed");
    return std::make_shared<Result>(cs,out);
}

CUSOLVER_ROUTINE_HANDLER(DnDgesvdaStridedBatched){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnDgesvdaStridedBatched"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    cusolverEigMode_t jobz = in->Get<cusolverEigMode_t>();
    int rank = in->Get<int>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    double *A = in->Get<double*>();
    int lda = in->Get<int>();
    size_t strideA = in->Get<size_t>();
    double *S = in->Get<double*>();
    size_t strideS = in->Get<size_t>();
    double *U = in->Get<double*>();
    int ldu = in->Get<int>();
    size_t strideU = in->Get<size_t>();
    double *V = in->Get<double*>();
    int ldv = in->Get<int>();
    size_t strideV = in->Get<size_t>();
    double *work = in->GetFromMarshal<double*>();
    int lwork = in->Get<int>();
    int *info = in->GetFromMarshal<int*>();
    int batchSize = in->Get<int>();
    double *h_R_nrmF = in->Assign<double>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnDgesvdaStridedBatched(handle, jobz, rank, m, n, A, lda, strideA, S, strideS, U, ldu, strideU, V, ldv, strideV, work, lwork, info, h_R_nrmF, batchSize);
        out->Add<double*>(S);
        out->Add<double*>(U);
        out->Add<double*>(V);
        out->Add<int*>(info);
        out->Add(h_R_nrmF);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnDgesvdaStridedBatched Executed");
    return std::make_shared<Result>(cs,out);
}

CUSOLVER_ROUTINE_HANDLER(DnCgesvdaStridedBatched){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnCgesvdaStridedBatched"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    cusolverEigMode_t jobz = in->Get<cusolverEigMode_t>();
    int rank = in->Get<int>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    cuComplex *A = in->Get<cuComplex*>();
    int lda = in->Get<int>();
    size_t strideA = in->Get<size_t>();
    float *S = in->Get<float*>();
    size_t strideS = in->Get<size_t>();
    cuComplex *U = in->Get<cuComplex*>();
    int ldu = in->Get<int>();
    size_t strideU = in->Get<size_t>();
    cuComplex *V = in->Get<cuComplex*>();
    int ldv = in->Get<int>();
    size_t strideV = in->Get<size_t>();
    cuComplex *work = in->GetFromMarshal<cuComplex*>();
    int lwork = in->Get<int>();
    int *info = in->GetFromMarshal<int*>();
    int batchSize = in->Get<int>();
    double *h_R_nrmF = in->Assign<double>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnCgesvdaStridedBatched(handle, jobz, rank, m, n, A, lda, strideA, S, strideS, U, ldu, strideU, V, ldv, strideV, work, lwork, info, h_R_nrmF, batchSize);
        out->Add<float*>(S);
        out->Add<cuComplex*>(U);
        out->Add<cuComplex*>(V);
        out->Add<int*>(info);
        out->Add(h_R_nrmF);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnCgesvdaStridedBatched Executed");
    return std::make_shared<Result>(cs,out);
}

CUSOLVER_ROUTINE_HANDLER(DnZgesvdaStridedBatched){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnZgesvdaStridedBatched"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    cusolverEigMode_t jobz = in->Get<cusolverEigMode_t>();
    int rank = in->Get<int>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    cuDoubleComplex *A = in->Get<cuDoubleComplex*>();
    int lda = in->Get<int>();
    size_t strideA = in->Get<size_t>();
    double *S = in->Get<double*>();
    size_t strideS = in->Get<size_t>();
    cuDoubleComplex *U = in->Get<cuDoubleComplex*>();
    int ldu = in->Get<int>();
    size_t strideU = in->Get<size_t>();
    cuDoubleComplex *V = in->Get<cuDoubleComplex*>();
    int ldv = in->Get<int>();
    size_t strideV = in->Get<size_t>();
    cuDoubleComplex *work = in->GetFromMarshal<cuDoubleComplex*>();
    int lwork = in->Get<int>();
    int *info = in->GetFromMarshal<int*>();
    int batchSize = in->Get<int>();
    double *h_R_nrmF = in->Assign<double>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnZgesvdaStridedBatched(handle, jobz, rank, m, n, A, lda, strideA, S, strideS, U, ldu, strideU, V, ldv, strideV, work, lwork, info, h_R_nrmF, batchSize);
        out->Add<double*>(S);
        out->Add<cuDoubleComplex*>(U);
        out->Add<cuDoubleComplex*>(V);
        out->Add<int*>(info);
        out->Add(h_R_nrmF);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnZgesvdaStridedBatched Executed");
    return std::make_shared<Result>(cs,out);
}

CUSOLVER_ROUTINE_HANDLER(DnSsyevd_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnSsyevd_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    cusolverEigMode_t jobz = in->Get<cusolverEigMode_t>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    int n = in->Get<int>();
    float *A = in->Get<float*>();
    int lda = in->Get<int>();
    float *W = in->Get<float*>();
    int lwork;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnSsyevd_bufferSize(handle, jobz, uplo, n, A, lda, W, &lwork);
        out->AddMarshal<int>(lwork);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnSsyevd_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnDsyevd_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnDsyevd_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    cusolverEigMode_t jobz = in->Get<cusolverEigMode_t>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    int n = in->Get<int>();
    double *A = in->Get<double*>();
    int lda = in->Get<int>();
    double *W = in->Get<double*>();
    int lwork;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnDsyevd_bufferSize(handle, jobz, uplo, n, A, lda, W, &lwork);
        out->AddMarshal<int>(lwork);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnDsyevd_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnCheevd_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnCheevd_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    cusolverEigMode_t jobz = in->Get<cusolverEigMode_t>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    int n = in->Get<int>();
    cuComplex *A = in->Get<cuComplex*>();
    int lda = in->Get<int>();
    float *W = in->Get<float*>();
    int lwork;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnCheevd_bufferSize(handle, jobz, uplo, n, A, lda, W, &lwork);
        out->AddMarshal<int>(lwork);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnCheevd_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnZheevd_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnZheevd_bufferSize"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    cusolverEigMode_t jobz = in->Get<cusolverEigMode_t>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    int n = in->Get<int>();
    cuDoubleComplex *A = in->Get<cuDoubleComplex*>();
    int lda = in->Get<int>();
    double *W = in->Get<double*>();
    int lwork;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnZheevd_bufferSize(handle, jobz, uplo, n, A, lda, W, &lwork);
        out->AddMarshal<int>(lwork);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnZheevd_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnSsyevd){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnSsyevd"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    cusolverEigMode_t jobz = in->Get<cusolverEigMode_t>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    int n = in->Get<int>();
    float *A = in->Get<float*>();
    int lda = in->Get<int>();
    float *W = in->Get<float*>();
    float *work = in->GetFromMarshal<float*>();
    int lwork = in->Get<int>();
    int *devInfo = in->GetFromMarshal<int*>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnSsyevd(handle, jobz, uplo, n, A, lda, W, work, lwork, devInfo);
        out->Add<float*>(W);
        out->Add<int*>(devInfo);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnSsyevd Executed");
    return std::make_shared<Result>(cs,out);
}

CUSOLVER_ROUTINE_HANDLER(DnDsyevd){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnDsyevd"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    cusolverEigMode_t jobz = in->Get<cusolverEigMode_t>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    int n = in->Get<int>();
    double *A = in->Get<double*>();
    int lda = in->Get<int>();
    double *W = in->Get<double*>();
    double *work = in->GetFromMarshal<double*>();
    int lwork = in->Get<int>();
    int *devInfo = in->GetFromMarshal<int*>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnDsyevd(handle, jobz, uplo, n, A, lda, W, work, lwork, devInfo);
        out->Add<double*>(W);
        out->Add<int*>(devInfo);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnDsyevd Executed");
    return std::make_shared<Result>(cs,out);
}

CUSOLVER_ROUTINE_HANDLER(DnCheevd){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnCheevd"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    cusolverEigMode_t jobz = in->Get<cusolverEigMode_t>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    int n = in->Get<int>();
    cuComplex *A = in->Get<cuComplex*>();
    int lda = in->Get<int>();
    float *W = in->Get<float*>();
    cuComplex *work = in->GetFromMarshal<cuComplex*>();
    int lwork = in->Get<int>();
    int *devInfo = in->GetFromMarshal<int*>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnCheevd(handle, jobz, uplo, n, A, lda, W, work, lwork, devInfo);
        out->Add<float*>(W);
        out->Add<int*>(devInfo);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnCheevd Executed");
    return std::make_shared<Result>(cs,out);
}

CUSOLVER_ROUTINE_HANDLER(DnZheevd){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnZheevd"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    cusolverEigMode_t jobz = in->Get<cusolverEigMode_t>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    int n = in->Get<int>();
    cuDoubleComplex *A = in->Get<cuDoubleComplex*>();
    int lda = in->Get<int>();
    double *W = in->Get<double*>();
    cuDoubleComplex *work = in->GetFromMarshal<cuDoubleComplex*>();
    int lwork = in->Get<int>();
    int *devInfo = in->GetFromMarshal<int*>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverDnZheevd(handle, jobz, uplo, n, A, lda, W, work, lwork, devInfo);
        out->Add<double*>(W);
        out->Add<int*>(devInfo);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnZheevd Executed");
    return std::make_shared<Result>(cs,out);
}