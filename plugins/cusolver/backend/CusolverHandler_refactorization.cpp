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
#include "Utilities.h"

using namespace log4cplus;

using gvirtus::communicators::Buffer;
using gvirtus::communicators::Result;

CUSOLVER_ROUTINE_HANDLER(RfAccessBundledFactorsDevice){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("RfAccessBundledFactorsDevice"));
    CusolverHandler::setLogLevel(&logger);
    cusolverRfHandle_t handle = (cusolverRfHandle_t)in->Get<size_t>();
    int nnzM = in->Get<int>();
    int* Mp = new int;
    int* Mi = new int;
    double* Mx = new double;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverRfAccessBundledFactorsDevice(handle, &nnzM, &Mp, &Mi, &Mx);
        out->Add<int>(nnzM);
        out->Add<int*>(Mp);
        out->Add<int*>(Mi);
        out->Add<double*>(Mx);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverRfAccessBundledFactorsDevice Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(RfAnalyze){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("RfAnalyze"));
    CusolverHandler::setLogLevel(&logger);
    cusolverRfHandle_t handle = (cusolverRfHandle_t)in->Get<size_t>();
    cusolverStatus_t cs = cusolverRfAnalyze(handle);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    LOG4CPLUS_DEBUG(logger,"cusolverRfAnalyze Executed");
    return std::make_shared<Result>(cs,out);
}

CUSOLVER_ROUTINE_HANDLER(RfSetupDevice){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("RfSetupDevice"));
    CusolverHandler::setLogLevel(&logger);
    int n = in->Get<int>();
    int nnzA = in->Get<int>();
    int* csrRowPtrA = in->Get<int*>();
    int* csrColIndA = in->Get<int*>();
    double* csrValA = in->Get<double*>();
    int nnzL = in->Get<int>();
    int* csrRowPtrL = in->Get<int*>();
    int* csrColIndL = in->Get<int*>();
    double* csrValL = in->Get<double*>();
    int nnzU = in->Get<int>();
    int* csrRowPtrU = in->Get<int*>();
    int* csrColIndU = in->Get<int*>();
    double* csrValU = in->Get<double*>();
    int* P = in->Get<int*>();
    int* Q = in->Get<int*>();
    cusolverRfHandle_t handle = (cusolverRfHandle_t)in->Get<size_t>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverRfSetupDevice(n, nnzA, csrRowPtrA, csrColIndA, csrValA, nnzL, csrRowPtrL, csrColIndL, csrValL, nnzU, csrRowPtrU, csrColIndU, csrValU, P, Q, handle);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverRfSetupDevice Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(RfSetupHost){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("RfSetupHost"));
    CusolverHandler::setLogLevel(&logger);
    int n = in->Get<int>();
    int nnzA = in->Get<int>();
    int* csrRowPtrA = in->Assign<int>(n + 1);
    int* csrColIndA = in->Assign<int>(nnzA);
    double* csrValA = in->Assign<double>(nnzA);
    int nnzL = in->Get<int>();
    int* csrRowPtrL = in->Assign<int>(n + 1);
    int* csrColIndL = in->Assign<int>(nnzL);
    double* csrValL = in->Assign<double>(nnzL);
    int nnzU = in->Get<int>();
    int* csrRowPtrU = in->Assign<int>(n + 1);
    int* csrColIndU = in->Assign<int>(nnzU);
    double* csrValU = in->Assign<double>(nnzU);
    int* P = in->Assign<int>(n);
    int* Q = in->Assign<int>(n);
    cusolverRfHandle_t handle = (cusolverRfHandle_t)in->Get<size_t>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverRfSetupHost(n, nnzA, csrRowPtrA, csrColIndA, csrValA, nnzL, csrRowPtrL, csrColIndL, csrValL, nnzU, csrRowPtrU, csrColIndU, csrValU, P, Q, handle);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverRfSetupHost Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(RfCreate){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("RfCreate"));
    CusolverHandler::setLogLevel(&logger);
    cusolverRfHandle_t handle;
    cusolverStatus_t cs = cusolverRfCreate(&handle);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        out->Add<cusolverRfHandle_t>(handle);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverRfCreate Executed");
    return std::make_shared<Result>(cs,out);
}

CUSOLVER_ROUTINE_HANDLER(RfExtractBundledFactorsHost){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("RfExtractBundledFactorsHost"));
    CusolverHandler::setLogLevel(&logger);
    cusolverRfHandle_t handle = (cusolverRfHandle_t)in->Get<size_t>();
    int* nnzM = new int;
    int** Mp = new int*;
    int** Mi = new int*;
    double** Mx = new double*;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverRfExtractBundledFactorsHost(handle, nnzM, Mp, Mi, Mx);
        out->Add<int>(*nnzM);
        out->Add<int>(*Mp, *nnzM);
        out->Add<int>(*Mi, *nnzM);
        out->Add<double>(*Mx, *nnzM);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverRfExtractBundledFactorsHost Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(RfExtractSplitFactorsHost){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("RfExtractSplitFactorsHost"));
    CusolverHandler::setLogLevel(&logger);
    cusolverRfHandle_t handle = (cusolverRfHandle_t)in->Get<size_t>();
    int* nnzL = new int;
    int** Lp = new int*;
    int** Li = new int*;
    double** Lx = new double*;
    int* nnzU = new int;
    int** Up = new int*;
    int** Ui = new int*;
    double** Ux = new double*;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverRfExtractSplitFactorsHost(handle, nnzL, Lp, Li, Lx, nnzU, Up, Ui, Ux);
        out->Add<int>(*nnzL);
        out->Add<int>(*Lp, *nnzL);
        out->Add<int>(*Li, *nnzL);
        out->Add<double>(*Lx, *nnzL);
        out->Add<int>(*nnzU);
        out->Add<int>(*Up, *nnzU);
        out->Add<int>(*Ui, *nnzU);
        out->Add<double>(*Ux, *nnzU);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverRfExtractSplitFactorsHost Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(RfDestroy){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("RfDestroy"));
    cusolverRfHandle_t handle = (cusolverRfHandle_t)in->Get<size_t>();
    cusolverStatus_t cs = cusolverRfDestroy(handle);
    LOG4CPLUS_DEBUG(logger,"cusolverRfDestroy Executed");
    return std::make_shared<Result>(cs);
}

CUSOLVER_ROUTINE_HANDLER(RfRefactor){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("RfRefactor"));
    CusolverHandler::setLogLevel(&logger);
    cusolverRfHandle_t handle = (cusolverRfHandle_t)in->Get<size_t>();
    cusolverStatus_t cs = cusolverRfRefactor(handle);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    LOG4CPLUS_DEBUG(logger,"cusolverRfRefactor Executed");
    return std::make_shared<Result>(cs,out);
}