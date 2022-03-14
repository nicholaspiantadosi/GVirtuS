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

CUSOLVER_ROUTINE_HANDLER(SpCreate){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SpCreate"));
    CusolverHandler::setLogLevel(&logger);
    cusolverSpHandle_t handle;
    cusolverStatus_t cs = cusolverSpCreate(&handle);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        out->Add<cusolverSpHandle_t>(handle);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverSpCreate Executed");
    return std::make_shared<Result>(cs,out);
}

CUSOLVER_ROUTINE_HANDLER(SpDestroy){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SpDestroy"));
    cusolverSpHandle_t handle = (cusolverSpHandle_t)in->Get<size_t>();
    cusolverStatus_t cs = cusolverSpDestroy(handle);
    LOG4CPLUS_DEBUG(logger,"cusolverSpDestroy Executed");
    return std::make_shared<Result>(cs);
}

CUSOLVER_ROUTINE_HANDLER(SpSetStream){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SpSetStream"));
    cusolverSpHandle_t handle = (cusolverSpHandle_t)in->Get<size_t>();
    cudaStream_t streamId = (cudaStream_t) in->Get<size_t>();
    cusolverStatus_t cs = cusolverSpSetStream(handle,streamId);
    LOG4CPLUS_DEBUG(logger,"cusolverSpSetStream Executed");
    return std::make_shared<Result>(cs);
}

CUSOLVER_ROUTINE_HANDLER(SpXcsrissymHost){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SpXcsrissymHost"));
    CusolverHandler::setLogLevel(&logger);
    cusolverSpHandle_t handle = (cusolverSpHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int nnzA = in->Get<int>();
    const cusparseMatDescr_t descrA = (cusparseMatDescr_t)in->Get<size_t>();
    int *csrRowPtrA = in->Get<int*>();
    int *csrEndPtrA = in->Get<int*>();
    int *csrColIndA = in->Get<int*>();
    int issym = 0;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverSpXcsrissymHost(handle, m, nnzA, descrA, csrRowPtrA, csrEndPtrA, csrColIndA, &issym);
        out->Add<int>(issym);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverSpXcsrissymHost Executed");
    return std::make_shared<Result>(cs, out);
}