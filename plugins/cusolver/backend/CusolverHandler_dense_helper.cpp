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
#include <cuda_runtime.h>

using namespace log4cplus;

using gvirtus::communicators::Buffer;
using gvirtus::communicators::Result;

CUSOLVER_ROUTINE_HANDLER(DnCreate){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnCreate"));
    CusolverHandler::setLogLevel(&logger);
    cusolverDnHandle_t handle;
    cusolverStatus_t cs = cusolverDnCreate(&handle);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        out->Add<cusolverDnHandle_t>(handle);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnCreate Executed");
    return std::make_shared<Result>(cs,out);
}

CUSOLVER_ROUTINE_HANDLER(DnDestroy){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnDestroy"));
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    cusolverStatus_t cs = cusolverDnDestroy(handle);
    LOG4CPLUS_DEBUG(logger,"cusolverDnDestroy Executed");
    return std::make_shared<Result>(cs);
}

CUSOLVER_ROUTINE_HANDLER(DnSetStream){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnSetStream"));
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    cudaStream_t streamId = (cudaStream_t) in->Get<size_t>();
    cusolverStatus_t cs = cusolverDnSetStream(handle,streamId);
    LOG4CPLUS_DEBUG(logger,"cusolverDnSetStream Executed");
    return std::make_shared<Result>(cs);
}

CUSOLVER_ROUTINE_HANDLER(DnGetStream){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnGetStream"));
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    cudaStream_t streamId;
    cusolverStatus_t cs = cusolverDnGetStream(handle, &streamId);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->Add<size_t>((size_t)streamId);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(cs);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnGetStream Executed");
    return std::make_shared<Result>(cs,out);
}