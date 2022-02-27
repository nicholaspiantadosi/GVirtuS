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

CUSOLVER_ROUTINE_HANDLER(DnCreateSyevjInfo){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnCreateSyevjInfo"));
    CusolverHandler::setLogLevel(&logger);
    syevjInfo_t info;
    cusolverStatus_t cs = cusolverDnCreateSyevjInfo(&info);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        out->Add<syevjInfo_t>(info);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnCreateSyevjInfo Executed");
    return std::make_shared<Result>(cs,out);
}

CUSOLVER_ROUTINE_HANDLER(DnDestroySyevjInfo){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnDestroySyevjInfo"));
    syevjInfo_t info = (syevjInfo_t)in->Get<size_t>();
    cusolverStatus_t cs = cusolverDnDestroySyevjInfo(info);
    LOG4CPLUS_DEBUG(logger,"cusolverDnDestroySyevjInfo Executed");
    return std::make_shared<Result>(cs);
}

CUSOLVER_ROUTINE_HANDLER(DnXsyevjSetTolerance){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnXsyevjSetTolerance"));
    syevjInfo_t info = (syevjInfo_t)in->Get<size_t>();
    double tolerance = in->Get<double>();
    cusolverStatus_t cs = cusolverDnXsyevjSetTolerance(info, tolerance);
    LOG4CPLUS_DEBUG(logger,"cusolverDnXsyevjSetTolerance Executed");
    return std::make_shared<Result>(cs);
}

CUSOLVER_ROUTINE_HANDLER(DnXsyevjSetMaxSweeps){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnXsyevjSetMaxSweeps"));
    syevjInfo_t info = (syevjInfo_t)in->Get<size_t>();
    int max_sweeps = in->Get<int>();
    cusolverStatus_t cs = cusolverDnXsyevjSetMaxSweeps(info, max_sweeps);
    LOG4CPLUS_DEBUG(logger,"cusolverDnXsyevjSetMaxSweeps Executed");
    return std::make_shared<Result>(cs);
}

CUSOLVER_ROUTINE_HANDLER(DnXsyevjSetSortEig){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnXsyevjSetSortEig"));
    syevjInfo_t info = (syevjInfo_t)in->Get<size_t>();
    int sort_eig = in->Get<int>();
    cusolverStatus_t cs = cusolverDnXsyevjSetSortEig(info, sort_eig);
    LOG4CPLUS_DEBUG(logger,"cusolverDnXsyevjSetSortEig Executed");
    return std::make_shared<Result>(cs);
}

CUSOLVER_ROUTINE_HANDLER(DnXsyevjGetResidual){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnXsyevjGetResidual"));
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    syevjInfo_t info = (syevjInfo_t)in->Get<size_t>();
    double residual;
    cusolverStatus_t cs = cusolverDnXsyevjGetResidual(handle, info, &residual);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->Add<double>(residual);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(cs);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnXsyevjGetResidual Executed");
    return std::make_shared<Result>(cs,out);
}

CUSOLVER_ROUTINE_HANDLER(DnXsyevjGetSweeps){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnXsyevjGetSweeps"));
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    syevjInfo_t info = (syevjInfo_t)in->Get<size_t>();
    int executed_sweeps;
    cusolverStatus_t cs = cusolverDnXsyevjGetSweeps(handle, info, &executed_sweeps);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->Add<int>(executed_sweeps);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(cs);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnXsyevjGetSweeps Executed");
    return std::make_shared<Result>(cs,out);
}

CUSOLVER_ROUTINE_HANDLER(DnCreateGesvdjInfo){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnCreateGesvdjInfo"));
    CusolverHandler::setLogLevel(&logger);
    gesvdjInfo_t info;
    cusolverStatus_t cs = cusolverDnCreateGesvdjInfo(&info);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        out->Add<gesvdjInfo_t>(info);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnCreateGesvdjInfo Executed");
    return std::make_shared<Result>(cs,out);
}

CUSOLVER_ROUTINE_HANDLER(DnDestroyGesvdjInfo){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnDestroyGesvdjInfo"));
    gesvdjInfo_t info = (gesvdjInfo_t)in->Get<size_t>();
    cusolverStatus_t cs = cusolverDnDestroyGesvdjInfo(info);
    LOG4CPLUS_DEBUG(logger,"cusolverDnDestroyGesvdjInfo Executed");
    return std::make_shared<Result>(cs);
}

CUSOLVER_ROUTINE_HANDLER(DnXgesvdjSetTolerance){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnXgesvdjSetTolerance"));
    gesvdjInfo_t info = (gesvdjInfo_t)in->Get<size_t>();
    double tolerance = in->Get<double>();
    cusolverStatus_t cs = cusolverDnXgesvdjSetTolerance(info, tolerance);
    LOG4CPLUS_DEBUG(logger,"cusolverDnXgesvdjSetTolerance Executed");
    return std::make_shared<Result>(cs);
}

CUSOLVER_ROUTINE_HANDLER(DnXgesvdjSetMaxSweeps){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnXgesvdjSetMaxSweeps"));
    gesvdjInfo_t info = (gesvdjInfo_t)in->Get<size_t>();
    int max_sweeps = in->Get<int>();
    cusolverStatus_t cs = cusolverDnXgesvdjSetMaxSweeps(info, max_sweeps);
    LOG4CPLUS_DEBUG(logger,"cusolverDnXgesvdjSetMaxSweeps Executed");
    return std::make_shared<Result>(cs);
}

    CUSOLVER_ROUTINE_HANDLER(DnXgesvdjSetSortEig){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnXgesvdjSetSortEig"));
    gesvdjInfo_t info = (gesvdjInfo_t)in->Get<size_t>();
    int sort_eig = in->Get<int>();
    cusolverStatus_t cs = cusolverDnXgesvdjSetSortEig(info, sort_eig);
    LOG4CPLUS_DEBUG(logger,"cusolverDnXgesvdjSetSortEig Executed");
    return std::make_shared<Result>(cs);
}

CUSOLVER_ROUTINE_HANDLER(DnXgesvdjGetResidual){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnXgesvdjGetResidual"));
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    gesvdjInfo_t info = (gesvdjInfo_t)in->Get<size_t>();
    double residual;
    cusolverStatus_t cs = cusolverDnXgesvdjGetResidual(handle, info, &residual);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->Add<double>(residual);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(cs);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnXgesvdjGetResidual Executed");
    return std::make_shared<Result>(cs,out);
}

CUSOLVER_ROUTINE_HANDLER(DnXgesvdjGetSweeps){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnXgesvdjGetSweeps"));
    cusolverDnHandle_t handle = (cusolverDnHandle_t)in->Get<size_t>();
    gesvdjInfo_t info = (gesvdjInfo_t)in->Get<size_t>();
    int executed_sweeps;
    cusolverStatus_t cs = cusolverDnXgesvdjGetSweeps(handle, info, &executed_sweeps);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->Add<int>(executed_sweeps);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(cs);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverDnXgesvdjGetSweeps Executed");
    return std::make_shared<Result>(cs,out);
}