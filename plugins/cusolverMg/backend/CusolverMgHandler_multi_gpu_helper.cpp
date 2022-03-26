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

#include "CusolverMgHandler.h"
#include "Utilities.h"

using namespace log4cplus;

using gvirtus::communicators::Buffer;
using gvirtus::communicators::Result;

CUSOLVERMG_ROUTINE_HANDLER(MgCreate){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("MgCreate"));
    CusolverMgHandler::setLogLevel(&logger);
    cusolverMgHandle_t handle;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverMgCreate(&handle);
        out->Add<cusolverMgHandle_t>(handle);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverMgCreate Executed");
    return std::make_shared<Result>(cs,out);
}

CUSOLVERMG_ROUTINE_HANDLER(MgDestroy){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("MgDestroy"));
    CusolverMgHandler::setLogLevel(&logger);
    cusolverMgHandle_t handle = (cusolverMgHandle_t)in->Get<size_t>();
    cusolverStatus_t cs;
    try{
        cs = cusolverMgDestroy(handle);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverMgDestroy Executed");
    return std::make_shared<Result>(cs);
}

CUSOLVERMG_ROUTINE_HANDLER(MgDeviceSelect){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("MgDeviceSelect"));
    CusolverMgHandler::setLogLevel(&logger);
    cusolverMgHandle_t handle = (cusolverMgHandle_t)in->Get<size_t>();
    int nbDevices = in->Get<int>();
    int* deviceId = in->Get<int>(nbDevices);
    cusolverStatus_t cs;
    try{
         cs = cusolverMgDeviceSelect(handle, nbDevices, deviceId);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverMgDeviceSelect Executed");
    return std::make_shared<Result>(cs);
}

CUSOLVERMG_ROUTINE_HANDLER(MgCreateDeviceGrid){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("MgCreateDeviceGrid"));
    CusolverMgHandler::setLogLevel(&logger);
    cudaLibMgGrid_t grid;
    int32_t numRowDevices = in->Get<int32_t>();
    int32_t numColDevices = in->Get<int32_t>();
    int32_t* deviceId = in->Get<int32_t>(numColDevices);
    cusolverMgGridMapping_t mapping = in->Get<cusolverMgGridMapping_t>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverMgCreateDeviceGrid(&grid, numRowDevices, numColDevices, deviceId, mapping);
        out->Add<cudaLibMgGrid_t>(grid);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"MgCreateDeviceGrid Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVERMG_ROUTINE_HANDLER(MgDestroyGrid){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("MgDestroyGrid"));
    CusolverMgHandler::setLogLevel(&logger);
    cudaLibMgGrid_t grid = (cudaLibMgGrid_t)in->Get<size_t>();
    cusolverStatus_t cs;
    try{
        cs = cusolverMgDestroyGrid(grid);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverMgDestroyGrid Executed");
    return std::make_shared<Result>(cs);
}

CUSOLVERMG_ROUTINE_HANDLER(MgCreateMatrixDesc){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("MgCreateMatrixDesc"));
    CusolverMgHandler::setLogLevel(&logger);
    cudaLibMgMatrixDesc_t desc;
    int64_t numRows = in->Get<int64_t>();
    int64_t numCols = in->Get<int64_t>();
    int64_t rowBlockSize = in->Get<int64_t>();
    int64_t colBlockSize = in->Get<int64_t>();
    cudaDataType_t dataType = in->Get<cudaDataType_t>();
    cudaLibMgGrid_t grid = (cudaLibMgGrid_t) in->Get<size_t>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverMgCreateMatrixDesc(&desc, numRows, numCols, rowBlockSize, colBlockSize, dataType, grid);
        out->Add<cudaLibMgMatrixDesc_t>(desc);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"MgCreateMatrixDesc Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVERMG_ROUTINE_HANDLER(MgDestroyMatrixDesc){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("MgDestroyMatrixDesc"));
    CusolverMgHandler::setLogLevel(&logger);
    cudaLibMgMatrixDesc_t desc = (cudaLibMgMatrixDesc_t)in->Get<size_t>();
    cusolverStatus_t cs;
    try{
        cs = cusolverMgDestroyMatrixDesc(desc);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverMgDestroyMatrixDesc Executed");
    return std::make_shared<Result>(cs);
}