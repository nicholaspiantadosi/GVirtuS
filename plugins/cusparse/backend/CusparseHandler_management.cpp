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

#include "CusparseHandler.h"

using namespace log4cplus;

using gvirtus::communicators::Buffer;
using gvirtus::communicators::Result;

CUSPARSE_ROUTINE_HANDLER(GetVersion){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetVersion"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle;
    cusparseStatus_t cs = cusparseCreate(&handle);
    int version;
    cusparseGetVersion(handle, &version);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        out->Add<int>(version);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseGetVersion Executed");
    return std::make_shared<Result>(version);
}

CUSPARSE_ROUTINE_HANDLER(Create){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Create"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle;
    cusparseStatus_t cs = cusparseCreate(&handle);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        out->Add<cusparseHandle_t>(handle);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseCreate Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Destroy){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Destroy"));
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<long long int>();
    cusparseStatus_t cs = cusparseDestroy(handle);
    LOG4CPLUS_DEBUG(logger,"cusparseDestroy Executed");
    return std::make_shared<Result>(cs);
}

CUSPARSE_ROUTINE_HANDLER(GetErrorString){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetErrorString"));
    cusparseStatus_t cs = in->Get<cusparseStatus_t>();
    const char * s = cusparseGetErrorString(cs);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        out->AddString(s);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseGetErrorString Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(SetStream){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetStream"));
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<long long int>();
    cudaStream_t streamId = (cudaStream_t) in->Get<long long int>();
    cusparseStatus_t cs = cusparseSetStream(handle,streamId);
    LOG4CPLUS_DEBUG(logger,"cusparseSetStream Executed");
    return std::make_shared<Result>(cs);
}

CUSPARSE_ROUTINE_HANDLER(GetStream){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetStream"));
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<long long int>();
    cudaStream_t streamId;
    cusparseStatus_t cs = cusparseGetStream(handle, &streamId);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
         out->Add<long long int>((long long int)streamId);
    } catch (string e){
         LOG4CPLUS_DEBUG(logger,e);
         return std::make_shared<Result>(cs);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseGetStream Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(GetProperty){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetProperty"));
    libraryPropertyType_t type = (libraryPropertyType_t)in->Get<libraryPropertyType_t>();
    int value;
    cusparseStatus_t cs = cusparseGetProperty(type, &value);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->Add<int>(value);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(cs);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseGetProperty Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(GetPointerMode){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetPointerMode"));
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<long long int>();
    cusparsePointerMode_t mode;
    cusparseStatus_t cs = cusparseGetPointerMode(handle, &mode);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->Add<cusparsePointerMode_t>(mode);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(cs);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseGetPointerMode Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(SetPointerMode){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetPointerMode"));
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<long long int>();
    cusparsePointerMode_t mode = (cusparsePointerMode_t)in->Get<cusparsePointerMode_t>();
    cusparseStatus_t cs = cusparseSetPointerMode(handle, mode);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    LOG4CPLUS_DEBUG(logger,"cusparseSetPointerMode Executed");
    return std::make_shared<Result>(cs,out);
}

#ifndef CUSPARSE_VERSION
#error CUSPARSE_VERSION not defined
#endif
