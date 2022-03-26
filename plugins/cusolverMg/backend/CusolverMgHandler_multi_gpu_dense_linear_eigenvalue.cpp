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

using namespace log4cplus;

using gvirtus::communicators::Buffer;
using gvirtus::communicators::Result;

CUSOLVERMG_ROUTINE_HANDLER(MgSyevd_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("MgSyevd_bufferSize"));
    CusolverMgHandler::setLogLevel(&logger);
    cusolverMgHandle_t handle = (cusolverMgHandle_t)in->Get<size_t>();
    cusolverEigMode_t jobz = in->Get<cusolverEigMode_t>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    int N = in->Get<int>();
    void **array_d_A = in->Assign<void*>(N * N);
    int IA = in->Get<int>();
    int JA = in->Get<int>();
    cudaLibMgMatrixDesc_t descrA = in->Get<cudaLibMgMatrixDesc_t>();
    void* W = in->Assign<size_t>(N);
    cudaDataType_t dataTypeW = in->Get<cudaDataType>();
    cudaDataType computeType = in->Get<cudaDataType>();
    int64_t lwork;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverMgSyevd_bufferSize(handle, jobz, uplo, N, array_d_A, IA, JA, descrA, W, dataTypeW, computeType, &lwork);
        out->Add<int64_t>(lwork);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverMgSyevd_bufferSize Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVERMG_ROUTINE_HANDLER(MgSyevd){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("MgSyevd"));
    CusolverMgHandler::setLogLevel(&logger);
    cusolverMgHandle_t handle = (cusolverMgHandle_t)in->Get<size_t>();
    cusolverEigMode_t jobz = in->Get<cusolverEigMode_t>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    int N = in->Get<int>();
    void **array_d_A = in->Assign<void*>(N * N);
    int IA = in->Get<int>();
    int JA = in->Get<int>();
    cudaLibMgMatrixDesc_t descrA = in->Get<cudaLibMgMatrixDesc_t>();
    void* W = in->Assign<size_t>(N);
    cudaDataType_t dataTypeW = in->Get<cudaDataType>();
    cudaDataType computeType = in->Get<cudaDataType>();
    int64_t lwork = in->Get<int64_t>();
    void **array_d_work = in->Assign<void*>(computeType * lwork);
    int info;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverMgSyevd(handle, jobz, uplo, N, array_d_A, IA, JA, descrA, W, dataTypeW, computeType, array_d_work, lwork, &info);
        out->Add<int>(info);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverMgSyevd Executed");
    return std::make_shared<Result>(cs, out);
}