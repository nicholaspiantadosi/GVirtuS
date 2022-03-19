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

CUSOLVER_ROUTINE_HANDLER(SpXcsrsymrcmHost){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SpXcsrsymrcmHost"));
    CusolverHandler::setLogLevel(&logger);
    cusolverSpHandle_t handle = (cusolverSpHandle_t)in->Get<size_t>();
    int n = in->Get<int>();
    int nnzA = in->Get<int>();
    cusparseMatDescr_t descrA = (cusparseMatDescr_t)in->Get<size_t>();
    int *csrRowPtrA = in->Assign<int>(n + 1);
    int *csrColIndA = in->Assign<int>(nnzA);
    int *p = in->Get<int>(n);
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverSpXcsrsymrcmHost(handle, n, nnzA, descrA, csrRowPtrA, csrColIndA, p);
        out->Add<int>(p, n);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    } catch(const char *e) {
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverSpXcsrsymrcmHost Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(SpXcsrsymmdqHost){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SpXcsrsymmdqHost"));
    CusolverHandler::setLogLevel(&logger);
    cusolverSpHandle_t handle = (cusolverSpHandle_t)in->Get<size_t>();
    int n = in->Get<int>();
    int nnzA = in->Get<int>();
    cusparseMatDescr_t descrA = (cusparseMatDescr_t)in->Get<size_t>();
    int *csrRowPtrA = in->Assign<int>(n + 1);
    int *csrColIndA = in->Assign<int>(nnzA);
    int *p = in->Get<int>(n);
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverSpXcsrsymmdqHost(handle, n, nnzA, descrA, csrRowPtrA, csrColIndA, p);
        out->Add<int>(p, n);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    } catch(const char *e) {
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverSpXcsrsymmdqHost Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(SpXcsrsymamdHost){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SpXcsrsymamdHost"));
    CusolverHandler::setLogLevel(&logger);
    cusolverSpHandle_t handle = (cusolverSpHandle_t)in->Get<size_t>();
    int n = in->Get<int>();
    int nnzA = in->Get<int>();
    cusparseMatDescr_t descrA = (cusparseMatDescr_t)in->Get<size_t>();
    int *csrRowPtrA = in->Assign<int>(n + 1);
    int *csrColIndA = in->Assign<int>(nnzA);
    int *p = in->Get<int>(n);
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverSpXcsrsymamdHost(handle, n, nnzA, descrA, csrRowPtrA, csrColIndA, p);
        out->Add<int>(p, n);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    } catch(const char *e) {
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverSpXcsrsymamdHost Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(SpXcsrmetisndHost){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SpXcsrmetisndHost"));
    CusolverHandler::setLogLevel(&logger);
    cusolverSpHandle_t handle = (cusolverSpHandle_t)in->Get<size_t>();
    int n = in->Get<int>();
    int nnzA = in->Get<int>();
    cusparseMatDescr_t descrA = (cusparseMatDescr_t)in->Get<size_t>();
    int *csrRowPtrA = in->Assign<int>(n + 1);
    int *csrColIndA = in->Assign<int>(nnzA);
    const int64_t *options = in->Assign<int64_t>(n);
    int *p = in->Get<int>(n);
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverSpXcsrmetisndHost(handle, n, nnzA, descrA, csrRowPtrA, csrColIndA, options, p);
        out->Add<int>(p, n);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    } catch(const char *e) {
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverSpXcsrmetisndHost Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(SpScsrzfdHost){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SpScsrzfdHost"));
    CusolverHandler::setLogLevel(&logger);
    cusolverSpHandle_t handle = (cusolverSpHandle_t)in->Get<size_t>();
    int n = in->Get<int>();
    int nnzA = in->Get<int>();
    cusparseMatDescr_t descrA = (cusparseMatDescr_t)in->Get<size_t>();
    float *csrValA = in->Assign<float>(nnzA);
    int *csrRowPtrA = in->Assign<int>(n + 1);
    int *csrColIndA = in->Assign<int>(nnzA);
    int *P = in->Get<int>(n);
    int numnz;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverSpScsrzfdHost(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, P, &numnz);
        out->Add<int>(P, n);
        out->Add<int>(numnz);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    } catch(const char *e) {
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverSpScsrzfdHost Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(SpDcsrzfdHost){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SpDcsrzfdHost"));
    CusolverHandler::setLogLevel(&logger);
    cusolverSpHandle_t handle = (cusolverSpHandle_t)in->Get<size_t>();
    int n = in->Get<int>();
    int nnzA = in->Get<int>();
    cusparseMatDescr_t descrA = (cusparseMatDescr_t)in->Get<size_t>();
    double *csrValA = in->Assign<double>(nnzA);
    int *csrRowPtrA = in->Assign<int>(n + 1);
    int *csrColIndA = in->Assign<int>(nnzA);
    int *P = in->Get<int>(n);
    int numnz;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverSpDcsrzfdHost(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, P, &numnz);
        out->Add<int>(P, n);
        out->Add<int>(numnz);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    } catch(const char *e) {
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverSpDcsrzfdHost Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(SpCcsrzfdHost){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SpCcsrzfdHost"));
    CusolverHandler::setLogLevel(&logger);
    cusolverSpHandle_t handle = (cusolverSpHandle_t)in->Get<size_t>();
    int n = in->Get<int>();
    int nnzA = in->Get<int>();
    cusparseMatDescr_t descrA = (cusparseMatDescr_t)in->Get<size_t>();
    cuComplex *csrValA = in->Assign<cuComplex>(nnzA);
    int *csrRowPtrA = in->Assign<int>(n + 1);
    int *csrColIndA = in->Assign<int>(nnzA);
    int *P = in->Get<int>(n);
    int numnz;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverSpCcsrzfdHost(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, P, &numnz);
        out->Add<int>(P, n);
        out->Add<int>(numnz);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    } catch(const char *e) {
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverSpCcsrzfdHost Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(SpZcsrzfdHost){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SpZcsrzfdHost"));
    CusolverHandler::setLogLevel(&logger);
    cusolverSpHandle_t handle = (cusolverSpHandle_t)in->Get<size_t>();
    int n = in->Get<int>();
    int nnzA = in->Get<int>();
    cusparseMatDescr_t descrA = (cusparseMatDescr_t)in->Get<size_t>();
    cuDoubleComplex *csrValA = in->Assign<cuDoubleComplex>(nnzA);
    int *csrRowPtrA = in->Assign<int>(n + 1);
    int *csrColIndA = in->Assign<int>(nnzA);
    int *P = in->Get<int>(n);
    int numnz;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverSpZcsrzfdHost(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, P, &numnz);
        out->Add<int>(P, n);
        out->Add<int>(numnz);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    } catch(const char *e) {
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverSpZcsrzfdHost Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(SpXcsrperm_bufferSizeHost){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SpXcsrperm_bufferSizeHost"));
    CusolverHandler::setLogLevel(&logger);
    cusolverSpHandle_t handle = (cusolverSpHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    int nnzA = in->Get<int>();
    cusparseMatDescr_t descrA = (cusparseMatDescr_t)in->Get<size_t>();
    int *csrRowPtrA = in->Assign<int>(n + 1);
    int *csrColIndA = in->Assign<int>(nnzA);
    int *p = in->Get<int>(n);
    int *q = in->Get<int>(n);
    size_t bufferSizeInBytes;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverSpXcsrperm_bufferSizeHost(handle, m, n, nnzA, descrA, csrRowPtrA, csrColIndA, p, q, &bufferSizeInBytes);
        out->Add<size_t>(bufferSizeInBytes);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    } catch(const char *e) {
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverSpXcsrperm_bufferSizeHost Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(SpXcsrpermHost){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SpXcsrpermHost"));
    CusolverHandler::setLogLevel(&logger);
    cusolverSpHandle_t handle = (cusolverSpHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    int nnzA = in->Get<int>();
    cusparseMatDescr_t descrA = (cusparseMatDescr_t)in->Get<size_t>();
    int *csrRowPtrA = in->Assign<int>(n + 1);
    int *csrColIndA = in->Assign<int>(nnzA);
    int *p = in->Get<int>(n);
    int *q = in->Get<int>(n);
    int *map = in->Get<int>(nnzA);
    void * pBuffer = in->Assign<void*>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverSpXcsrpermHost(handle, m, n, nnzA, descrA, csrRowPtrA, csrColIndA, p, q, map, pBuffer);
        out->Add<int>(csrRowPtrA, m+1);
        out->Add<int>(csrColIndA, nnzA);
        out->Add<int>(map, nnzA);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    } catch(const char *e) {
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverSpXcsrpermHost Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(SpCreateCsrqrInfo){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SpCreateCsrqrInfo"));
    CusolverHandler::setLogLevel(&logger);
    csrqrInfo_t info;
    cusolverStatus_t cs = cusolverSpCreateCsrqrInfo(&info);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        out->Add<csrqrInfo_t>(info);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverSpCreateCsrqrInfo Executed");
    return std::make_shared<Result>(cs,out);
}

CUSOLVER_ROUTINE_HANDLER(SpDestroyCsrqrInfo){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SpDestroyCsrqrInfo"));
    csrqrInfo_t info = in->Get<csrqrInfo_t>();
    cusolverStatus_t cs = cusolverSpDestroyCsrqrInfo(info);
    LOG4CPLUS_DEBUG(logger,"cusolverSpDestroyCsrqrInfo Executed");
    return std::make_shared<Result>(cs);
}

CUSOLVER_ROUTINE_HANDLER(SpXcsrqrAnalysisBatched){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SpXcsrqrAnalysisBatched"));
    CusolverHandler::setLogLevel(&logger);
    cusolverSpHandle_t handle = (cusolverSpHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    int nnzA = in->Get<int>();
    cusparseMatDescr_t descrA = (cusparseMatDescr_t)in->Get<size_t>();
    int *csrRowPtrA = in->Assign<int>(m + 1);
    int *csrColIndA = in->Assign<int>(nnzA);
    csrqrInfo_t info = (csrqrInfo_t)in->Get<size_t>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverSpXcsrqrAnalysisBatched(handle, m, n, nnzA, descrA, csrRowPtrA, csrColIndA, info);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    } catch(const char *e) {
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverSpXcsrqrAnalysisBatched Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(SpScsrqrBufferInfoBatched){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SpScsrqrBufferInfoBatched"));
    CusolverHandler::setLogLevel(&logger);
    cusolverSpHandle_t handle = (cusolverSpHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    int nnzA = in->Get<int>();
    int batchSize = in->Get<int>();
    cusparseMatDescr_t descrA = (cusparseMatDescr_t)in->Get<size_t>();
    float *csrValA = in->Assign<float>(nnzA*batchSize);
    int *csrRowPtrA = in->Assign<int>(m + 1);
    int *csrColIndA = in->Assign<int>(nnzA);
    csrqrInfo_t info = in->Get<csrqrInfo_t>();
    size_t internalDataInBytes;
    size_t workspaceInBytes;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverSpScsrqrBufferInfoBatched(handle, m, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, batchSize, info, &internalDataInBytes, &workspaceInBytes);
        out->Add<size_t>(internalDataInBytes);
        out->Add<size_t>(workspaceInBytes);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    } catch(const char *e) {
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverSpScsrqrBufferInfoBatched Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(SpDcsrqrBufferInfoBatched){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SpDcsrqrBufferInfoBatched"));
    CusolverHandler::setLogLevel(&logger);
    cusolverSpHandle_t handle = (cusolverSpHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    int nnzA = in->Get<int>();
    int batchSize = in->Get<int>();
    cusparseMatDescr_t descrA = (cusparseMatDescr_t)in->Get<size_t>();
    double *csrValA = in->Assign<double>(nnzA*batchSize);
    int *csrRowPtrA = in->Assign<int>(m + 1);
    int *csrColIndA = in->Assign<int>(nnzA);
    csrqrInfo_t info = in->Get<csrqrInfo_t>();
    size_t internalDataInBytes;
    size_t workspaceInBytes;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverSpDcsrqrBufferInfoBatched(handle, m, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, batchSize, info, &internalDataInBytes, &workspaceInBytes);
        out->Add<size_t>(internalDataInBytes);
        out->Add<size_t>(workspaceInBytes);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    } catch(const char *e) {
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverSpDcsrqrBufferInfoBatched Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(SpCcsrqrBufferInfoBatched){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SpCcsrqrBufferInfoBatched"));
    CusolverHandler::setLogLevel(&logger);
    cusolverSpHandle_t handle = (cusolverSpHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    int nnzA = in->Get<int>();
    int batchSize = in->Get<int>();
    cusparseMatDescr_t descrA = (cusparseMatDescr_t)in->Get<size_t>();
    cuComplex *csrValA = in->Assign<cuComplex>(nnzA*batchSize);
    int *csrRowPtrA = in->Assign<int>(m + 1);
    int *csrColIndA = in->Assign<int>(nnzA);
    csrqrInfo_t info = in->Get<csrqrInfo_t>();
    size_t internalDataInBytes;
    size_t workspaceInBytes;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverSpCcsrqrBufferInfoBatched(handle, m, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, batchSize, info, &internalDataInBytes, &workspaceInBytes);
        out->Add<size_t>(internalDataInBytes);
        out->Add<size_t>(workspaceInBytes);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    } catch(const char *e) {
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverSpCcsrqrBufferInfoBatched Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(SpZcsrqrBufferInfoBatched){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SpZcsrqrBufferInfoBatched"));
    CusolverHandler::setLogLevel(&logger);
    cusolverSpHandle_t handle = (cusolverSpHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    int nnzA = in->Get<int>();
    int batchSize = in->Get<int>();
    cusparseMatDescr_t descrA = (cusparseMatDescr_t)in->Get<size_t>();
    cuDoubleComplex *csrValA = in->Assign<cuDoubleComplex>(nnzA*batchSize);
    int *csrRowPtrA = in->Assign<int>(m + 1);
    int *csrColIndA = in->Assign<int>(nnzA);
    csrqrInfo_t info = in->Get<csrqrInfo_t>();
    size_t internalDataInBytes;
    size_t workspaceInBytes;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverSpZcsrqrBufferInfoBatched(handle, m, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, batchSize, info, &internalDataInBytes, &workspaceInBytes);
        out->Add<size_t>(internalDataInBytes);
        out->Add<size_t>(workspaceInBytes);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    } catch(const char *e) {
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverSpZcsrqrBufferInfoBatched Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(SpScsrqrsvBatched){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SpScsrqrsvBatched"));
    CusolverHandler::setLogLevel(&logger);
    cusolverSpHandle_t handle = (cusolverSpHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    int nnzA = in->Get<int>();
    int batchSize = in->Get<int>();
    cusparseMatDescr_t descrA = (cusparseMatDescr_t)in->Get<size_t>();
    float *csrValA = in->Assign<float>(nnzA*batchSize);
    int *csrRowPtrA = in->Assign<int>(m + 1);
    int *csrColIndA = in->Assign<int>(nnzA);
    float *b = in->Assign<float>(m*batchSize);
    float *x = in->Assign<float>(m*batchSize);
    csrqrInfo_t info = in->Get<csrqrInfo_t>();
    void *pBuffer = in->Get<void*>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverSpScsrqrsvBatched(handle, m, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, b, x, batchSize, info, pBuffer);
        out->Add<float>(x, m*batchSize);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    } catch(const char *e) {
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverSpScsrqrsvBatched Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(SpDcsrqrsvBatched){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SpDcsrqrsvBatched"));
    CusolverHandler::setLogLevel(&logger);
    cusolverSpHandle_t handle = (cusolverSpHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    int nnzA = in->Get<int>();
    int batchSize = in->Get<int>();
    cusparseMatDescr_t descrA = (cusparseMatDescr_t)in->Get<size_t>();
    double *csrValA = in->Assign<double>(nnzA*batchSize);
    int *csrRowPtrA = in->Assign<int>(m + 1);
    int *csrColIndA = in->Assign<int>(nnzA);
    double *b = in->Assign<double>(m*batchSize);
    double *x = in->Assign<double>(m*batchSize);
    csrqrInfo_t info = in->Get<csrqrInfo_t>();
    void *pBuffer = in->Get<void*>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverSpDcsrqrsvBatched(handle, m, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, b, x, batchSize, info, pBuffer);
        out->Add<double>(x, m*batchSize);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    } catch(const char *e) {
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverSpDcsrqrsvBatched Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(SpCcsrqrsvBatched){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SpCcsrqrsvBatched"));
    CusolverHandler::setLogLevel(&logger);
    cusolverSpHandle_t handle = (cusolverSpHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    int nnzA = in->Get<int>();
    int batchSize = in->Get<int>();
    cusparseMatDescr_t descrA = (cusparseMatDescr_t)in->Get<size_t>();
    cuComplex *csrValA = in->Assign<cuComplex>(nnzA*batchSize);
    int *csrRowPtrA = in->Assign<int>(m + 1);
    int *csrColIndA = in->Assign<int>(nnzA);
    cuComplex *b = in->Assign<cuComplex>(m*batchSize);
    cuComplex *x = in->Assign<cuComplex>(m*batchSize);
    csrqrInfo_t info = in->Get<csrqrInfo_t>();
    void *pBuffer = in->Get<void*>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverSpCcsrqrsvBatched(handle, m, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, b, x, batchSize, info, pBuffer);
        out->Add<cuComplex>(x, m*batchSize);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    } catch(const char *e) {
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverSpCcsrqrsvBatched Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(SpZcsrqrsvBatched){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SpZcsrqrsvBatched"));
    CusolverHandler::setLogLevel(&logger);
    cusolverSpHandle_t handle = (cusolverSpHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    int nnzA = in->Get<int>();
    int batchSize = in->Get<int>();
    cusparseMatDescr_t descrA = (cusparseMatDescr_t)in->Get<size_t>();
    cuDoubleComplex *csrValA = in->Assign<cuDoubleComplex>(nnzA*batchSize);
    int *csrRowPtrA = in->Assign<int>(m + 1);
    int *csrColIndA = in->Assign<int>(nnzA);
    cuDoubleComplex *b = in->Assign<cuDoubleComplex>(m*batchSize);
    cuDoubleComplex *x = in->Assign<cuDoubleComplex>(m*batchSize);
    csrqrInfo_t info = in->Get<csrqrInfo_t>();
    void *pBuffer = in->Get<void*>();
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverSpZcsrqrsvBatched(handle, m, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, b, x, batchSize, info, pBuffer);
        out->Add<cuDoubleComplex>(x, m*batchSize);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    } catch(const char *e) {
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverSpZcsrqrsvBatched Executed");
    return std::make_shared<Result>(cs, out);
}