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

CUSOLVER_ROUTINE_HANDLER(SpScsrlsvluHost){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SpScsrlsvluHost"));
    CusolverHandler::setLogLevel(&logger);
    cusolverSpHandle_t handle = (cusolverSpHandle_t)in->Get<size_t>();
    int n = in->Get<int>();
    int nnzA = in->Get<int>();
    cusparseMatDescr_t descrA = (cusparseMatDescr_t)in->Get<size_t>();
    float *csrValA = in->Assign<float>(nnzA);
    int *csrRowPtrA = in->Assign<int>(n + 1);
    int *csrColIndA = in->Assign<int>(nnzA);
    float *b = in->Assign<float>(n);
    float tol = in->Get<float>();
    int reorder = in->Get<int>();
    float *x = in->Get<float>(n);
    int singularity;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverSpScsrlsvluHost(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol, reorder, x, &singularity);
        out->Add<float>(x, n);
        out->Add<int>(singularity);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    } catch(const char *e) {
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverSpScsrlsvluHost Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(SpDcsrlsvluHost){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SpDcsrlsvluHost"));
    CusolverHandler::setLogLevel(&logger);
    cusolverSpHandle_t handle = (cusolverSpHandle_t)in->Get<size_t>();
    int n = in->Get<int>();
    int nnzA = in->Get<int>();
    cusparseMatDescr_t descrA = (cusparseMatDescr_t)in->Get<size_t>();
    double *csrValA = in->Assign<double>(nnzA);
    int *csrRowPtrA = in->Assign<int>(n + 1);
    int *csrColIndA = in->Assign<int>(nnzA);
    double *b = in->Assign<double>(n);
    double tol = in->Get<double>();
    int reorder = in->Get<int>();
    double *x = in->Assign<double>(n);
    int singularity;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverSpDcsrlsvluHost(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol, reorder, x, &singularity);
        out->Add<double>(x, n);
        out->Add<int>(singularity);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    } catch(const char *e) {
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverSpDcsrlsvluHost Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(SpCcsrlsvluHost){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SpCcsrlsvluHost"));
    CusolverHandler::setLogLevel(&logger);
    cusolverSpHandle_t handle = (cusolverSpHandle_t)in->Get<size_t>();
    int n = in->Get<int>();
    int nnzA = in->Get<int>();
    cusparseMatDescr_t descrA = (cusparseMatDescr_t)in->Get<size_t>();
    cuComplex *csrValA = in->Assign<cuComplex>(nnzA);
    int *csrRowPtrA = in->Assign<int>(n + 1);
    int *csrColIndA = in->Assign<int>(nnzA);
    cuComplex *b = in->Assign<cuComplex>(n);
    float tol = in->Get<float>();
    int reorder = in->Get<int>();
    cuComplex *x = in->Assign<cuComplex>(n);
    int singularity;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverSpCcsrlsvluHost(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol, reorder, x, &singularity);
        out->Add<cuComplex>(x, n);
        out->Add<int>(singularity);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    } catch(const char *e) {
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverSpCcsrlsvluHost Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(SpZcsrlsvluHost){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SpZcsrlsvluHost"));
    CusolverHandler::setLogLevel(&logger);
    cusolverSpHandle_t handle = (cusolverSpHandle_t)in->Get<size_t>();
    int n = in->Get<int>();
    int nnzA = in->Get<int>();
    cusparseMatDescr_t descrA = (cusparseMatDescr_t)in->Get<size_t>();
    cuDoubleComplex *csrValA = in->Assign<cuDoubleComplex>(nnzA);
    int *csrRowPtrA = in->Assign<int>(n + 1);
    int *csrColIndA = in->Assign<int>(nnzA);
    cuDoubleComplex *b = in->Assign<cuDoubleComplex>(n);
    double tol = in->Get<double>();
    int reorder = in->Get<int>();
    cuDoubleComplex *x = in->Assign<cuDoubleComplex>(n);
    int singularity;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverSpZcsrlsvluHost(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol, reorder, x, &singularity);
        out->Add<cuDoubleComplex>(x, n);
        out->Add<int>(singularity);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    } catch(const char *e) {
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverSpZcsrlsvluHost Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(SpScsrlsvqrHost){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SpScsrlsvqrHost"));
    CusolverHandler::setLogLevel(&logger);
    cusolverSpHandle_t handle = (cusolverSpHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int nnz = in->Get<int>();
    cusparseMatDescr_t descrA = (cusparseMatDescr_t)in->Get<size_t>();
    float *csrValA = in->Assign<float>(nnz);
    int *csrRowPtrA = in->Assign<int>(m + 1);
    int *csrColIndA = in->Assign<int>(nnz);
    float *b = in->Assign<float>(m);
    float tol = in->Get<float>();
    int reorder = in->Get<int>();
    float *x = in->Get<float>(m);
    int singularity;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverSpScsrlsvqrHost(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol, reorder, x, &singularity);
        out->Add<float>(x, m);
        out->Add<int>(singularity);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    } catch(const char *e) {
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverSpScsrlsvqrHost Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(SpDcsrlsvqrHost){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SpDcsrlsvqrHost"));
    CusolverHandler::setLogLevel(&logger);
    cusolverSpHandle_t handle = (cusolverSpHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int nnz = in->Get<int>();
    cusparseMatDescr_t descrA = (cusparseMatDescr_t)in->Get<size_t>();
    double *csrValA = in->Assign<double>(nnz);
    int *csrRowPtrA = in->Assign<int>(m + 1);
    int *csrColIndA = in->Assign<int>(nnz);
    double *b = in->Assign<double>(m);
    double tol = in->Get<double>();
    int reorder = in->Get<int>();
    double *x = in->Get<double>(m);
    int singularity;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverSpDcsrlsvqrHost(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol, reorder, x, &singularity);
        out->Add<double>(x, m);
        out->Add<int>(singularity);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    } catch(const char *e) {
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverSpDcsrlsvqrHost Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(SpCcsrlsvqrHost){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SpCcsrlsvqrHost"));
    CusolverHandler::setLogLevel(&logger);
    cusolverSpHandle_t handle = (cusolverSpHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int nnz = in->Get<int>();
    cusparseMatDescr_t descrA = (cusparseMatDescr_t)in->Get<size_t>();
    cuComplex *csrValA = in->Assign<cuComplex>(nnz);
    int *csrRowPtrA = in->Assign<int>(m + 1);
    int *csrColIndA = in->Assign<int>(nnz);
    cuComplex *b = in->Assign<cuComplex>(m);
    float tol = in->Get<float>();
    int reorder = in->Get<int>();
    cuComplex *x = in->Get<cuComplex>(m);
    int singularity;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverSpCcsrlsvqrHost(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol, reorder, x, &singularity);
        out->Add<cuComplex>(x, m);
        out->Add<int>(singularity);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    } catch(const char *e) {
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverSpCcsrlsvqrHost Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(SpZcsrlsvqrHost){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SpZcsrlsvqrHost"));
    CusolverHandler::setLogLevel(&logger);
    cusolverSpHandle_t handle = (cusolverSpHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int nnz = in->Get<int>();
    cusparseMatDescr_t descrA = (cusparseMatDescr_t)in->Get<size_t>();
    cuDoubleComplex *csrValA = in->Assign<cuDoubleComplex>(nnz);
    int *csrRowPtrA = in->Assign<int>(m + 1);
    int *csrColIndA = in->Assign<int>(nnz);
    cuDoubleComplex *b = in->Assign<cuDoubleComplex>(m);
    double tol = in->Get<double>();
    int reorder = in->Get<int>();
    cuDoubleComplex *x = in->Get<cuDoubleComplex>(m);
    int singularity;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverSpZcsrlsvqrHost(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol, reorder, x, &singularity);
        out->Add<cuDoubleComplex>(x, m);
        out->Add<int>(singularity);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    } catch(const char *e) {
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverSpZcsrlsvqrHost Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(SpScsrlsvcholHost){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SpScsrlsvcholHost"));
    CusolverHandler::setLogLevel(&logger);
    cusolverSpHandle_t handle = (cusolverSpHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int nnz = in->Get<int>();
    cusparseMatDescr_t descrA = (cusparseMatDescr_t)in->Get<size_t>();
    float *csrValA = in->Assign<float>(nnz);
    int *csrRowPtrA = in->Assign<int>(m + 1);
    int *csrColIndA = in->Assign<int>(nnz);
    float *b = in->Assign<float>(m);
    float tol = in->Get<float>();
    int reorder = in->Get<int>();
    float *x = in->Get<float>(m);
    int singularity;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverSpScsrlsvcholHost(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol, reorder, x, &singularity);
        out->Add<float>(x, m);
        out->Add<int>(singularity);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    } catch(const char *e) {
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverSpScsrlsvcholHost Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(SpDcsrlsvcholHost){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SpDcsrlsvcholHost"));
    CusolverHandler::setLogLevel(&logger);
    cusolverSpHandle_t handle = (cusolverSpHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int nnz = in->Get<int>();
    cusparseMatDescr_t descrA = (cusparseMatDescr_t)in->Get<size_t>();
    double *csrValA = in->Assign<double>(nnz);
    int *csrRowPtrA = in->Assign<int>(m + 1);
    int *csrColIndA = in->Assign<int>(nnz);
    double *b = in->Assign<double>(m);
    double tol = in->Get<double>();
    int reorder = in->Get<int>();
    double *x = in->Get<double>(m);
    int singularity;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverSpDcsrlsvcholHost(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol, reorder, x, &singularity);
        out->Add<double>(x, m);
        out->Add<int>(singularity);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    } catch(const char *e) {
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverSpDcsrlsvcholHost Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(SpCcsrlsvcholHost){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SpCcsrlsvcholHost"));
    CusolverHandler::setLogLevel(&logger);
    cusolverSpHandle_t handle = (cusolverSpHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int nnz = in->Get<int>();
    cusparseMatDescr_t descrA = (cusparseMatDescr_t)in->Get<size_t>();
    cuComplex *csrValA = in->Assign<cuComplex>(nnz);
    int *csrRowPtrA = in->Assign<int>(m + 1);
    int *csrColIndA = in->Assign<int>(nnz);
    cuComplex *b = in->Assign<cuComplex>(m);
    float tol = in->Get<float>();
    int reorder = in->Get<int>();
    cuComplex *x = in->Get<cuComplex>(m);
    int singularity;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverSpCcsrlsvcholHost(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol, reorder, x, &singularity);
        out->Add<cuComplex>(x, m);
        out->Add<int>(singularity);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    } catch(const char *e) {
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverSpCcsrlsvcholHost Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(SpZcsrlsvcholHost){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SpZcsrlsvcholHost"));
    CusolverHandler::setLogLevel(&logger);
    cusolverSpHandle_t handle = (cusolverSpHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int nnz = in->Get<int>();
    cusparseMatDescr_t descrA = (cusparseMatDescr_t)in->Get<size_t>();
    cuDoubleComplex *csrValA = in->Assign<cuDoubleComplex>(nnz);
    int *csrRowPtrA = in->Assign<int>(m + 1);
    int *csrColIndA = in->Assign<int>(nnz);
    cuDoubleComplex *b = in->Assign<cuDoubleComplex>(m);
    double tol = in->Get<double>();
    int reorder = in->Get<int>();
    cuDoubleComplex *x = in->Get<cuDoubleComplex>(m);
    int singularity;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverSpZcsrlsvcholHost(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol, reorder, x, &singularity);
        out->Add<cuDoubleComplex>(x, m);
        out->Add<int>(singularity);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    } catch(const char *e) {
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverSpZcsrlsvcholHost Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(SpScsrlsqvqrHost){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SpScsrlsqvqrHost"));
    CusolverHandler::setLogLevel(&logger);
    cusolverSpHandle_t handle = (cusolverSpHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    int nnz = in->Get<int>();
    cusparseMatDescr_t descrA = (cusparseMatDescr_t)in->Get<size_t>();
    float *csrValA = in->Assign<float>(nnz);
    int *csrRowPtrA = in->Assign<int>(m + 1);
    int *csrColIndA = in->Assign<int>(nnz);
    float *b = in->Assign<float>(m);
    float tol = in->Get<float>();
    int rankA;
    float *x = in->Get<float>(n);
    int *p = in->Get<int>(n);
    float min_norm;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverSpScsrlsqvqrHost(handle, m, n, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol, &rankA, x, p, &min_norm);
        out->Add<int>(rankA);
        out->Add<float>(x, n);
        out->Add<int>(p, n);
        out->Add<float>(min_norm);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    } catch(const char *e) {
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverSpScsrlsqvqrHost Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(SpDcsrlsqvqrHost){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SpDcsrlsqvqrHost"));
    CusolverHandler::setLogLevel(&logger);
    cusolverSpHandle_t handle = (cusolverSpHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    int nnz = in->Get<int>();
    cusparseMatDescr_t descrA = (cusparseMatDescr_t)in->Get<size_t>();
    double *csrValA = in->Assign<double>(nnz);
    int *csrRowPtrA = in->Assign<int>(m + 1);
    int *csrColIndA = in->Assign<int>(nnz);
    double *b = in->Assign<double>(m);
    double tol = in->Get<double>();
    int rankA;
    double *x = in->Get<double>(n);
    int *p = in->Get<int>(n);
    double min_norm;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverSpDcsrlsqvqrHost(handle, m, n, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol, &rankA, x, p, &min_norm);
        out->Add<int>(rankA);
        out->Add<double>(x, n);
        out->Add<int>(p, n);
        out->Add<double>(min_norm);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    } catch(const char *e) {
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverSpDcsrlsqvqrHost Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(SpCcsrlsqvqrHost){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SpCcsrlsqvqrHost"));
    CusolverHandler::setLogLevel(&logger);
    cusolverSpHandle_t handle = (cusolverSpHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    int nnz = in->Get<int>();
    cusparseMatDescr_t descrA = (cusparseMatDescr_t)in->Get<size_t>();
    cuComplex *csrValA = in->Assign<cuComplex>(nnz);
    int *csrRowPtrA = in->Assign<int>(m + 1);
    int *csrColIndA = in->Assign<int>(nnz);
    cuComplex *b = in->Assign<cuComplex>(m);
    float tol = in->Get<float>();
    int rankA;
    cuComplex *x = in->Get<cuComplex>(n);
    int *p = in->Get<int>(n);
    float min_norm;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverSpCcsrlsqvqrHost(handle, m, n, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol, &rankA, x, p, &min_norm);
        out->Add<int>(rankA);
        out->Add<cuComplex>(x, n);
        out->Add<int>(p, n);
        out->Add<float>(min_norm);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    } catch(const char *e) {
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverSpCcsrlsqvqrHost Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(SpZcsrlsqvqrHost){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SpZcsrlsqvqrHost"));
    CusolverHandler::setLogLevel(&logger);
    cusolverSpHandle_t handle = (cusolverSpHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    int nnz = in->Get<int>();
    cusparseMatDescr_t descrA = (cusparseMatDescr_t)in->Get<size_t>();
    cuDoubleComplex *csrValA = in->Assign<cuDoubleComplex>(nnz);
    int *csrRowPtrA = in->Assign<int>(m + 1);
    int *csrColIndA = in->Assign<int>(nnz);
    cuDoubleComplex *b = in->Assign<cuDoubleComplex>(m);
    double tol = in->Get<double>();
    int rankA;
    cuDoubleComplex *x = in->Get<cuDoubleComplex>(n);
    int *p = in->Get<int>(n);
    double min_norm;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverSpZcsrlsqvqrHost(handle, m, n, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol, &rankA, x, p, &min_norm);
        out->Add<int>(rankA);
        out->Add<cuDoubleComplex>(x, n);
        out->Add<int>(p, n);
        out->Add<double>(min_norm);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    } catch(const char *e) {
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverSpZcsrlsqvqrHost Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(SpScsreigvsiHost){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SpScsreigvsiHost"));
    CusolverHandler::setLogLevel(&logger);
    cusolverSpHandle_t handle = (cusolverSpHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int nnz = in->Get<int>();
    cusparseMatDescr_t descrA = (cusparseMatDescr_t)in->Get<size_t>();
    float *csrValA = in->Assign<float>(nnz);
    int *csrRowPtrA = in->Assign<int>(m + 1);
    int *csrColIndA = in->Assign<int>(nnz);
    float mu0 = in->Get<float>();
    float *x0 = in->Get<float>(m);
    int maxite = in->Get<int>();
    float tol = in->Get<float>();
    float mu;
    float *x = in->Get<float>(m);
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverSpScsreigvsiHost(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, mu0, x0, maxite, tol, &mu, x);
        out->Add<float>(mu);
        out->Add<float>(x, m);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    } catch(const char *e) {
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverSpScsreigvsiHost Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(SpDcsreigvsiHost){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SpDcsreigvsiHost"));
    CusolverHandler::setLogLevel(&logger);
    cusolverSpHandle_t handle = (cusolverSpHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int nnz = in->Get<int>();
    cusparseMatDescr_t descrA = (cusparseMatDescr_t)in->Get<size_t>();
    double *csrValA = in->Assign<double>(nnz);
    int *csrRowPtrA = in->Assign<int>(m + 1);
    int *csrColIndA = in->Assign<int>(nnz);
    double mu0 = in->Get<double>();
    double *x0 = in->Get<double>(m);
    int maxite = in->Get<int>();
    double tol = in->Get<double>();
    double mu;
    double *x = in->Get<double>(m);
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverSpDcsreigvsiHost(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, mu0, x0, maxite, tol, &mu, x);
        out->Add<double>(mu);
        out->Add<double>(x, m);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    } catch(const char *e) {
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverSpDcsreigvsiHost Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(SpCcsreigvsiHost){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SpCcsreigvsiHost"));
    CusolverHandler::setLogLevel(&logger);
    cusolverSpHandle_t handle = (cusolverSpHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int nnz = in->Get<int>();
    cusparseMatDescr_t descrA = (cusparseMatDescr_t)in->Get<size_t>();
    cuComplex *csrValA = in->Assign<cuComplex>(nnz);
    int *csrRowPtrA = in->Assign<int>(m + 1);
    int *csrColIndA = in->Assign<int>(nnz);
    cuComplex mu0 = in->Get<cuComplex>();
    cuComplex *x0 = in->Get<cuComplex>(m);
    int maxite = in->Get<int>();
    float tol = in->Get<float>();
    cuComplex mu;
    cuComplex *x = in->Get<cuComplex>(m);
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverSpCcsreigvsiHost(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, mu0, x0, maxite, tol, &mu, x);
        out->Add<cuComplex>(mu);
        out->Add<cuComplex>(x, m);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    } catch(const char *e) {
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverSpCcsreigvsiHost Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(SpZcsreigvsiHost){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SpZcsreigvsiHost"));
    CusolverHandler::setLogLevel(&logger);
    cusolverSpHandle_t handle = (cusolverSpHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int nnz = in->Get<int>();
    cusparseMatDescr_t descrA = (cusparseMatDescr_t)in->Get<size_t>();
    cuDoubleComplex *csrValA = in->Assign<cuDoubleComplex>(nnz);
    int *csrRowPtrA = in->Assign<int>(m + 1);
    int *csrColIndA = in->Assign<int>(nnz);
    cuDoubleComplex mu0 = in->Get<cuDoubleComplex>();
    cuDoubleComplex *x0 = in->Get<cuDoubleComplex>(m);
    int maxite = in->Get<int>();
    double tol = in->Get<double>();
    cuDoubleComplex mu;
    cuDoubleComplex *x = in->Get<cuDoubleComplex>(m);
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverSpZcsreigvsiHost(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, mu0, x0, maxite, tol, &mu, x);
        out->Add<cuDoubleComplex>(mu);
        out->Add<cuDoubleComplex>(x, m);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    } catch(const char *e) {
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverSpZcsreigvsiHost Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(SpScsreigsHost){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SpScsreigsHost"));
    CusolverHandler::setLogLevel(&logger);
    cusolverSpHandle_t handle = (cusolverSpHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int nnz = in->Get<int>();
    cusparseMatDescr_t descrA = (cusparseMatDescr_t)in->Get<size_t>();
    float *csrValA = in->Assign<float>(nnz);
    int *csrRowPtrA = in->Assign<int>(m + 1);
    int *csrColIndA = in->Assign<int>(nnz);
    cuComplex left_bottom_corner = in->Get<cuComplex>();
    cuComplex right_upper_corner = in->Get<cuComplex>();
    int num_eigs;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverSpScsreigsHost(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, left_bottom_corner, right_upper_corner, &num_eigs);
        out->Add<int>(num_eigs);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    } catch(const char *e) {
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverSpScsreigsHost Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(SpDcsreigsHost){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SpDcsreigsHost"));
    CusolverHandler::setLogLevel(&logger);
    cusolverSpHandle_t handle = (cusolverSpHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int nnz = in->Get<int>();
    cusparseMatDescr_t descrA = (cusparseMatDescr_t)in->Get<size_t>();
    double *csrValA = in->Assign<double>(nnz);
    int *csrRowPtrA = in->Assign<int>(m + 1);
    int *csrColIndA = in->Assign<int>(nnz);
    cuDoubleComplex left_bottom_corner = in->Get<cuDoubleComplex>();
    cuDoubleComplex right_upper_corner = in->Get<cuDoubleComplex>();
    int num_eigs;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverSpDcsreigsHost(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, left_bottom_corner, right_upper_corner, &num_eigs);
        out->Add<int>(num_eigs);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    } catch(const char *e) {
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverSpDcsreigsHost Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(SpCcsreigsHost){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SpCcsreigsHost"));
    CusolverHandler::setLogLevel(&logger);
    cusolverSpHandle_t handle = (cusolverSpHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int nnz = in->Get<int>();
    cusparseMatDescr_t descrA = (cusparseMatDescr_t)in->Get<size_t>();
    cuComplex *csrValA = in->Assign<cuComplex>(nnz);
    int *csrRowPtrA = in->Assign<int>(m + 1);
    int *csrColIndA = in->Assign<int>(nnz);
    cuComplex left_bottom_corner = in->Get<cuComplex>();
    cuComplex right_upper_corner = in->Get<cuComplex>();
    int num_eigs;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverSpCcsreigsHost(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, left_bottom_corner, right_upper_corner, &num_eigs);
        out->Add<int>(num_eigs);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    } catch(const char *e) {
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverSpCcsreigsHost Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(SpZcsreigsHost){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SpZcsreigsHost"));
    CusolverHandler::setLogLevel(&logger);
    cusolverSpHandle_t handle = (cusolverSpHandle_t)in->Get<size_t>();
    int m = in->Get<int>();
    int nnz = in->Get<int>();
    cusparseMatDescr_t descrA = (cusparseMatDescr_t)in->Get<size_t>();
    cuDoubleComplex *csrValA = in->Assign<cuDoubleComplex>(nnz);
    int *csrRowPtrA = in->Assign<int>(m + 1);
    int *csrColIndA = in->Assign<int>(nnz);
    cuDoubleComplex left_bottom_corner = in->Get<cuDoubleComplex>();
    cuDoubleComplex right_upper_corner = in->Get<cuDoubleComplex>();
    int num_eigs;
    cusolverStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusolverSpZcsreigsHost(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, left_bottom_corner, right_upper_corner, &num_eigs);
        out->Add<int>(num_eigs);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    } catch(const char *e) {
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusolverSpZcsreigsHost Executed");
    return std::make_shared<Result>(cs, out);
}