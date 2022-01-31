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

CUSPARSE_ROUTINE_HANDLER(Scsrcolor){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Scsrcolor"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    const int m = in->Get<int>();
    const int nnz = in->Get<int>();
    const cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    float * csrValA = in->GetFromMarshal<float*>();
    int * csrRowPtrA = in->GetFromMarshal<int*>();
    int * csrColIndA = in->GetFromMarshal<int*>();
    float * fractionToColor = in->Assign<float>();
    int* ncolors = in->Assign<int>();
    int* coloring = in->GetFromMarshal<int*>();
    int* reordering = in->GetFromMarshal<int*>();
    cusparseColorInfo_t info = in->Get<cusparseColorInfo_t>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseScsrcolor(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, fractionToColor, ncolors, coloring, reordering, info);
        out->Add<int*>(ncolors);
        out->Add<int*>(coloring);
        out->Add<int*>(reordering);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseScsrcolor Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Dcsrcolor){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Dcsrcolor"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    const int m = in->Get<int>();
    const int nnz = in->Get<int>();
    const cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    double * csrValA = in->GetFromMarshal<double*>();
    int * csrRowPtrA = in->GetFromMarshal<int*>();
    int * csrColIndA = in->GetFromMarshal<int*>();
    double * fractionToColor = in->Assign<double>();
    int* ncolors = in->Assign<int>();
    int* coloring = in->GetFromMarshal<int*>();
    int* reordering = in->GetFromMarshal<int*>();
    cusparseColorInfo_t info = in->Get<cusparseColorInfo_t>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseDcsrcolor(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, fractionToColor, ncolors, coloring, reordering, info);
        out->Add<int*>(ncolors);
        out->Add<int*>(coloring);
        out->Add<int*>(reordering);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseDcsrcolor Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Ccsrcolor){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Ccsrcolor"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    const int m = in->Get<int>();
    const int nnz = in->Get<int>();
    const cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    cuComplex * csrValA = in->GetFromMarshal<cuComplex*>();
    int * csrRowPtrA = in->GetFromMarshal<int*>();
    int * csrColIndA = in->GetFromMarshal<int*>();
    float * fractionToColor = in->Assign<float>();
    int* ncolors = in->Assign<int>();
    int* coloring = in->GetFromMarshal<int*>();
    int* reordering = in->GetFromMarshal<int*>();
    cusparseColorInfo_t info = in->Get<cusparseColorInfo_t>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseCcsrcolor(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, fractionToColor, ncolors, coloring, reordering, info);
        out->Add<int*>(ncolors);
        out->Add<int*>(coloring);
        out->Add<int*>(reordering);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseCcsrcolor Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Zcsrcolor){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Zcsrcolor"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    const int m = in->Get<int>();
    const int nnz = in->Get<int>();
    const cusparseMatDescr_t descrA = in->Get<cusparseMatDescr_t>();
    cuDoubleComplex * csrValA = in->GetFromMarshal<cuDoubleComplex*>();
    int * csrRowPtrA = in->GetFromMarshal<int*>();
    int * csrColIndA = in->GetFromMarshal<int*>();
    double * fractionToColor = in->Assign<double>();
    int* ncolors = in->Assign<int>();
    int* coloring = in->GetFromMarshal<int*>();
    int* reordering = in->GetFromMarshal<int*>();
    cusparseColorInfo_t info = in->Get<cusparseColorInfo_t>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseZcsrcolor(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, fractionToColor, ncolors, coloring, reordering, info);
        out->Add<int*>(ncolors);
        out->Add<int*>(coloring);
        out->Add<int*>(reordering);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseZcsrcolor Executed");
    return std::make_shared<Result>(cs,out);
}

#ifndef CUSPARSE_VERSION
#error CUSPARSE_VERSION not defined
#endif
