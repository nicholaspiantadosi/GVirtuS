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
#include <cuda_runtime.h>

using namespace log4cplus;

using gvirtus::communicators::Buffer;
using gvirtus::communicators::Result;

CUSPARSE_ROUTINE_HANDLER(CreateSpVec){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("CreateSpVec"));
    CusparseHandler::setLogLevel(&logger);
    int64_t size = in->Get<int64_t>();
    int64_t nnz = in->Get<int64_t>();
    void* indices = in->Get<void*>();
    void* values = in->Get<void*>();
    cusparseIndexType_t idxType = in->Get<cusparseIndexType_t>();
    cusparseIndexBase_t idxBase = in->Get<cusparseIndexBase_t>();
    cudaDataType valueType = in->Get<cudaDataType>();
    cusparseSpVecDescr_t * spVecDescr = new cusparseSpVecDescr_t;
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseCreateSpVec(spVecDescr, size, nnz, indices, values, idxType, idxBase, valueType);
        out->Add<cusparseSpVecDescr_t>(spVecDescr);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseCreateSpVec Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(DestroySpVec){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DestroySpVec"));
    CusparseHandler::setLogLevel(&logger);
    cusparseSpVecDescr_t spVecDescr = (cusparseSpVecDescr_t)in->Get<size_t>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseDestroySpVec(spVecDescr);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseDestroySpVec Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(SpVecGet){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SpVecGet"));
    CusparseHandler::setLogLevel(&logger);
    cusparseSpVecDescr_t spVecDescr = in->Get<cusparseSpVecDescr_t>();
    int64_t size = 0;
    int64_t nnz = 0;
    void* indices;
    void* values;
    cusparseIndexType_t idxType;
    cusparseIndexBase_t idxBase;
    cudaDataType valueType;
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseSpVecGet(spVecDescr, &size, &nnz, &indices, &values, &idxType, &idxBase, &valueType);
        out->Add<int64_t>(size);
        out->Add<int64_t>(nnz);
        out->Add<void*>(indices);
        out->Add<void*>(values);
        out->Add<cusparseIndexType_t>(idxType);
        out->Add<cusparseIndexBase_t>(idxBase);
        out->Add<cudaDataType>(valueType);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseSpVecGet Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(SpVecGetIndexBase){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SpVecGetIndexBase"));
    CusparseHandler::setLogLevel(&logger);
    cusparseSpVecDescr_t spVecDescr = in->Get<cusparseSpVecDescr_t>();
    cusparseIndexBase_t idxBase;
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseSpVecGetIndexBase(spVecDescr, &idxBase);
        out->Add<cusparseIndexBase_t>(idxBase);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseSpVecGetIndexBase Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(SpVecGetValues){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SpVecGetValues"));
    CusparseHandler::setLogLevel(&logger);
    cusparseSpVecDescr_t spVecDescr = in->Get<cusparseSpVecDescr_t>();
    void* values;
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseSpVecGetValues(spVecDescr, &values);
        out->Add<void*>(values);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseSpVecGetValues Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(SpVecSetValues){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SpVecSetValues"));
    CusparseHandler::setLogLevel(&logger);
    cusparseSpVecDescr_t spVecDescr = in->Get<cusparseSpVecDescr_t>();
    void* values = in->Get<void*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseSpVecSetValues(spVecDescr, values);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseSpVecSetValues Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(CreateCoo){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("CreateCoo"));
    CusparseHandler::setLogLevel(&logger);
    int64_t rows = in->Get<int64_t>();
    int64_t cols = in->Get<int64_t>();
    int64_t nnz = in->Get<int64_t>();
    void *cooRowInd = in->Get<void*>();
    void *cooColInd = in->Get<void*>();
    void *cooValues = in->Get<void*>();
    cusparseIndexType_t cooIdxType = in->Get<cusparseIndexType_t>();
    cusparseIndexBase_t idxBase = in->Get<cusparseIndexBase_t>();
    cudaDataType valueType = in->Get<cudaDataType>();
    cusparseSpMatDescr_t * spMatDescr = new cusparseSpMatDescr_t;
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseCreateCoo(spMatDescr, rows, cols, nnz, cooRowInd, cooColInd, cooValues, cooIdxType, idxBase, valueType);
        out->Add<cusparseSpMatDescr_t>(spMatDescr);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseCreateCoo Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(CreateCooAoS){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("CreateCooAoS"));
    CusparseHandler::setLogLevel(&logger);
    int64_t rows = in->Get<int64_t>();
    int64_t cols = in->Get<int64_t>();
    int64_t nnz = in->Get<int64_t>();
    void *cooInd = in->Get<void*>();
    void *cooValues = in->Get<void*>();
    cusparseIndexType_t cooIdxType = in->Get<cusparseIndexType_t>();
    cusparseIndexBase_t idxBase = in->Get<cusparseIndexBase_t>();
    cudaDataType valueType = in->Get<cudaDataType>();
    cusparseSpMatDescr_t * spMatDescr = new cusparseSpMatDescr_t;
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseCreateCooAoS(spMatDescr, rows, cols, nnz, cooInd, cooValues, cooIdxType, idxBase, valueType);
        out->Add<cusparseSpMatDescr_t>(spMatDescr);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseCreateCooAoS Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(CreateCsr){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("CreateCsr"));
    CusparseHandler::setLogLevel(&logger);
    int64_t rows = in->Get<int64_t>();
    int64_t cols = in->Get<int64_t>();
    int64_t nnz = in->Get<int64_t>();
    void *csrRowOffsets = in->Get<void*>();
    void *csrColInd = in->Get<void*>();
    void *csrValues = in->Get<void*>();
    cusparseIndexType_t csrRowOffsetsType = in->Get<cusparseIndexType_t>();
    cusparseIndexType_t csrColIndType = in->Get<cusparseIndexType_t>();
    cusparseIndexBase_t idxBase = in->Get<cusparseIndexBase_t>();
    cudaDataType valueType = in->Get<cudaDataType>();
    cusparseSpMatDescr_t * spMatDescr = new cusparseSpMatDescr_t;
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseCreateCsr(spMatDescr, rows, cols, nnz, csrRowOffsets, csrColInd, csrValues, csrRowOffsetsType, csrColIndType, idxBase, valueType);
        out->Add<cusparseSpMatDescr_t>(spMatDescr);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseCreateCsr Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(CreateCsc){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("CreateCsc"));
    CusparseHandler::setLogLevel(&logger);
    int64_t rows = in->Get<int64_t>();
    int64_t cols = in->Get<int64_t>();
    int64_t nnz = in->Get<int64_t>();
    void *cscColOffsets = in->Get<void*>();
    void *cscRowInd = in->Get<void*>();
    void *cscValues = in->Get<void*>();
    cusparseIndexType_t cscColOffsetsType = in->Get<cusparseIndexType_t>();
    cusparseIndexType_t cscRowIndType = in->Get<cusparseIndexType_t>();
    cusparseIndexBase_t idxBase = in->Get<cusparseIndexBase_t>();
    cudaDataType valueType = in->Get<cudaDataType>();
    cusparseSpMatDescr_t * spMatDescr = new cusparseSpMatDescr_t;
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseCreateCsc(spMatDescr, rows, cols, nnz, cscColOffsets, cscRowInd, cscValues, cscColOffsetsType, cscRowIndType, idxBase, valueType);
        out->Add<cusparseSpMatDescr_t>(spMatDescr);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseCreateCsc Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(CreateBlockedEll){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("CreateBlockedEll"));
    CusparseHandler::setLogLevel(&logger);
    int64_t rows = in->Get<int64_t>();
    int64_t cols = in->Get<int64_t>();
    int64_t ellBlockSize = in->Get<int64_t>();
    int64_t ellCols = in->Get<int64_t>();
    void *ellColInd = in->Get<void*>();
    void *ellValue = in->Get<void*>();
    cusparseIndexType_t ellIdxType = in->Get<cusparseIndexType_t>();
    cusparseIndexBase_t idxBase = in->Get<cusparseIndexBase_t>();
    cudaDataType valueType = in->Get<cudaDataType>();
    cusparseSpMatDescr_t * spMatDescr = new cusparseSpMatDescr_t;
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseCreateBlockedEll(spMatDescr, rows, cols, ellBlockSize, ellCols, ellColInd, ellValue, ellIdxType, idxBase, valueType);
        out->Add<cusparseSpMatDescr_t>(spMatDescr);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseCreateBlockedEll Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(DestroySpMat){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DestroySpMat"));
    CusparseHandler::setLogLevel(&logger);
    cusparseSpMatDescr_t spMatDescr = (cusparseSpMatDescr_t)in->Get<size_t>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseDestroySpMat(spMatDescr);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseDestroySpMat Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(CooGet){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("CooGet"));
    CusparseHandler::setLogLevel(&logger);
    cusparseSpMatDescr_t spMatDescr = in->Get<cusparseSpMatDescr_t>();
    int64_t rows = 0;
    int64_t cols = 0;
    int64_t nnz = 0;
    void* cooRowInd;
    void* cooColInd;
    void* cooValues;
    cusparseIndexType_t idxType;
    cusparseIndexBase_t idxBase;
    cudaDataType valueType;
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseCooGet(spMatDescr, &rows, &cols, &nnz, &cooRowInd, &cooColInd, &cooValues, &idxType, &idxBase, &valueType);
        out->Add<int64_t>(rows);
        out->Add<int64_t>(cols);
        out->Add<int64_t>(nnz);
        out->Add<void*>(cooRowInd);
        out->Add<void*>(cooColInd);
        out->Add<void*>(cooValues);
        out->Add<cusparseIndexType_t>(idxType);
        out->Add<cusparseIndexBase_t>(idxBase);
        out->Add<cudaDataType>(valueType);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseCooGet Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(CooAoSGet){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("CooAoSGet"));
    CusparseHandler::setLogLevel(&logger);
    cusparseSpMatDescr_t spMatDescr = in->Get<cusparseSpMatDescr_t>();
    int64_t rows = 0;
    int64_t cols = 0;
    int64_t nnz = 0;
    void* cooInd;
    void* cooValues;
    cusparseIndexType_t idxType;
    cusparseIndexBase_t idxBase;
    cudaDataType valueType;
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseCooAoSGet(spMatDescr, &rows, &cols, &nnz, &cooInd, &cooValues, &idxType, &idxBase, &valueType);
        out->Add<int64_t>(rows);
        out->Add<int64_t>(cols);
        out->Add<int64_t>(nnz);
        out->Add<void*>(cooInd);
        out->Add<void*>(cooValues);
        out->Add<cusparseIndexType_t>(idxType);
        out->Add<cusparseIndexBase_t>(idxBase);
        out->Add<cudaDataType>(valueType);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseCooAoSGet Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(CsrGet){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("CsrGet"));
    CusparseHandler::setLogLevel(&logger);
    cusparseSpMatDescr_t spMatDescr = in->Get<cusparseSpMatDescr_t>();
    int64_t rows = 0;
    int64_t cols = 0;
    int64_t nnz = 0;
    void* csrRowOffsets;
    void* csrColInd;
    void* csrValues;
    cusparseIndexType_t csrRowOffsetsType;
    cusparseIndexType_t csrColIndType;
    cusparseIndexBase_t idxBase;
    cudaDataType valueType;
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseCsrGet(spMatDescr, &rows, &cols, &nnz, &csrRowOffsets, &csrColInd, &csrValues, &csrRowOffsetsType, &csrColIndType, &idxBase, &valueType);
        out->Add<int64_t>(rows);
        out->Add<int64_t>(cols);
        out->Add<int64_t>(nnz);
        out->Add<void*>(csrRowOffsets);
        out->Add<void*>(csrColInd);
        out->Add<void*>(csrValues);
        out->Add<cusparseIndexType_t>(csrRowOffsetsType);
        out->Add<cusparseIndexType_t>(csrColIndType);
        out->Add<cusparseIndexBase_t>(idxBase);
        out->Add<cudaDataType>(valueType);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseCsrGet Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(CsrSetPointers){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("CsrSetPointers"));
    CusparseHandler::setLogLevel(&logger);
    cusparseSpMatDescr_t spMatDescr = in->Get<cusparseSpMatDescr_t>();
    void* csrRowOffsets = in->Get<void*>();
    void* csrColInd = in->Get<void*>();
    void* csrValues = in->Get<void*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseCsrSetPointers(spMatDescr, csrRowOffsets, csrColInd, csrValues);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseCsrSetPointers Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(CscSetPointers){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("CscSetPointers"));
    CusparseHandler::setLogLevel(&logger);
    cusparseSpMatDescr_t spMatDescr = in->Get<cusparseSpMatDescr_t>();
    void* cscColOffsets = in->Get<void*>();
    void* cscRowInd = in->Get<void*>();
    void* cscValues = in->Get<void*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseCscSetPointers(spMatDescr, cscColOffsets, cscRowInd, cscValues);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseCscSetPointers Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(CooSetPointers){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("CooSetPointers"));
    CusparseHandler::setLogLevel(&logger);
    cusparseSpMatDescr_t spMatDescr = in->Get<cusparseSpMatDescr_t>();
    void* cooRows = in->Get<void*>();
    void* cooColumns = in->Get<void*>();
    void* cooValues = in->Get<void*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseCooSetPointers(spMatDescr, cooRows, cooColumns, cooValues);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseCooSetPointers Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(BlockedEllGet){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("BlockedEllGet"));
    CusparseHandler::setLogLevel(&logger);
    cusparseSpMatDescr_t spMatDescr = in->Get<cusparseSpMatDescr_t>();
    int64_t rows = 0;
    int64_t cols = 0;
    int64_t ellBlockSize = 0;
    int64_t ellCols = 0;
    void* ellColInd;
    void* ellValue;
    cusparseIndexType_t ellIdxType;
    cusparseIndexBase_t idxBase;
    cudaDataType valueType;
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseBlockedEllGet(spMatDescr, &rows, &cols, &ellBlockSize, &ellCols, &ellColInd, &ellValue, &ellIdxType, &idxBase, &valueType);
        out->Add<int64_t>(rows);
        out->Add<int64_t>(cols);
        out->Add<int64_t>(ellBlockSize);
        out->Add<int64_t>(ellCols);
        out->Add<void*>(ellColInd);
        out->Add<void*>(ellValue);
        out->Add<cusparseIndexType_t>(ellIdxType);
        out->Add<cusparseIndexBase_t>(idxBase);
        out->Add<cudaDataType>(valueType);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseBlockedEllGet Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(SpMatGetSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SpMatGetSize"));
    CusparseHandler::setLogLevel(&logger);
    cusparseSpMatDescr_t spMatDescr = in->Get<cusparseSpMatDescr_t>();
    int64_t rows = 0;
    int64_t cols = 0;
    int64_t nnz = 0;
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseSpMatGetSize(spMatDescr, &rows, &cols, &nnz);
        out->Add<int64_t>(rows);
        out->Add<int64_t>(cols);
        out->Add<int64_t>(nnz);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseSpMatGetSize Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(SpMatGetFormat){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SpMatGetFormat"));
    CusparseHandler::setLogLevel(&logger);
    cusparseSpMatDescr_t spMatDescr = in->Get<cusparseSpMatDescr_t>();
    cusparseFormat_t format;
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseSpMatGetFormat(spMatDescr, &format);
        out->Add<size_t>((size_t) format);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseSpMatGetFormat Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(SpMatGetIndexBase){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SpMatGetIndexBase"));
    CusparseHandler::setLogLevel(&logger);
    cusparseSpMatDescr_t spMatDescr = in->Get<cusparseSpMatDescr_t>();
    cusparseIndexBase_t idxBase;
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseSpMatGetIndexBase(spMatDescr, &idxBase);
        out->Add<cusparseIndexBase_t>(idxBase);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseSpMatGetIndexBase Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(SpMatGetValues){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SpMatGetValues"));
    CusparseHandler::setLogLevel(&logger);
    cusparseSpMatDescr_t spMatDescr = in->Get<cusparseSpMatDescr_t>();
    void* values;
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseSpMatGetValues(spMatDescr, &values);
        out->Add<void*>(values);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseSpMatGetValues Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(SpMatSetValues){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SpMatSetValues"));
    CusparseHandler::setLogLevel(&logger);
    cusparseSpMatDescr_t spMatDescr = in->Get<cusparseSpMatDescr_t>();
    void* values = in->Get<void*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseSpMatSetValues(spMatDescr, values);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseSpMatSetValues Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(SpMatGetStridedBatch){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SpMatGetStridedBatch"));
    CusparseHandler::setLogLevel(&logger);
    cusparseSpMatDescr_t spMatDescr = in->Get<cusparseSpMatDescr_t>();
    int batchCount = 0;
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseSpMatGetStridedBatch(spMatDescr, &batchCount);
        out->Add<int>(batchCount);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseSpMatGetStridedBatch Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(SpMatSetStridedBatch){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SpMatSetStridedBatch"));
    CusparseHandler::setLogLevel(&logger);
    cusparseSpMatDescr_t spMatDescr = in->Get<cusparseSpMatDescr_t>();
    int batchCount = in->Get<int>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseSpMatSetStridedBatch(spMatDescr, batchCount);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseSpMatSetStridedBatch Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(CooSetStridedBatch){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("CooSetStridedBatch"));
    CusparseHandler::setLogLevel(&logger);
    cusparseSpMatDescr_t spMatDescr = in->Get<cusparseSpMatDescr_t>();
    int batchCount = in->Get<int>();
    int64_t batchStride = in->Get<int64_t>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseCooSetStridedBatch(spMatDescr, batchCount, batchStride);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseCooSetStridedBatch Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(CsrSetStridedBatch){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("CsrSetStridedBatch"));
    CusparseHandler::setLogLevel(&logger);
    cusparseSpMatDescr_t spMatDescr = in->Get<cusparseSpMatDescr_t>();
    int batchCount = in->Get<int>();
    int64_t offsetsBatchStride = in->Get<int64_t>();
    int64_t columnsValuesBatchStride = in->Get<int64_t>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseCsrSetStridedBatch(spMatDescr, batchCount, offsetsBatchStride, columnsValuesBatchStride);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseCsrSetStridedBatch Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(SpMatGetAttribute){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SpMatGetAttribute"));
    CusparseHandler::setLogLevel(&logger);
    cusparseSpMatDescr_t spMatDescr = in->Get<cusparseSpMatDescr_t>();
    cusparseSpMatAttribute_t attribute = in->Get<cusparseSpMatAttribute_t>();
    size_t dataSize = in->Get<size_t>();
    void* data;
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseSpMatGetAttribute(spMatDescr, attribute, data, dataSize);
        out->Add<void*>(data);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseSpMatGetAttribute Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(SpMatSetAttribute){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SpMatSetAttribute"));
    CusparseHandler::setLogLevel(&logger);
    cusparseSpMatAttribute_t attribute = in->Get<cusparseSpMatAttribute_t>();
    size_t dataSize = in->Get<size_t>();
    void* data = in->Get<void*>();
    cusparseSpMatDescr_t spMatDescr;
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseSpMatSetAttribute(spMatDescr, attribute, data, dataSize);
        out->Add<cusparseSpMatDescr_t>(spMatDescr);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseSpMatSetAttribute Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(CreateDnVec){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("CreateDnVec"));
    CusparseHandler::setLogLevel(&logger);
    int64_t size = in->Get<int64_t>();
    void* values = in->Get<void*>();
    /*
    float hV[4];
    cudaMemcpy(hV, values, 4 * sizeof(float), cudaMemcpyDeviceToHost);
    printf("\tCreateDnVec BEFORE - values: [");
    for (int i = 0; i < 4; i++) {
        printf("%f", hV[i]);
        if (i < 3) {
            printf(", ");
        }
    }
    printf("]\n");
    */
    cudaDataType valueType = in->Get<cudaDataType>();
    //cusparseDnVecDescr_t* dnVecDescr = in->Get<cusparseDnVecDescr_t*>();
    //cusparseDnVecDescr_t * dnVecDescr = in->Assign<cusparseDnVecDescr_t>();
    cusparseDnVecDescr_t * dnVecDescr = new cusparseDnVecDescr_t;
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseCreateDnVec(dnVecDescr, size, values, valueType);
        //printf("\n\tBE - CreateDnVec - dnVecDescr pointer: %p\n", *dnVecDescr);
        /*
        cudaMemcpy(hV, dnVecDescr, 4 * sizeof(float), cudaMemcpyDeviceToHost);
        printf("\tCreateDnVec AFTER - values: [");
        for (int i = 0; i < 4; i++) {
            printf("%f", hV[i]);
            if (i < 3) {
                printf(", ");
            }
        }
        printf("]\n");
        */
        //out->Add<cusparseDnVecDescr_t*>(dnVecDescr);
        //out->Add<cusparseDnVecDescr_t>(dnVecDescr);
        out->Add<size_t>((size_t)*dnVecDescr);
        //out->AddMarshal<cusparseDnVecDescr_t*>(dnVecDescr);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseCreateDnVec Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(DestroyDnVec){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DestroyDnVec"));
    CusparseHandler::setLogLevel(&logger);
    cusparseDnVecDescr_t dnVecDescr = (cusparseDnVecDescr_t)in->Get<size_t>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseDestroyDnVec(dnVecDescr);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseDestroyDnVec Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(DnVecGet){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnVecGet"));
    CusparseHandler::setLogLevel(&logger);
    cusparseDnVecDescr_t dnVecDescr = in->Get<cusparseDnVecDescr_t>();
    int64_t size = 0;
    void* values;
    cudaDataType valueType;
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseDnVecGet(dnVecDescr, &size, &values, &valueType);
        out->Add<int64_t>(size);
        out->Add<void*>(values);
        out->Add<cudaDataType>(valueType);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseDnVecGet Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(DnVecGetValues){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnVecGetValues"));
    CusparseHandler::setLogLevel(&logger);
    cusparseDnVecDescr_t dnVecDescr = in->Get<cusparseDnVecDescr_t>();
    void* values;
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseDnVecGetValues(dnVecDescr, &values);
        out->Add<void*>(values);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseDnVecGetValues Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(DnVecSetValues){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnVecSetValues"));
    CusparseHandler::setLogLevel(&logger);
    cusparseDnVecDescr_t dnVecDescr = in->Get<cusparseDnVecDescr_t>();
    void* values = in->Get<void*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseDnVecSetValues(dnVecDescr, values);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseDnVecSetValues Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(CreateDnMat){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("CreateDnMat"));
    CusparseHandler::setLogLevel(&logger);
    int64_t rows = in->Get<int64_t>();
    int64_t cols = in->Get<int64_t>();
    int64_t ld = in->Get<int64_t>();
    void *values = in->Get<void*>();
    cudaDataType valueType = in->Get<cudaDataType>();
    cusparseOrder_t order = in->Get<cusparseOrder_t>();
    cusparseDnMatDescr_t * dnMatDescr = new cusparseDnMatDescr_t;
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseCreateDnMat(dnMatDescr, rows, cols, ld, values, valueType, order);
        out->Add<cusparseDnMatDescr_t>(dnMatDescr);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseCreateDnMat Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(DestroyDnMat){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DestroyDnMat"));
    CusparseHandler::setLogLevel(&logger);
    cusparseDnMatDescr_t dnMatDescr = (cusparseDnMatDescr_t)in->Get<size_t>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseDestroyDnMat(dnMatDescr);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseDestroyDnMat Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(DnMatGet){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnMatGet"));
    CusparseHandler::setLogLevel(&logger);
    cusparseDnMatDescr_t dnMatDescr = in->Get<cusparseDnMatDescr_t>();
    int64_t rows = 0;
    int64_t cols = 0;
    int64_t ld = 0;
    void* values;
    cudaDataType type;
    cusparseOrder_t order;
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseDnMatGet(dnMatDescr, &rows, &cols, &ld, &values, &type, &order);
        out->Add<int64_t>(rows);
        out->Add<int64_t>(cols);
        out->Add<int64_t>(ld);
        out->Add<void*>(values);
        out->Add<cudaDataType>(type);
        out->Add<cusparseOrder_t>(order);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseDnMatGet Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(DnMatGetValues){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnMatGetValues"));
    CusparseHandler::setLogLevel(&logger);
    cusparseDnMatDescr_t dnMatDescr = in->Get<cusparseDnMatDescr_t>();
    void* values;
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseDnMatGetValues(dnMatDescr, &values);
        out->Add<void*>(values);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseDnMatGetValues Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(DnMatSetValues){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnMatSetValues"));
    CusparseHandler::setLogLevel(&logger);
    cusparseDnMatDescr_t dnMatDescr = in->Get<cusparseDnMatDescr_t>();
    void* values = in->Get<void*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseDnMatSetValues(dnMatDescr, values);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseDnMatSetValues Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(DnMatGetStridedBatch){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnMatGetStridedBatch"));
    CusparseHandler::setLogLevel(&logger);
    cusparseDnMatDescr_t dnMatDescr = in->Get<cusparseDnMatDescr_t>();
    int batchCount = 0;
    int64_t batchStride = 0;
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseDnMatGetStridedBatch(dnMatDescr, &batchCount, &batchStride);
        out->Add<int>(batchCount);
        out->Add<int64_t>(batchStride);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseDnMatGetStridedBatch Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(DnMatSetStridedBatch){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DnMatSetStridedBatch"));
    CusparseHandler::setLogLevel(&logger);
    cusparseDnMatDescr_t dnMatDescr = in->Get<cusparseDnMatDescr_t>();
    int batchCount = in->Get<int>();
    int64_t batchStride = in->Get<int64_t>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseDnMatSetStridedBatch(dnMatDescr, batchCount, batchStride);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseDnMatSetStridedBatch Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(SparseToDense_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SparseToDense_bufferSize"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    cusparseSpMatDescr_t matA = in->Get<cusparseSpMatDescr_t>();
    cusparseDnMatDescr_t matB = in->Get<cusparseDnMatDescr_t>();
    cusparseSparseToDenseAlg_t alg = in->Get<cusparseSparseToDenseAlg_t>();
    size_t * bufferSize = new size_t;
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseSparseToDense_bufferSize(handle, matA, matB, alg, bufferSize);
        out->Add<size_t>(bufferSize);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseSparseToDense_bufferSize Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(SparseToDense){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SparseToDense"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    cusparseSpMatDescr_t matA = in->Get<cusparseSpMatDescr_t>();
    cusparseDnMatDescr_t matB = in->Get<cusparseDnMatDescr_t>();
    cusparseSparseToDenseAlg_t alg = in->Get<cusparseSparseToDenseAlg_t>();
    void * buffer = in->Get<void*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseSparseToDense(handle, matA, matB, alg, buffer);
        out->Add<cusparseDnMatDescr_t>(matB);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseSparseToDense Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(DenseToSparse_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DenseToSparse_bufferSize"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    cusparseDnMatDescr_t matA = in->Get<cusparseDnMatDescr_t>();
    cusparseSpMatDescr_t matB = in->Get<cusparseSpMatDescr_t>();
    cusparseDenseToSparseAlg_t alg = in->Get<cusparseDenseToSparseAlg_t>();
    size_t * bufferSize = new size_t;
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseDenseToSparse_bufferSize(handle, matA, matB, alg, bufferSize);
        out->Add<size_t>(bufferSize);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseDenseToSparse_bufferSize Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(DenseToSparse_analysis){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DenseToSparse_analysis"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    cusparseDnMatDescr_t matA = in->Get<cusparseDnMatDescr_t>();
    cusparseSpMatDescr_t matB = in->Get<cusparseSpMatDescr_t>();
    cusparseDenseToSparseAlg_t alg = in->Get<cusparseDenseToSparseAlg_t>();
    void * buffer = in->Get<void*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseDenseToSparse_analysis(handle, matA, matB, alg, buffer);
        out->Add<cusparseSpMatDescr_t>(matB);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseDenseToSparse_analysis Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(DenseToSparse_convert){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DenseToSparse_convert"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    cusparseDnMatDescr_t matA = in->Get<cusparseDnMatDescr_t>();
    cusparseSpMatDescr_t matB = in->Get<cusparseSpMatDescr_t>();
    cusparseDenseToSparseAlg_t alg = in->Get<cusparseDenseToSparseAlg_t>();
    void * buffer = in->Get<void*>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseDenseToSparse_convert(handle, matA, matB, alg, buffer);
        out->Add<cusparseSpMatDescr_t>(matB);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseDenseToSparse_convert Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Axpby){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Axpby"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    const float * alpha = in->Assign<float>();
    cusparseSpVecDescr_t vecX = in->Get<cusparseSpVecDescr_t>();
    const void* beta = in->Assign<float>();
    cusparseDnVecDescr_t vecY = in->Get<cusparseDnVecDescr_t>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseAxpby(handle, alpha, vecX, beta, vecY);
        out->Add<cusparseDnVecDescr_t>(vecY);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseAxpby Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Gather){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Gather"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    cusparseDnVecDescr_t vecY = in->Get<cusparseDnVecDescr_t>();
    cusparseSpVecDescr_t vecX = in->Get<cusparseSpVecDescr_t>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseGather(handle, vecY, vecX);
        out->Add<cusparseSpVecDescr_t>(vecX);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseGather Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Scatter){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Scatter"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    cusparseSpVecDescr_t vecX = in->Get<cusparseSpVecDescr_t>();
    cusparseDnVecDescr_t vecY = in->Get<cusparseDnVecDescr_t>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseScatter(handle, vecX, vecY);
        out->Add<cusparseDnVecDescr_t>(vecY);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseScatter Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(Rot){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Rot"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    const float * c_coeff = in->Assign<float>();
    const float * s_coeff = in->Assign<float>();
    cusparseSpVecDescr_t vecX = in->Get<cusparseSpVecDescr_t>();
    cusparseDnVecDescr_t vecY = in->Get<cusparseDnVecDescr_t>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseRot(handle, c_coeff, s_coeff, vecX, vecY);
        out->Add<cusparseSpVecDescr_t>(vecX);
        out->Add<cusparseDnVecDescr_t>(vecY);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseRot Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(SpVV_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SpVV_bufferSize"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    cusparseOperation_t opX = in->Get<cusparseOperation_t>();
    cusparseSpVecDescr_t vecX = (cusparseSpVecDescr_t)in->Get<size_t>();
    cusparseDnVecDescr_t vecY = (cusparseDnVecDescr_t)in->Get<size_t>();
    void* result = in->Get<void*>();
    cudaDataType computeType = in->Get<cudaDataType>();
    size_t * bufferSize = new size_t;
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseSpVV_bufferSize(handle, opX, vecX, vecY, result, computeType, bufferSize);
        out->Add<size_t>(bufferSize);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseSpVV_bufferSize Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(SpVV){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SpVV"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    cusparseOperation_t opX = in->Get<cusparseOperation_t>();
    cusparseSpVecDescr_t vecX = (cusparseSpVecDescr_t)in->Get<size_t>();
    cusparseDnVecDescr_t vecY = (cusparseDnVecDescr_t)in->Get<size_t>();
    cudaDataType computeType = in->Get<cudaDataType>();
    void* buffer = in->Get<void*>();
    //void* result = in->Get<void*>();

    //float* result = in->Get<float*>();

    //float * result = in->Assign<float>();
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        if (computeType == CUDA_R_32F) {
            // float

            //printf("%p\n", result);

            //printf("%f\n", *((float*)result));

            //printf("%f\n", *result);

            //cs = cusparseSpVV(handle, opX, vecX, vecY, result, computeType, buffer);

            //printf("EXECUTED\n");

            //out->Add<float>(result);
            //printf("%f\n", *((float*)result));

            //printf("%p\n", result);
            //printf("%f\n", *result);

            float result;
            cs = cusparseSpVV(handle, opX, vecX, vecY, &result, computeType, buffer);
            //printf("%p\n", result);
            //printf("%f\n", result);
            out->Add<float>(result);

            //out->Add<void*>(result);
            //out->AddMarshal<float*>(result);
            //out->Add<float*>(result);
        } else if (computeType == CUDA_R_64F) {
            // double
            double result;
            cs = cusparseSpVV(handle, opX, vecX, vecY, &result, computeType, buffer);
            out->Add<double>(result);
        } else if (computeType == CUDA_C_32F) {
            // cuComplex
            cuComplex result;
            cs = cusparseSpVV(handle, opX, vecX, vecY, &result, computeType, buffer);
            out->Add<cuComplex>(result);
        } else if (computeType == CUDA_C_64F) {
            // cuDoubleComplex
            cuDoubleComplex result;
            cs = cusparseSpVV(handle, opX, vecX, vecY, &result, computeType, buffer);
            out->Add<cuDoubleComplex>(result);
        } else {
            throw "Type not supported by GVirtus!";
        }
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseSpVV Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(SpMV_bufferSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SpMV_bufferSize"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    cusparseOperation_t opA = in->Get<cusparseOperation_t>();
    cusparseSpMatDescr_t matA = (cusparseSpMatDescr_t)in->Get<size_t>();
    cusparseDnVecDescr_t vecX = (cusparseDnVecDescr_t)in->Get<size_t>();
    //printf("\n\tBE - SpMV_bufferSize - vecX pointer: %p\n", vecX);
    cusparseDnVecDescr_t vecY = (cusparseDnVecDescr_t)in->Get<size_t>();
    //printf("\n\tBE - SpMV_bufferSize - vecY pointer: %p\n", vecY);
    cudaDataType computeType = in->Get<cudaDataType>();
    cusparseSpMVAlg_t alg = in->Get<cusparseSpMVAlg_t>();
    size_t * bufferSize = new size_t;
    void* alpha;
    void* beta;
    if (computeType == CUDA_R_32F) {
        // float
        float alphaFloat = in->Get<float>();
        float betaFloat = in->Get<float>();
        alpha = &alphaFloat;
        beta = &betaFloat;
    } else if (computeType == CUDA_R_64F) {
        // double
        double alphaDouble = in->Get<double>();
        double betaDouble = in->Get<double>();
        alpha = &alphaDouble;
        beta = &betaDouble;
    } else if (computeType == CUDA_C_32F) {
        // cuComplex
        cuComplex alphaCuComplex = in->Get<cuComplex>();
        cuComplex betaCuComplex = in->Get<cuComplex>();
        alpha = &alphaCuComplex;
        beta = &betaCuComplex;
    } else if (computeType == CUDA_C_64F) {
        // cuDoubleComplex
        cuDoubleComplex alphaCuDoubleComplex = in->Get<cuDoubleComplex>();
        cuDoubleComplex betaCuDoubleComplex = in->Get<cuDoubleComplex>();
        alpha = &alphaCuDoubleComplex;
        beta = &betaCuDoubleComplex;
    } else {
        throw "Type not supported by GVirtus!";
    }
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseSpMV_bufferSize(handle, opA, alpha, matA, vecX, beta, vecY, computeType, alg, bufferSize);
        out->Add<size_t>(bufferSize);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseSpMV_bufferSize Executed");
    return std::make_shared<Result>(cs,out);
}

CUSPARSE_ROUTINE_HANDLER(SpMV){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SpMV"));
    CusparseHandler::setLogLevel(&logger);
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<size_t>();
    cusparseOperation_t opA = in->Get<cusparseOperation_t>();
    cusparseSpMatDescr_t matA = (cusparseSpMatDescr_t)in->Get<size_t>();
    cusparseDnVecDescr_t vecX = (cusparseDnVecDescr_t)in->Get<size_t>();
    //printf("\n\tBE - SpMV - vecX pointer: %p\n", vecX);
    cusparseDnVecDescr_t vecY = (cusparseDnVecDescr_t)in->Get<size_t>();
    //printf("\n\tBE - SpMV - vecY pointer: %p\n", vecY);
    cudaDataType computeType = in->Get<cudaDataType>();
    cusparseSpMVAlg_t alg = in->Get<cusparseSpMVAlg_t>();
    void* externalBuffer = in->Get<void*>();
    void* alpha;
    void* beta;
    if (computeType == CUDA_R_32F) {
        // float
        float alphaFloat = in->Get<float>();
        float betaFloat = in->Get<float>();
        alpha = &alphaFloat;
        beta = &betaFloat;
    } else if (computeType == CUDA_R_64F) {
        // double
        double alphaDouble = in->Get<double>();
        double betaDouble = in->Get<double>();
        alpha = &alphaDouble;
        beta = &betaDouble;
    } else if (computeType == CUDA_C_32F) {
        // cuComplex
        cuComplex alphaCuComplex = in->Get<cuComplex>();
        cuComplex betaCuComplex = in->Get<cuComplex>();
        alpha = &alphaCuComplex;
        beta = &betaCuComplex;
    } else if (computeType == CUDA_C_64F) {
        // cuDoubleComplex
        cuDoubleComplex alphaCuDoubleComplex = in->Get<cuDoubleComplex>();
        cuDoubleComplex betaCuDoubleComplex = in->Get<cuDoubleComplex>();
        alpha = &alphaCuDoubleComplex;
        beta = &betaCuDoubleComplex;
    } else {
        throw "Type not supported by GVirtus!";
    }
    cusparseStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        cs = cusparseSpMV(handle, opA, alpha, matA, vecX, beta, vecY, computeType, alg, externalBuffer);
        out->Add<size_t>((size_t)vecY);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    } catch(const char *e) {
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseSpMV Executed");
    return std::make_shared<Result>(cs,out);
}

#ifndef CUSPARSE_VERSION
#error CUSPARSE_VERSION not defined
#endif
