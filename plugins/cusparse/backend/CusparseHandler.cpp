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
 * Written by: Antonio Pilato <antonio.pilato001@studenti.uniparthenope.it>
 *             Department of Science and Technologies
 *
 */

#include <cstring>
#include <map>
#include <errno.h>
#include <cuda_runtime_api.h>
#include "CusparseHandler.h"

using namespace std;
using namespace log4cplus;

using gvirtus::communicators::Buffer;
using gvirtus::communicators::Result;

std::map<string, CusparseHandler::CusparseRoutineHandler> * CusparseHandler::mspHandlers = NULL;

extern "C" std::shared_ptr<CusparseHandler> create_t() {
    return std::make_shared<CusparseHandler>();
}

extern "C" int HandlerInit() {
    return 0;
}

CusparseHandler::CusparseHandler() {
    logger=Logger::getInstance(LOG4CPLUS_TEXT("CusparseHandler"));
    setLogLevel(&logger);
    Initialize();
}

CusparseHandler::~CusparseHandler() {

}

void CusparseHandler::setLogLevel(Logger *logger) {
        log4cplus::LogLevel logLevel=log4cplus::INFO_LOG_LEVEL;
        char * val = getenv("GVIRTUS_LOGLEVEL");
        std::string logLevelString =(val == NULL ? std::string("") : std::string(val));
        if(logLevelString != "") {
                logLevel=std::stoi(logLevelString);
        }
        logger->setLogLevel(logLevel);
}

bool CusparseHandler::CanExecute(std::string routine) {
    return mspHandlers->find(routine) != mspHandlers->end();

}

std::shared_ptr<Result> CusparseHandler::Execute(std::string routine, std::shared_ptr<Buffer> input_buffer) {
    LOG4CPLUS_DEBUG(logger,"Called " << routine);
    map<string, CusparseHandler::CusparseRoutineHandler>::iterator it;
    it = mspHandlers->find(routine);
    if (it == mspHandlers->end())
        throw "No handler for '" + routine + "' found!";
    try {
        return it->second(this, input_buffer);
    } catch (const char *ex) {
        LOG4CPLUS_DEBUG(logger,ex);
        LOG4CPLUS_DEBUG(logger,strerror(errno));
    }
    return NULL;
}

void CusparseHandler::Initialize(){
   if (mspHandlers != NULL)
        return;
    mspHandlers = new map<string, CusparseHandler::CusparseRoutineHandler> ();
    // MANAGEMENT
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(GetVersion));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Create));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Destroy));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(GetErrorString));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(SetStream));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(GetStream));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(GetProperty));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(GetPointerMode));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(SetPointerMode));
    // HELPER
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(CreateColorInfo));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(CreateMatDescr));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(DestroyColorInfo));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(DestroyMatDescr));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(GetMatDiagType));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(GetMatFillMode));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(GetMatIndexBase));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(GetMatType));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(SetMatDiagType));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(SetMatFillMode));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(SetMatIndexBase));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(SetMatType));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(CreateCsrsv2Info));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(DestroyCsrsv2Info));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(CreateCsrsm2Info));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(DestroyCsrsm2Info));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(CreateCsric02Info));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(DestroyCsric02Info));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(CreateCsrilu02Info));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(DestroyCsrilu02Info));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(CreateBsrsv2Info));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(DestroyBsrsv2Info));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(CreateBsrsm2Info));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(DestroyBsrsm2Info));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(CreateBsric02Info));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(DestroyBsric02Info));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(CreateBsrilu02Info));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(DestroyBsrilu02Info));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(CreateCsrgemm2Info));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(DestroyCsrgemm2Info));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(CreatePruneInfo));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(DestroyPruneInfo));
    // LEVEL1
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Saxpyi));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Daxpyi));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Caxpyi));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Zaxpyi));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Sgthr));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Dgthr));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Cgthr));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Zgthr));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Sgthrz));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Dgthrz));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Cgthrz));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Zgthrz));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Sroti));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Droti));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Ssctr));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Dsctr));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Csctr));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Zsctr));
    // LEVEL2
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Sbsrmv));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Dbsrmv));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Cbsrmv));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Zbsrmv));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Sbsrxmv));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Dbsrxmv));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Cbsrxmv));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Zbsrxmv));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Sbsrsv2_bufferSize));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Dbsrsv2_bufferSize));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Cbsrsv2_bufferSize));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Zbsrsv2_bufferSize));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Sbsrsv2_analysis));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Dbsrsv2_analysis));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Cbsrsv2_analysis));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Zbsrsv2_analysis));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Sbsrsv2_solve));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Dbsrsv2_solve));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Cbsrsv2_solve));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Zbsrsv2_solve));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Xbsrsv2_zeroPivot));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(CsrmvEx_bufferSize));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(CsrmvEx));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Scsrsv2_bufferSize));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Dcsrsv2_bufferSize));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Ccsrsv2_bufferSize));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Zcsrsv2_bufferSize));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Scsrsv2_analysis));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Dcsrsv2_analysis));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Ccsrsv2_analysis));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Zcsrsv2_analysis));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Scsrsv2_solve));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Dcsrsv2_solve));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Ccsrsv2_solve));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Zcsrsv2_solve));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Xcsrsv2_zeroPivot));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Sgemvi_bufferSize));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Dgemvi_bufferSize));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Cgemvi_bufferSize));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Zgemvi_bufferSize));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Sgemvi));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Dgemvi));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Cgemvi));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Zgemvi));
    // LEVEL3
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Sbsrmm));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Dbsrmm));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Cbsrmm));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Zbsrmm));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Sbsrsm2_bufferSize));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Dbsrsm2_bufferSize));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Cbsrsm2_bufferSize));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Zbsrsm2_bufferSize));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Sbsrsm2_analysis));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Dbsrsm2_analysis));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Cbsrsm2_analysis));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Zbsrsm2_analysis));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Sbsrsm2_solve));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Dbsrsm2_solve));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Cbsrsm2_solve));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Zbsrsm2_solve));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Xbsrsm2_zeroPivot));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Scsrsm2_bufferSizeExt));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Dcsrsm2_bufferSizeExt));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Ccsrsm2_bufferSizeExt));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Zcsrsm2_bufferSizeExt));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Scsrsm2_analysis));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Dcsrsm2_analysis));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Ccsrsm2_analysis));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Zcsrsm2_analysis));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Scsrsm2_solve));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Dcsrsm2_solve));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Ccsrsm2_solve));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Zcsrsm2_solve));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Xcsrsm2_zeroPivot));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Sgemmi));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Dgemmi));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Cgemmi));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Zgemmi));
    // EXTRA
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Scsrgeam2_bufferSizeExt));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Dcsrgeam2_bufferSizeExt));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Ccsrgeam2_bufferSizeExt));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Zcsrgeam2_bufferSizeExt));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Xcsrgeam2Nnz));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Scsrgeam2));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Dcsrgeam2));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Ccsrgeam2));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Zcsrgeam2));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Scsrgemm2_bufferSizeExt));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Dcsrgemm2_bufferSizeExt));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Ccsrgemm2_bufferSizeExt));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Zcsrgemm2_bufferSizeExt));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Xcsrgemm2Nnz));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Scsrgemm2));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Dcsrgemm2));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Ccsrgemm2));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Zcsrgemm2));
    // PRECONDITIONERS REFERENCE
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Scsric02_bufferSize));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Dcsric02_bufferSize));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Ccsric02_bufferSize));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Zcsric02_bufferSize));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Scsric02_analysis));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Dcsric02_analysis));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Ccsric02_analysis));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Zcsric02_analysis));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Scsric02));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Dcsric02));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Ccsric02));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Zcsric02));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Xcsric02_zeroPivot));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Sbsric02_bufferSize));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Dbsric02_bufferSize));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Cbsric02_bufferSize));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Zbsric02_bufferSize));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Sbsric02_analysis));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Dbsric02_analysis));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Cbsric02_analysis));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Zbsric02_analysis));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Sbsric02));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Dbsric02));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Cbsric02));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Zbsric02));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Xbsric02_zeroPivot));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Scsrilu02_numericBoost));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Dcsrilu02_numericBoost));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Ccsrilu02_numericBoost));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Zcsrilu02_numericBoost));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Scsrilu02_bufferSize));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Dcsrilu02_bufferSize));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Ccsrilu02_bufferSize));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Zcsrilu02_bufferSize));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Scsrilu02_analysis));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Dcsrilu02_analysis));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Ccsrilu02_analysis));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Zcsrilu02_analysis));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Scsrilu02));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Dcsrilu02));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Ccsrilu02));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Zcsrilu02));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Xcsrilu02_zeroPivot));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Sbsrilu02_numericBoost));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Dbsrilu02_numericBoost));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Cbsrilu02_numericBoost));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Zbsrilu02_numericBoost));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Sbsrilu02_bufferSize));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Dbsrilu02_bufferSize));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Cbsrilu02_bufferSize));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Zbsrilu02_bufferSize));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Sbsrilu02_analysis));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Dbsrilu02_analysis));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Cbsrilu02_analysis));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Zbsrilu02_analysis));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Sbsrilu02));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Dbsrilu02));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Cbsrilu02));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Zbsrilu02));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Xbsrilu02_zeroPivot));
    // FORMAT CONVERSION REFERENCE
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Xcsr2bsrNnz));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Scsr2bsr));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Dcsr2bsr));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Ccsr2bsr));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Zcsr2bsr));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Sdense2csr));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Ddense2csr));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Cdense2csr));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Zdense2csr));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Snnz));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Dnnz));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Cnnz));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Znnz));
    // GENERIC API REFERENCE - SPARSE MATRIX API
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(CreateCsr));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(DestroySpMat));
    // GENERIC API REFERENCE - DENSE VECTOR API
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(CreateDnVec));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(DestroyDnVec));
    // GENERIC API REFERENCE - GENERIC API FUNCTIONS
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(SpMV_bufferSize));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(SpMV));
}
