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
 * Written by: Antonio Pilato <antonio.pilato001@studenti.uniparthenope.it>,
 *             Nicholas Piantadosi <nicholas.piantadosi@studenti.uniparthenope.it>,
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
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Sgtsv2_bufferSizeExt));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Dgtsv2_bufferSizeExt));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Cgtsv2_bufferSizeExt));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Zgtsv2_bufferSizeExt));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Sgtsv2));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Dgtsv2));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Cgtsv2));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Zgtsv2));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Sgtsv2_nopivot_bufferSizeExt));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Dgtsv2_nopivot_bufferSizeExt));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Cgtsv2_nopivot_bufferSizeExt));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Zgtsv2_nopivot_bufferSizeExt));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Sgtsv2_nopivot));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Dgtsv2_nopivot));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Cgtsv2_nopivot));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Zgtsv2_nopivot));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Sgtsv2StridedBatch_bufferSizeExt));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Dgtsv2StridedBatch_bufferSizeExt));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Cgtsv2StridedBatch_bufferSizeExt));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Zgtsv2StridedBatch_bufferSizeExt));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Sgtsv2StridedBatch));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Dgtsv2StridedBatch));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Cgtsv2StridedBatch));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Zgtsv2StridedBatch));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(SgtsvInterleavedBatch_bufferSizeExt));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(DgtsvInterleavedBatch_bufferSizeExt));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(CgtsvInterleavedBatch_bufferSizeExt));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(ZgtsvInterleavedBatch_bufferSizeExt));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(SgtsvInterleavedBatch));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(DgtsvInterleavedBatch));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(CgtsvInterleavedBatch));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(ZgtsvInterleavedBatch));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(SgpsvInterleavedBatch_bufferSizeExt));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(DgpsvInterleavedBatch_bufferSizeExt));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(CgpsvInterleavedBatch_bufferSizeExt));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(ZgpsvInterleavedBatch_bufferSizeExt));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(SgpsvInterleavedBatch));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(DgpsvInterleavedBatch));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(CgpsvInterleavedBatch));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(ZgpsvInterleavedBatch));
    //REORDERINGS REFERENCE
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Scsrcolor));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Dcsrcolor));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Ccsrcolor));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Zcsrcolor));
    // FORMAT CONVERSION REFERENCE
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Sbsr2csr));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Dbsr2csr));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Cbsr2csr));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Zbsr2csr));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Sgebsr2gebsc_bufferSize));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Dgebsr2gebsc_bufferSize));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Cgebsr2gebsc_bufferSize));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Zgebsr2gebsc_bufferSize));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Sgebsr2gebsc));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Dgebsr2gebsc));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Cgebsr2gebsc));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Zgebsr2gebsc));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Sgebsr2gebsr_bufferSize));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Dgebsr2gebsr_bufferSize));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Cgebsr2gebsr_bufferSize));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Zgebsr2gebsr_bufferSize));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Xgebsr2gebsrNnz));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Sgebsr2gebsr));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Dgebsr2gebsr));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Cgebsr2gebsr));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Zgebsr2gebsr));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Sgebsr2csr));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Dgebsr2csr));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Cgebsr2csr));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Zgebsr2csr));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Scsr2gebsr_bufferSize));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Dcsr2gebsr_bufferSize));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Ccsr2gebsr_bufferSize));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Zcsr2gebsr_bufferSize));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Xcsr2gebsrNnz));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Scsr2gebsr));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Dcsr2gebsr));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Ccsr2gebsr));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Zcsr2gebsr));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Xcoo2csr));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Scsc2dense));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Dcsc2dense));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Ccsc2dense));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Zcsc2dense));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Xcsr2bsrNnz));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Scsr2bsr));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Dcsr2bsr));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Ccsr2bsr));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Zcsr2bsr));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Xcsr2coo));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Csr2cscEx2_bufferSize));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Csr2cscEx2));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Scsr2dense));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Dcsr2dense));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Ccsr2dense));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Zcsr2dense));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Scsr2csr_compress));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Dcsr2csr_compress));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Ccsr2csr_compress));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Zcsr2csr_compress));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Sdense2csc));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Ddense2csc));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Cdense2csc));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Zdense2csc));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Sdense2csr));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Ddense2csr));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Cdense2csr));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Zdense2csr));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Snnz));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Dnnz));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Cnnz));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Znnz));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(CreateIdentityPermutation));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Xcoosort_bufferSizeExt));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(XcoosortByRow));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(XcoosortByColumn));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Xcsrsort_bufferSizeExt));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Xcsrsort));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Xcscsort_bufferSizeExt));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Xcscsort));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(CreateCsru2csrInfo));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(DestroyCsru2csrInfo));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Scsru2csr_bufferSizeExt));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Dcsru2csr_bufferSizeExt));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Ccsru2csr_bufferSizeExt));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Zcsru2csr_bufferSizeExt));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Scsru2csr));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Dcsru2csr));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Ccsru2csr));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Zcsru2csr));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Scsr2csru));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Dcsr2csru));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Ccsr2csru));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Zcsr2csru));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(HpruneDense2csr_bufferSizeExt));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(SpruneDense2csr_bufferSizeExt));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(DpruneDense2csr_bufferSizeExt));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(HpruneDense2csrNnz));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(SpruneDense2csrNnz));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(DpruneDense2csrNnz));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(HpruneDense2csr));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(SpruneDense2csr));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(DpruneDense2csr));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(HpruneCsr2csr_bufferSizeExt));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(SpruneCsr2csr_bufferSizeExt));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(DpruneCsr2csr_bufferSizeExt));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(HpruneCsr2csrNnz));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(SpruneCsr2csrNnz));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(DpruneCsr2csrNnz));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(HpruneCsr2csr));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(SpruneCsr2csr));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(DpruneCsr2csr));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(HpruneDense2csrByPercentage_bufferSizeExt));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(SpruneDense2csrByPercentage_bufferSizeExt));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(DpruneDense2csrByPercentage_bufferSizeExt));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(HpruneDense2csrNnzByPercentage));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(SpruneDense2csrNnzByPercentage));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(DpruneDense2csrNnzByPercentage));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(HpruneDense2csrByPercentage));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(SpruneDense2csrByPercentage));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(DpruneDense2csrByPercentage));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(HpruneCsr2csrByPercentage_bufferSizeExt));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(SpruneCsr2csrByPercentage_bufferSizeExt));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(DpruneCsr2csrByPercentage_bufferSizeExt));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(HpruneCsr2csrNnzByPercentage));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(SpruneCsr2csrNnzByPercentage));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(DpruneCsr2csrNnzByPercentage));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(HpruneCsr2csrByPercentage));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(SpruneCsr2csrByPercentage));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(DpruneCsr2csrByPercentage));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Snnz_compress));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Dnnz_compress));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Cnnz_compress));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Znnz_compress));
    // GENERIC API REFERENCE - SPARSE VECTOR API
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(CreateSpVec));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(DestroySpVec));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(SpVecGet));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(SpVecGetIndexBase));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(SpVecGetValues));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(SpVecSetValues));
    // GENERIC API REFERENCE - SPARSE MATRIX API
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(CreateCoo));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(CreateCooAoS));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(CreateCsr));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(CreateCsc));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(CreateBlockedEll));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(DestroySpMat));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(CooGet));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(CooAoSGet));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(CsrGet));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(CsrSetPointers));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(CscSetPointers));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(CooSetPointers));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(BlockedEllGet));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(SpMatGetSize));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(SpMatGetFormat));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(SpMatGetIndexBase));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(SpMatGetValues));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(SpMatSetValues));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(SpMatGetStridedBatch));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(SpMatSetStridedBatch));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(CooSetStridedBatch));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(CsrSetStridedBatch));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(SpMatGetAttribute));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(SpMatSetAttribute));
    // GENERIC API REFERENCE - DENSE VECTOR API
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(CreateDnVec));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(DestroyDnVec));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(DnVecGet));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(DnVecGetValues));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(DnVecSetValues));
    // GENERIC API REFERENCE - DENSE MATRIX API
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(CreateDnMat));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(DestroyDnMat));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(DnMatGet));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(DnMatGetValues));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(DnMatSetValues));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(DnMatGetStridedBatch));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(DnMatSetStridedBatch));
    // GENERIC API REFERENCE - GENERIC API FUNCTIONS
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(SparseToDense_bufferSize));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(SparseToDense));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(DenseToSparse_bufferSize));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(DenseToSparse_analysis));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(DenseToSparse_convert));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Axpby));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Gather));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Scatter));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Rot));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(SpVV_bufferSize));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(SpVV));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(SpMV_bufferSize));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(SpMV));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(SpSV_createDescr));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(SpSV_destroyDescr));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(SpSV_bufferSize));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(SpSV_analysis));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(SpSV_solve));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(SpMM_bufferSize));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(SpMM_preprocess));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(SpMM));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(SpSM_createDescr));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(SpSM_destroyDescr));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(SpSM_bufferSize));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(SpSM_analysis));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(SpSM_solve));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(ConstrainedGeMM));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(ConstrainedGeMM_bufferSize));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(SDDMM_bufferSize));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(SDDMM_preprocess));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(SDDMM));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(SpGEMM_createDescr));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(SpGEMM_destroyDescr));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(SpGEMM_workEstimation));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(SpGEMM_compute));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(SpGEMM_copy));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(SpGEMM_createDescr));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(SpGEMM_destroyDescr));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(SpGEMMreuse_workEstimation));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(SpGEMMreuse_nnz));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(SpGEMMreuse_copy));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(SpGEMMreuse_compute));
}
