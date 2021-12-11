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
    // LEVEL2
    // LEVEL3
}
