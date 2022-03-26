/*
 *   gVirtuS -- A GPGPU transparent virtualization component.
 *
 *   Copyright (C) 2009-2010  The University of Napoli Parthenope at Naples.
 *
 *   This file is part of gVirtuS.
 *
 *   gVirtuS is free software; you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation; either version 2 of the License, or
 *   (at your option) any later version.
 *
 *   gVirtuS is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU Lesser General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with gVirtuS; if not, write to the Free Software
 *   Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 *  Written by: Nicholas Piantadosi <nicholas.piantadosi@studenti.uniparthenope.it>,
 *              Department of Science and Technologies
 *
 */

#include <cstring>
#include <map>
#include <errno.h>
#include <cuda_runtime_api.h>
#include "CusolverMgHandler.h"

using namespace std;
using namespace log4cplus;

using gvirtus::communicators::Buffer;
using gvirtus::communicators::Result;

std::map<string, CusolverMgHandler::CusolverMgRoutineHandler> * CusolverMgHandler::mspHandlers = NULL;

extern "C" std::shared_ptr<CusolverMgHandler> create_t() {
    return std::make_shared<CusolverMgHandler>();
}


extern "C" int HandlerInit() {
    return 0;
}

CusolverMgHandler::CusolverMgHandler() {
    logger=Logger::getInstance(LOG4CPLUS_TEXT("CusolverMgHandler"));
    setLogLevel(&logger);
    Initialize();
}

CusolverMgHandler::~CusolverMgHandler() {

}

void CusolverMgHandler::setLogLevel(Logger *logger) {
        log4cplus::LogLevel logLevel=log4cplus::INFO_LOG_LEVEL;
        char * val = getenv("GVIRTUS_LOGLEVEL");
        std::string logLevelString =(val == NULL ? std::string("") : std::string(val));
        if(logLevelString != "") {
                logLevel=std::stoi(logLevelString);
        }
        logger->setLogLevel(logLevel);
}

bool CusolverMgHandler::CanExecute(std::string routine) {
    return mspHandlers->find(routine) != mspHandlers->end();
}

std::shared_ptr<Result> CusolverMgHandler::Execute(std::string routine, std::shared_ptr<Buffer> input_buffer) {
    LOG4CPLUS_DEBUG(logger,"Called " << routine);
    map<string, CusolverMgHandler::CusolverMgRoutineHandler>::iterator it;
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

void CusolverMgHandler::Initialize(){
   if (mspHandlers != NULL)
        return;
    mspHandlers = new map<string, CusolverMgHandler::CusolverMgRoutineHandler> ();
    // MULTI GPU - HELPER FUNCTION
    mspHandlers->insert(CUSOLVERMG_ROUTINE_HANDLER_PAIR(MgCreate));
    mspHandlers->insert(CUSOLVERMG_ROUTINE_HANDLER_PAIR(MgDestroy));
    mspHandlers->insert(CUSOLVERMG_ROUTINE_HANDLER_PAIR(MgDeviceSelect));
    mspHandlers->insert(CUSOLVERMG_ROUTINE_HANDLER_PAIR(MgCreateDeviceGrid));
    mspHandlers->insert(CUSOLVERMG_ROUTINE_HANDLER_PAIR(MgDestroyGrid));
    mspHandlers->insert(CUSOLVERMG_ROUTINE_HANDLER_PAIR(MgCreateMatrixDesc));
    mspHandlers->insert(CUSOLVERMG_ROUTINE_HANDLER_PAIR(MgDestroyMatrixDesc));
    // MULTI GPU - DENSE LINEAR SOLVER FUNCTION
    mspHandlers->insert(CUSOLVERMG_ROUTINE_HANDLER_PAIR(MgPotrf_bufferSize));
    mspHandlers->insert(CUSOLVERMG_ROUTINE_HANDLER_PAIR(MgPotrf));
    mspHandlers->insert(CUSOLVERMG_ROUTINE_HANDLER_PAIR(MgPotrs_bufferSize));
    mspHandlers->insert(CUSOLVERMG_ROUTINE_HANDLER_PAIR(MgPotrs));
    mspHandlers->insert(CUSOLVERMG_ROUTINE_HANDLER_PAIR(MgPotri_bufferSize));
    mspHandlers->insert(CUSOLVERMG_ROUTINE_HANDLER_PAIR(MgPotri));
    mspHandlers->insert(CUSOLVERMG_ROUTINE_HANDLER_PAIR(MgGetrf_bufferSize));
    mspHandlers->insert(CUSOLVERMG_ROUTINE_HANDLER_PAIR(MgGetrf));
    mspHandlers->insert(CUSOLVERMG_ROUTINE_HANDLER_PAIR(MgGetrs_bufferSize));
    mspHandlers->insert(CUSOLVERMG_ROUTINE_HANDLER_PAIR(MgGetrs));
    // MULTI GPU - DENSE LINEAR EIGENVALUE FUNCTION
    mspHandlers->insert(CUSOLVERMG_ROUTINE_HANDLER_PAIR(MgSyevd_bufferSize));
    mspHandlers->insert(CUSOLVERMG_ROUTINE_HANDLER_PAIR(MgSyevd));

}
