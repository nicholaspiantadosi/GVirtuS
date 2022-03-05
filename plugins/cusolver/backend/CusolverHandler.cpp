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
 *  Written by: Antonio Pilato <antonio.pilato001@studenti.uniparthenope.it>,
 *              Nicholas Piantadosi <nicholas.piantadosi@studenti.uniparthenope.it>,
 *              Department of Science and Technologies
 *
 */

#include <cstring>
#include <map>
#include <errno.h>
#include <cuda_runtime_api.h>
#include "CusolverHandler.h"

using namespace std;
using namespace log4cplus;

using gvirtus::communicators::Buffer;
using gvirtus::communicators::Result;

std::map<string, CusolverHandler::CusolverRoutineHandler> * CusolverHandler::mspHandlers = NULL;

extern "C" std::shared_ptr<CusolverHandler> create_t() {
    return std::make_shared<CusolverHandler>();
}


extern "C" int HandlerInit() {
    return 0;
}

CusolverHandler::CusolverHandler() {
    logger=Logger::getInstance(LOG4CPLUS_TEXT("CusolverHandler"));
    setLogLevel(&logger);
    Initialize();
}

CusolverHandler::~CusolverHandler() {

}

void CusolverHandler::setLogLevel(Logger *logger) {
        log4cplus::LogLevel logLevel=log4cplus::INFO_LOG_LEVEL;
        char * val = getenv("GVIRTUS_LOGLEVEL");
        std::string logLevelString =(val == NULL ? std::string("") : std::string(val));
        if(logLevelString != "") {
                logLevel=std::stoi(logLevelString);
        }
        logger->setLogLevel(logLevel);
}

bool CusolverHandler::CanExecute(std::string routine) {
    return mspHandlers->find(routine) != mspHandlers->end();
}

std::shared_ptr<Result> CusolverHandler::Execute(std::string routine, std::shared_ptr<Buffer> input_buffer) {
    LOG4CPLUS_DEBUG(logger,"Called " << routine);
    map<string, CusolverHandler::CusolverRoutineHandler>::iterator it;
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

void CusolverHandler::Initialize(){
   if (mspHandlers != NULL)
        return;
    mspHandlers = new map<string, CusolverHandler::CusolverRoutineHandler> ();
    // DENSE LAPACK - HELPER FUNCTION
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnCreate));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnDestroy));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnSetStream));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnGetStream));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnCreateSyevjInfo));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnDestroySyevjInfo));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnXsyevjSetTolerance));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnXsyevjSetMaxSweeps));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnXsyevjSetSortEig));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnXsyevjGetResidual));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnXsyevjGetSweeps));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnCreateGesvdjInfo));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnDestroyGesvdjInfo));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnXgesvdjSetTolerance));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnXgesvdjSetMaxSweeps));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnXgesvdjSetSortEig));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnXgesvdjGetResidual));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnXgesvdjGetSweeps));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnIRSParamsCreate));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnIRSParamsDestroy));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnIRSParamsSetSolverPrecisions));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnIRSParamsSetSolverMainPrecision));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnIRSParamsSetSolverLowestPrecision));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnIRSParamsSetRefinementSolver));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnIRSParamsSetTol));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnIRSParamsSetTolInner));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnIRSParamsSetMaxIters));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnIRSParamsSetMaxItersInner));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnIRSParamsEnableFallback));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnIRSParamsDisableFallback));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnIRSParamsGetMaxIters));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnIRSInfosCreate));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnIRSInfosDestroy));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnIRSInfosGetMaxIters));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnIRSInfosGetNiters));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnIRSInfosGetOuterNiters));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnIRSInfosRequestResidual));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnIRSInfosGetResidualHistory));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnCreateParams));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnDestroyParams));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnSetAdvOptions));
    // DENSE LAPACK - DENSE LINEAR SOLVER - LEGACY
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnSpotrf_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnDpotrf_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnCpotrf_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnZpotrf_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnSpotrf));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnDpotrf));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnCpotrf));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnZpotrf));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnPotrf_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnPotrf));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnSpotrs));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnDpotrs));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnCpotrs));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnZpotrs));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnPotrs));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnSpotri_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnDpotri_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnCpotri_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnZpotri_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnSpotri));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnDpotri));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnCpotri));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnZpotri));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnSgetrf_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnDgetrf_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnCgetrf_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnZgetrf_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnSgetrf));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnDgetrf));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnCgetrf));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnZgetrf));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnGetrf_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnGetrf));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnSgetrs));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnDgetrs));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnCgetrs));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnZgetrs));
    // DENSE LAPACK - DENSE EIGENVALUES SOLVER
    // DENSE LAPACK - DENSE LINEAR SOLVER - 64-BIT
    // SPARSE LAPACK - HELPER FUNCTION
    // SPARSE LAPACK - HIGH LEVEL FUNCTION
    // SPARSE LAPACK - LOW LEVEL FUNCTION
    // REFACTORIZATION

}
