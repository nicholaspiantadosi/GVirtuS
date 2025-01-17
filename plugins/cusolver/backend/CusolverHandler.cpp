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
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnGetrs));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnZZgesv_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnZCgesv_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnZKgesv_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnZEgesv_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnZYgesv_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnCCgesv_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnCKgesv_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnCEgesv_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnCYgesv_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnDDgesv_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnDSgesv_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnDHgesv_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnDBgesv_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnDXgesv_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnSSgesv_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnSHgesv_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnSBgesv_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnSXgesv_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnZZgesv));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnZCgesv));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnZKgesv));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnZEgesv));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnZYgesv));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnCCgesv));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnCKgesv));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnCEgesv));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnCYgesv));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnDDgesv));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnDSgesv));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnDHgesv));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnDBgesv));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnDXgesv));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnSSgesv));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnSHgesv));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnSBgesv));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnSXgesv));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnIRSXgesv_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnIRSXgesv));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnSgeqrf_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnDgeqrf_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnCgeqrf_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnZgeqrf_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnSgeqrf));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnDgeqrf));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnCgeqrf));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnZgeqrf));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnGeqrf_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnGeqrf));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnZZgels_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnZCgels_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnZKgels_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnZEgels_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnZYgels_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnCCgels_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnCKgels_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnCEgels_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnCYgels_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnDDgels_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnDSgels_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnDHgels_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnDBgels_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnDXgels_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnSSgels_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnSHgels_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnSBgels_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnSXgels_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnZZgels));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnZCgels));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnZKgels));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnZEgels));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnZYgels));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnCCgels));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnCKgels));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnCEgels));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnCYgels));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnDDgels));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnDSgels));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnDHgels));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnDBgels));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnDXgels));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnSSgels));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnSHgels));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnSBgels));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnSXgels));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnIRSXgels_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnIRSXgels));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnSormqr_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnDormqr_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnCunmqr_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnZunmqr_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnSormqr));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnDormqr));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnCunmqr));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnZunmqr));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnSorgqr_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnDorgqr_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnCungqr_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnZungqr_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnSorgqr));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnDorgqr));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnCungqr));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnZungqr));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnSsytrf_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnDsytrf_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnCsytrf_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnZsytrf_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnSsytrf));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnDsytrf));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnCsytrf));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnZsytrf));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnSpotrfBatched));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnDpotrfBatched));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnCpotrfBatched));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnZpotrfBatched));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnSpotrsBatched));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnDpotrsBatched));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnCpotrsBatched));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnZpotrsBatched));
    // DENSE LAPACK - DENSE EIGENVALUES SOLVER
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnSgebrd_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnDgebrd_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnCgebrd_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnZgebrd_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnSgebrd));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnDgebrd));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnCgebrd));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnZgebrd));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnSorgbr_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnDorgbr_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnCungbr_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnZungbr_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnSorgbr));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnDorgbr));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnCungbr));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnZungbr));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnSsytrd_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnDsytrd_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnChetrd_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnZhetrd_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnSsytrd));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnDsytrd));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnChetrd));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnZhetrd));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnSormtr_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnDormtr_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnCunmtr_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnZunmtr_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnSormtr));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnDormtr));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnCunmtr));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnZunmtr));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnSorgtr_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnDorgtr_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnCungtr_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnZungtr_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnSorgtr));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnDorgtr));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnCungtr));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnZungtr));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnSgesvd_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnDgesvd_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnCgesvd_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnZgesvd_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnSgesvd));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnDgesvd));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnCgesvd));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnZgesvd));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnGesvd_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnGesvd));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnSgesvdj_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnDgesvdj_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnCgesvdj_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnZgesvdj_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnSgesvdj));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnDgesvdj));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnCgesvdj));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnZgesvdj));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnSgesvdjBatched_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnDgesvdjBatched_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnCgesvdjBatched_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnZgesvdjBatched_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnSgesvdjBatched));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnDgesvdjBatched));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnCgesvdjBatched));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnZgesvdjBatched));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnSgesvdaStridedBatched_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnDgesvdaStridedBatched_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnCgesvdaStridedBatched_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnZgesvdaStridedBatched_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnSgesvdaStridedBatched));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnDgesvdaStridedBatched));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnCgesvdaStridedBatched));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnZgesvdaStridedBatched));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnSsyevd_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnDsyevd_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnCheevd_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnZheevd_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnSsyevd));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnDsyevd));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnCheevd));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnZheevd));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnSyevd_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnSyevd));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnSsyevdx_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnDsyevdx_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnCheevdx_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnZheevdx_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnSsyevdx));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnDsyevdx));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnCheevdx));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnZheevdx));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnSsygvd_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnDsygvd_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnChegvd_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnZhegvd_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnSsygvd));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnDsygvd));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnChegvd));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnZhegvd));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnSsygvdx_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnDsygvdx_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnChegvdx_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnZhegvdx_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnSsygvdx));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnDsygvdx));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnChegvdx));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnZhegvdx));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnSsyevj_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnDsyevj_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnCheevj_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnZheevj_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnSsyevj));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnDsyevj));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnCheevj));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnZheevj));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnSsygvj_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnDsygvj_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnChegvj_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnZhegvj_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnSsygvj));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnDsygvj));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnChegvj));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnZhegvj));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnSsyevjBatched_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnDsyevjBatched_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnCheevjBatched_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnZheevjBatched_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnSsyevjBatched));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnDsyevjBatched));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnCheevjBatched));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnZheevjBatched));
    // DENSE LAPACK - DENSE LINEAR SOLVER - 64-BIT
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnXpotrf_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnXpotrf));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnXpotrs));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnXgetrf_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnXgetrf));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnXgetrs));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnXgeqrf_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnXgeqrf));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnXsytrs_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnXsytrs));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnXtrtri_bufferSize));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnXtrtri));
    // SPARSE LAPACK - HELPER FUNCTION
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(SpCreate));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(SpDestroy));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(SpSetStream));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(SpXcsrissymHost));
    // SPARSE LAPACK - HIGH LEVEL FUNCTION
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(SpScsrlsvluHost));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(SpDcsrlsvluHost));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(SpCcsrlsvluHost));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(SpZcsrlsvluHost));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(SpScsrlsvqrHost));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(SpDcsrlsvqrHost));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(SpCcsrlsvqrHost));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(SpZcsrlsvqrHost));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(SpScsrlsvcholHost));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(SpDcsrlsvcholHost));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(SpCcsrlsvcholHost));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(SpZcsrlsvcholHost));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(SpScsrlsqvqrHost));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(SpDcsrlsqvqrHost));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(SpCcsrlsqvqrHost));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(SpZcsrlsqvqrHost));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(SpScsreigvsiHost));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(SpDcsreigvsiHost));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(SpCcsreigvsiHost));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(SpZcsreigvsiHost));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(SpScsreigsHost));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(SpDcsreigsHost));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(SpCcsreigsHost));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(SpZcsreigsHost));
    // SPARSE LAPACK - LOW LEVEL FUNCTION
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(SpXcsrsymrcmHost));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(SpXcsrsymmdqHost));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(SpXcsrsymamdHost));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(SpXcsrmetisndHost));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(SpScsrzfdHost));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(SpDcsrzfdHost));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(SpCcsrzfdHost));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(SpZcsrzfdHost));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(SpXcsrperm_bufferSizeHost));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(SpXcsrpermHost));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(SpCreateCsrqrInfo));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(SpDestroyCsrqrInfo));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(SpXcsrqrAnalysisBatched));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(SpScsrqrBufferInfoBatched));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(SpDcsrqrBufferInfoBatched));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(SpCcsrqrBufferInfoBatched));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(SpZcsrqrBufferInfoBatched));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(SpScsrqrsvBatched));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(SpDcsrqrsvBatched));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(SpCcsrqrsvBatched));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(SpZcsrqrsvBatched));
    // REFACTORIZATION
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(RfAccessBundledFactorsDevice));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(RfAnalyze));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(RfSetupDevice));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(RfSetupHost));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(RfCreate));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(RfExtractBundledFactorsHost));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(RfExtractSplitFactorsHost));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(RfDestroy));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(RfGetMatrixFormat));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(RfGetNumericProperties));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(RfGetNumericBoostReport));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(RfGetResetValuesFastMode));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(RfGetAlgs));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(RfRefactor));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(RfResetValues));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(RfSetMatrixFormat));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(RfSetNumericProperties));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(RfSetResetValuesFastMode));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(RfSetAlgs));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(RfSolve));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(RfBatchSetupHost));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(RfBatchAnalyze));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(RfBatchResetValues));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(RfBatchRefactor));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(RfBatchSolve));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(RfBatchZeroPivot));
}
