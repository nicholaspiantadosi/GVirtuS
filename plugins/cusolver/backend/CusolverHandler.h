
/*
 *   gVirtuS -- A GPGPU transparent virtualization component.
 *   
 *  Copyright (C) 2009-2010  The University of Napoli Parthenope at Naples.
 *     
 *  This file is part of gVirtuS.
 *       
 *  gVirtuS is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *            
 *  gVirtuS is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU Lesser General Public License for more details.
 *                 
 *  You should have received a copy of the GNU General Public License
 *  along with gVirtuS; if not, write to the Free Software
 *  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 *                    
 *  Written by: Antonio Pilato <antonio.pilato001@studenti.uniparthenope.it>,
 *              Nicholas Piantadosi <nicholas.piantadosi@studenti.uniparthenope.it>,
 *  Department of Science and Technologies
 */

#ifndef CUSOLVERHANDLER_H
#define CUSOLVERHANDLER_H

#include <gvirtus/backend/Handler.h>
#include <gvirtus/communicators/Result.h>

#include <cusolverDn.h>
#include <cusolver_common.h>

using gvirtus::common::pointer_t;
using gvirtus::communicators::Buffer;
using gvirtus::communicators::Result;

#include "log4cplus/logger.h"
#include "log4cplus/loggingmacros.h"
#include "log4cplus/configurator.h"

#include <limits.h>
#if ( __WORDSIZE == 64 )
    #define BUILD_64   1
#endif

using namespace std;
using namespace log4cplus;

class CusolverHandler : public gvirtus::backend::Handler {
public:
    CusolverHandler();
    virtual ~CusolverHandler();
    bool CanExecute(std::string routine);
    std::shared_ptr<Result> Execute(std::string routine, std::shared_ptr<Buffer> input_buffer);
    static void setLogLevel(Logger *logger);
private:
    log4cplus::Logger logger;
    void Initialize();
    typedef std::shared_ptr<Result> (*CusolverRoutineHandler)(CusolverHandler *, std::shared_ptr<Buffer>);
    static std::map<std::string, CusolverRoutineHandler> * mspHandlers;
};

#define CUSOLVER_ROUTINE_HANDLER(name) std::shared_ptr<Result> handle##name(CusolverHandler * pThis, std::shared_ptr<Buffer> in)
#define CUSOLVER_ROUTINE_HANDLER_PAIR(name) make_pair("cusolver" #name, handle##name)

// DENSE LAPACK - HELPER FUNCTION
CUSOLVER_ROUTINE_HANDLER(DnCreate);
CUSOLVER_ROUTINE_HANDLER(DnDestroy);
CUSOLVER_ROUTINE_HANDLER(DnSetStream);
CUSOLVER_ROUTINE_HANDLER(DnGetStream);
CUSOLVER_ROUTINE_HANDLER(DnCreateSyevjInfo);
CUSOLVER_ROUTINE_HANDLER(DnDestroySyevjInfo);
CUSOLVER_ROUTINE_HANDLER(DnXsyevjSetTolerance);
CUSOLVER_ROUTINE_HANDLER(DnXsyevjSetMaxSweeps);
CUSOLVER_ROUTINE_HANDLER(DnXsyevjSetSortEig);
CUSOLVER_ROUTINE_HANDLER(DnXsyevjGetResidual);
CUSOLVER_ROUTINE_HANDLER(DnXsyevjGetSweeps);
CUSOLVER_ROUTINE_HANDLER(DnCreateGesvdjInfo);
CUSOLVER_ROUTINE_HANDLER(DnDestroyGesvdjInfo);
CUSOLVER_ROUTINE_HANDLER(DnXgesvdjSetTolerance);
CUSOLVER_ROUTINE_HANDLER(DnXgesvdjSetMaxSweeps);
CUSOLVER_ROUTINE_HANDLER(DnXgesvdjSetSortEig);
CUSOLVER_ROUTINE_HANDLER(DnXgesvdjGetResidual);
CUSOLVER_ROUTINE_HANDLER(DnXgesvdjGetSweeps);
CUSOLVER_ROUTINE_HANDLER(DnIRSParamsCreate);
CUSOLVER_ROUTINE_HANDLER(DnIRSParamsDestroy);
CUSOLVER_ROUTINE_HANDLER(DnIRSParamsSetSolverPrecisions);
CUSOLVER_ROUTINE_HANDLER(DnIRSParamsSetSolverMainPrecision);
CUSOLVER_ROUTINE_HANDLER(DnIRSParamsSetSolverLowestPrecision);
CUSOLVER_ROUTINE_HANDLER(DnIRSParamsSetRefinementSolver);
CUSOLVER_ROUTINE_HANDLER(DnIRSParamsSetTol);
CUSOLVER_ROUTINE_HANDLER(DnIRSParamsSetTolInner);
CUSOLVER_ROUTINE_HANDLER(DnIRSParamsSetMaxIters);
CUSOLVER_ROUTINE_HANDLER(DnIRSParamsSetMaxItersInner);
CUSOLVER_ROUTINE_HANDLER(DnIRSParamsEnableFallback);
CUSOLVER_ROUTINE_HANDLER(DnIRSParamsDisableFallback);
CUSOLVER_ROUTINE_HANDLER(DnIRSParamsGetMaxIters);
CUSOLVER_ROUTINE_HANDLER(DnIRSInfosCreate);
CUSOLVER_ROUTINE_HANDLER(DnIRSInfosDestroy);
CUSOLVER_ROUTINE_HANDLER(DnIRSInfosGetMaxIters);
CUSOLVER_ROUTINE_HANDLER(DnIRSInfosGetNiters);
CUSOLVER_ROUTINE_HANDLER(DnIRSInfosGetOuterNiters);
CUSOLVER_ROUTINE_HANDLER(DnIRSInfosRequestResidual);
CUSOLVER_ROUTINE_HANDLER(DnIRSInfosGetResidualHistory);
CUSOLVER_ROUTINE_HANDLER(DnCreateParams);
CUSOLVER_ROUTINE_HANDLER(DnDestroyParams);
CUSOLVER_ROUTINE_HANDLER(DnSetAdvOptions);
// DENSE LAPACK - DENSE LINEAR SOLVER - LEGACY
CUSOLVER_ROUTINE_HANDLER(DnSpotrf_bufferSize);
CUSOLVER_ROUTINE_HANDLER(DnDpotrf_bufferSize);
CUSOLVER_ROUTINE_HANDLER(DnCpotrf_bufferSize);
CUSOLVER_ROUTINE_HANDLER(DnZpotrf_bufferSize);
CUSOLVER_ROUTINE_HANDLER(DnSpotrf);
CUSOLVER_ROUTINE_HANDLER(DnDpotrf);
CUSOLVER_ROUTINE_HANDLER(DnCpotrf);
CUSOLVER_ROUTINE_HANDLER(DnZpotrf);
CUSOLVER_ROUTINE_HANDLER(DnPotrf_bufferSize);
CUSOLVER_ROUTINE_HANDLER(DnPotrf);
CUSOLVER_ROUTINE_HANDLER(DnSpotrs);
CUSOLVER_ROUTINE_HANDLER(DnDpotrs);
CUSOLVER_ROUTINE_HANDLER(DnCpotrs);
CUSOLVER_ROUTINE_HANDLER(DnZpotrs);
CUSOLVER_ROUTINE_HANDLER(DnPotrs);
CUSOLVER_ROUTINE_HANDLER(DnSpotri_bufferSize);
CUSOLVER_ROUTINE_HANDLER(DnDpotri_bufferSize);
CUSOLVER_ROUTINE_HANDLER(DnCpotri_bufferSize);
CUSOLVER_ROUTINE_HANDLER(DnZpotri_bufferSize);
CUSOLVER_ROUTINE_HANDLER(DnSpotri);
CUSOLVER_ROUTINE_HANDLER(DnDpotri);
CUSOLVER_ROUTINE_HANDLER(DnCpotri);
CUSOLVER_ROUTINE_HANDLER(DnZpotri);
CUSOLVER_ROUTINE_HANDLER(DnSgetrf_bufferSize);
CUSOLVER_ROUTINE_HANDLER(DnDgetrf_bufferSize);
CUSOLVER_ROUTINE_HANDLER(DnCgetrf_bufferSize);
CUSOLVER_ROUTINE_HANDLER(DnZgetrf_bufferSize);
CUSOLVER_ROUTINE_HANDLER(DnSgetrf);
CUSOLVER_ROUTINE_HANDLER(DnDgetrf);
CUSOLVER_ROUTINE_HANDLER(DnCgetrf);
CUSOLVER_ROUTINE_HANDLER(DnZgetrf);
CUSOLVER_ROUTINE_HANDLER(DnGetrf_bufferSize);
CUSOLVER_ROUTINE_HANDLER(DnGetrf);
CUSOLVER_ROUTINE_HANDLER(DnSgetrs);
CUSOLVER_ROUTINE_HANDLER(DnDgetrs);
CUSOLVER_ROUTINE_HANDLER(DnCgetrs);
CUSOLVER_ROUTINE_HANDLER(DnZgetrs);
CUSOLVER_ROUTINE_HANDLER(DnGetrs);
CUSOLVER_ROUTINE_HANDLER(DnZZgesv_bufferSize);
CUSOLVER_ROUTINE_HANDLER(DnZCgesv_bufferSize);
CUSOLVER_ROUTINE_HANDLER(DnZKgesv_bufferSize);
CUSOLVER_ROUTINE_HANDLER(DnZEgesv_bufferSize);
CUSOLVER_ROUTINE_HANDLER(DnZYgesv_bufferSize);
CUSOLVER_ROUTINE_HANDLER(DnCCgesv_bufferSize);
CUSOLVER_ROUTINE_HANDLER(DnCKgesv_bufferSize);
CUSOLVER_ROUTINE_HANDLER(DnCEgesv_bufferSize);
CUSOLVER_ROUTINE_HANDLER(DnCYgesv_bufferSize);
CUSOLVER_ROUTINE_HANDLER(DnDDgesv_bufferSize);
CUSOLVER_ROUTINE_HANDLER(DnDSgesv_bufferSize);
CUSOLVER_ROUTINE_HANDLER(DnDHgesv_bufferSize);
CUSOLVER_ROUTINE_HANDLER(DnDBgesv_bufferSize);
CUSOLVER_ROUTINE_HANDLER(DnDXgesv_bufferSize);
CUSOLVER_ROUTINE_HANDLER(DnSSgesv_bufferSize);
CUSOLVER_ROUTINE_HANDLER(DnSHgesv_bufferSize);
CUSOLVER_ROUTINE_HANDLER(DnSBgesv_bufferSize);
CUSOLVER_ROUTINE_HANDLER(DnSXgesv_bufferSize);
CUSOLVER_ROUTINE_HANDLER(DnZZgesv);
CUSOLVER_ROUTINE_HANDLER(DnZCgesv);
CUSOLVER_ROUTINE_HANDLER(DnZKgesv);
CUSOLVER_ROUTINE_HANDLER(DnZEgesv);
CUSOLVER_ROUTINE_HANDLER(DnZYgesv);
CUSOLVER_ROUTINE_HANDLER(DnCCgesv);
CUSOLVER_ROUTINE_HANDLER(DnCKgesv);
CUSOLVER_ROUTINE_HANDLER(DnCEgesv);
CUSOLVER_ROUTINE_HANDLER(DnCYgesv);
CUSOLVER_ROUTINE_HANDLER(DnDDgesv);
CUSOLVER_ROUTINE_HANDLER(DnDSgesv);
CUSOLVER_ROUTINE_HANDLER(DnDHgesv);
CUSOLVER_ROUTINE_HANDLER(DnDBgesv);
CUSOLVER_ROUTINE_HANDLER(DnDXgesv);
CUSOLVER_ROUTINE_HANDLER(DnSSgesv);
CUSOLVER_ROUTINE_HANDLER(DnSHgesv);
CUSOLVER_ROUTINE_HANDLER(DnSBgesv);
CUSOLVER_ROUTINE_HANDLER(DnSXgesv);
CUSOLVER_ROUTINE_HANDLER(DnIRSXgesv_bufferSize);
CUSOLVER_ROUTINE_HANDLER(DnIRSXgesv);
CUSOLVER_ROUTINE_HANDLER(DnSgeqrf_bufferSize);
CUSOLVER_ROUTINE_HANDLER(DnDgeqrf_bufferSize);
CUSOLVER_ROUTINE_HANDLER(DnCgeqrf_bufferSize);
CUSOLVER_ROUTINE_HANDLER(DnZgeqrf_bufferSize);
CUSOLVER_ROUTINE_HANDLER(DnSgeqrf);
CUSOLVER_ROUTINE_HANDLER(DnDgeqrf);
CUSOLVER_ROUTINE_HANDLER(DnCgeqrf);
CUSOLVER_ROUTINE_HANDLER(DnZgeqrf);
CUSOLVER_ROUTINE_HANDLER(DnGeqrf_bufferSize);
CUSOLVER_ROUTINE_HANDLER(DnGeqrf);
CUSOLVER_ROUTINE_HANDLER(DnZZgels_bufferSize);
CUSOLVER_ROUTINE_HANDLER(DnZCgels_bufferSize);
CUSOLVER_ROUTINE_HANDLER(DnZKgels_bufferSize);
CUSOLVER_ROUTINE_HANDLER(DnZEgels_bufferSize);
CUSOLVER_ROUTINE_HANDLER(DnZYgels_bufferSize);
CUSOLVER_ROUTINE_HANDLER(DnCCgels_bufferSize);
CUSOLVER_ROUTINE_HANDLER(DnCKgels_bufferSize);
CUSOLVER_ROUTINE_HANDLER(DnCEgels_bufferSize);
CUSOLVER_ROUTINE_HANDLER(DnCYgels_bufferSize);
CUSOLVER_ROUTINE_HANDLER(DnDDgels_bufferSize);
CUSOLVER_ROUTINE_HANDLER(DnDSgels_bufferSize);
CUSOLVER_ROUTINE_HANDLER(DnDHgels_bufferSize);
CUSOLVER_ROUTINE_HANDLER(DnDBgels_bufferSize);
CUSOLVER_ROUTINE_HANDLER(DnDXgels_bufferSize);
CUSOLVER_ROUTINE_HANDLER(DnSSgels_bufferSize);
CUSOLVER_ROUTINE_HANDLER(DnSHgels_bufferSize);
CUSOLVER_ROUTINE_HANDLER(DnSBgels_bufferSize);
CUSOLVER_ROUTINE_HANDLER(DnSXgels_bufferSize);
CUSOLVER_ROUTINE_HANDLER(DnZZgels);
CUSOLVER_ROUTINE_HANDLER(DnZCgels);
CUSOLVER_ROUTINE_HANDLER(DnZKgels);
CUSOLVER_ROUTINE_HANDLER(DnZEgels);
CUSOLVER_ROUTINE_HANDLER(DnZYgels);
CUSOLVER_ROUTINE_HANDLER(DnCCgels);
CUSOLVER_ROUTINE_HANDLER(DnCKgels);
CUSOLVER_ROUTINE_HANDLER(DnCEgels);
CUSOLVER_ROUTINE_HANDLER(DnCYgels);
CUSOLVER_ROUTINE_HANDLER(DnDDgels);
CUSOLVER_ROUTINE_HANDLER(DnDSgels);
CUSOLVER_ROUTINE_HANDLER(DnDHgels);
CUSOLVER_ROUTINE_HANDLER(DnDBgels);
CUSOLVER_ROUTINE_HANDLER(DnDXgels);
CUSOLVER_ROUTINE_HANDLER(DnSSgels);
CUSOLVER_ROUTINE_HANDLER(DnSHgels);
CUSOLVER_ROUTINE_HANDLER(DnSBgels);
CUSOLVER_ROUTINE_HANDLER(DnSXgels);
CUSOLVER_ROUTINE_HANDLER(DnIRSXgels_bufferSize);
CUSOLVER_ROUTINE_HANDLER(DnIRSXgels);
CUSOLVER_ROUTINE_HANDLER(DnSormqr_bufferSize);
CUSOLVER_ROUTINE_HANDLER(DnDormqr_bufferSize);
CUSOLVER_ROUTINE_HANDLER(DnCunmqr_bufferSize);
CUSOLVER_ROUTINE_HANDLER(DnZunmqr_bufferSize);
CUSOLVER_ROUTINE_HANDLER(DnSormqr);
CUSOLVER_ROUTINE_HANDLER(DnDormqr);
CUSOLVER_ROUTINE_HANDLER(DnCunmqr);
CUSOLVER_ROUTINE_HANDLER(DnZunmqr);
CUSOLVER_ROUTINE_HANDLER(DnSorgqr_bufferSize);
CUSOLVER_ROUTINE_HANDLER(DnDorgqr_bufferSize);
CUSOLVER_ROUTINE_HANDLER(DnCungqr_bufferSize);
CUSOLVER_ROUTINE_HANDLER(DnZungqr_bufferSize);
CUSOLVER_ROUTINE_HANDLER(DnSorgqr);
CUSOLVER_ROUTINE_HANDLER(DnDorgqr);
CUSOLVER_ROUTINE_HANDLER(DnCungqr);
CUSOLVER_ROUTINE_HANDLER(DnZungqr);
// DENSE LAPACK - DENSE EIGENVALUES SOLVER
// DENSE LAPACK - DENSE LINEAR SOLVER - 64-BIT
// SPARSE LAPACK - HELPER FUNCTION
// SPARSE LAPACK - HIGH LEVEL FUNCTION
// SPARSE LAPACK - LOW LEVEL FUNCTION
// REFACTORIZATION


#endif  /* CUSOLVERHANDLER_H */
