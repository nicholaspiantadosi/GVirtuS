
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
 *  Written by: Nicholas Piantadosi <nicholas.piantadosi@studenti.uniparthenope.it>,
 *  Department of Science and Technologies
 */

#ifndef CUSOLVERMGHANDLER_H
#define CUSOLVERMGHANDLER_H

#include <gvirtus/backend/Handler.h>
#include <gvirtus/communicators/Result.h>

#include <cusolverMg.h>
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

class CusolverMgHandler : public gvirtus::backend::Handler {
public:
    CusolverMgHandler();
    virtual ~CusolverMgHandler();
    bool CanExecute(std::string routine);
    std::shared_ptr<Result> Execute(std::string routine, std::shared_ptr<Buffer> input_buffer);
    static void setLogLevel(Logger *logger);
private:
    log4cplus::Logger logger;
    void Initialize();
    typedef std::shared_ptr<Result> (*CusolverMgRoutineHandler)(CusolverMgHandler *, std::shared_ptr<Buffer>);
    static std::map<std::string, CusolverMgRoutineHandler> * mspHandlers;
};

#define CUSOLVERMG_ROUTINE_HANDLER(name) std::shared_ptr<Result> handle##name(CusolverMgHandler * pThis, std::shared_ptr<Buffer> in)
#define CUSOLVERMG_ROUTINE_HANDLER_PAIR(name) make_pair("cusolver" #name, handle##name)

// MULTI GPU - HELPER FUNCTION
CUSOLVERMG_ROUTINE_HANDLER(MgCreate);
CUSOLVERMG_ROUTINE_HANDLER(MgDestroy);
CUSOLVERMG_ROUTINE_HANDLER(MgDeviceSelect);
CUSOLVERMG_ROUTINE_HANDLER(MgCreateDeviceGrid);
CUSOLVERMG_ROUTINE_HANDLER(MgDestroyGrid);
CUSOLVERMG_ROUTINE_HANDLER(MgCreateMatrixDesc);
CUSOLVERMG_ROUTINE_HANDLER(MgDestroyMatrixDesc);
// MULTI GPU - DENSE LINEAR SOLVER FUNCTION
CUSOLVERMG_ROUTINE_HANDLER(MgPotrf_bufferSize);
CUSOLVERMG_ROUTINE_HANDLER(MgPotrf);
CUSOLVERMG_ROUTINE_HANDLER(MgPotrs_bufferSize);
CUSOLVERMG_ROUTINE_HANDLER(MgPotrs);
CUSOLVERMG_ROUTINE_HANDLER(MgPotri_bufferSize);
CUSOLVERMG_ROUTINE_HANDLER(MgPotri);
CUSOLVERMG_ROUTINE_HANDLER(MgGetrf_bufferSize);
CUSOLVERMG_ROUTINE_HANDLER(MgGetrf);
CUSOLVERMG_ROUTINE_HANDLER(MgGetrs_bufferSize);
CUSOLVERMG_ROUTINE_HANDLER(MgGetrs);
// MULTI GPU - DENSE LINEAR EIGENVALUE FUNCTION
CUSOLVERMG_ROUTINE_HANDLER(MgSyevd_bufferSize);
CUSOLVERMG_ROUTINE_HANDLER(MgSyevd);

#endif  /* CUSOLVERMGHANDLER_H */
