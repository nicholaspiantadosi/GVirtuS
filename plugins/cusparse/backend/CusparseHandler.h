/*
 * gVirtuS -- A GPGPU transparent virtualization component.
 *
 * Copyright (C) 2009-2010  The University of Napoli Parthenope at Naples.
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
 *                Department of Science and Technologies
 */

#ifndef CUSPARSEHANDLER_H
#define CUSPARSEHANDLER_H

#include <gvirtus/backend/Handler.h>
#include <gvirtus/communicators/Result.h>

#include <cusparse.h>

#include "log4cplus/logger.h"
#include "log4cplus/loggingmacros.h"
#include "log4cplus/configurator.h"

#include <limits.h>
#if ( __WORDSIZE == 64 )
    #define BUILD_64   1
#endif
using namespace std;
using namespace log4cplus;

using gvirtus::communicators::Buffer;
using gvirtus::communicators::Result;

class CusparseHandler : public gvirtus::backend::Handler {
public:
    CusparseHandler();
    virtual ~CusparseHandler();
    bool CanExecute(std::string routine);
    std::shared_ptr<Result> Execute(std::string routine, std::shared_ptr<Buffer> input_buffer);
    static void setLogLevel(Logger *logger);
private:
    log4cplus::Logger logger;
    void Initialize();
    typedef std::shared_ptr<Result> (*CusparseRoutineHandler)(CusparseHandler *, std::shared_ptr<Buffer>);
    static std::map<std::string, CusparseRoutineHandler> * mspHandlers;
};

#define CUSPARSE_ROUTINE_HANDLER(name) std::shared_ptr<Result> handle##name(CusparseHandler * pThis, std::shared_ptr<Buffer> in)
#define CUSPARSE_ROUTINE_HANDLER_PAIR(name) make_pair("cusparse" #name, handle##name)

// MANAGEMENT
CUSPARSE_ROUTINE_HANDLER(GetVersion);
CUSPARSE_ROUTINE_HANDLER(Create);
CUSPARSE_ROUTINE_HANDLER(Destroy);
CUSPARSE_ROUTINE_HANDLER(GetErrorString);
CUSPARSE_ROUTINE_HANDLER(SetStream);
CUSPARSE_ROUTINE_HANDLER(GetStream);
CUSPARSE_ROUTINE_HANDLER(GetProperty);
CUSPARSE_ROUTINE_HANDLER(GetPointerMode);
CUSPARSE_ROUTINE_HANDLER(SetPointerMode);
// HELPER
CUSPARSE_ROUTINE_HANDLER(CreateColorInfo);
CUSPARSE_ROUTINE_HANDLER(CreateMatDescr);
CUSPARSE_ROUTINE_HANDLER(DestroyColorInfo);
CUSPARSE_ROUTINE_HANDLER(DestroyMatDescr);
CUSPARSE_ROUTINE_HANDLER(GetMatDiagType);
CUSPARSE_ROUTINE_HANDLER(GetMatFillMode);
CUSPARSE_ROUTINE_HANDLER(GetMatIndexBase);
CUSPARSE_ROUTINE_HANDLER(GetMatType);
CUSPARSE_ROUTINE_HANDLER(SetMatDiagType);
CUSPARSE_ROUTINE_HANDLER(SetMatFillMode);
CUSPARSE_ROUTINE_HANDLER(SetMatIndexBase);
CUSPARSE_ROUTINE_HANDLER(SetMatType);

#endif  /* CUSPARSEHANDLER_H */
