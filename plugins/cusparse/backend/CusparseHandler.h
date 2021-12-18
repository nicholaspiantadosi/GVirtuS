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
CUSPARSE_ROUTINE_HANDLER(CreateCsrsv2Info);
CUSPARSE_ROUTINE_HANDLER(DestroyCsrsv2Info);
CUSPARSE_ROUTINE_HANDLER(CreateCsrsm2Info);
CUSPARSE_ROUTINE_HANDLER(DestroyCsrsm2Info);
CUSPARSE_ROUTINE_HANDLER(CreateCsric02Info);
CUSPARSE_ROUTINE_HANDLER(DestroyCsric02Info);
CUSPARSE_ROUTINE_HANDLER(CreateCsrilu02Info);
CUSPARSE_ROUTINE_HANDLER(DestroyCsrilu02Info);
CUSPARSE_ROUTINE_HANDLER(CreateBsrsv2Info);
CUSPARSE_ROUTINE_HANDLER(DestroyBsrsv2Info);
CUSPARSE_ROUTINE_HANDLER(CreateBsrsm2Info);
CUSPARSE_ROUTINE_HANDLER(DestroyBsrsm2Info);
CUSPARSE_ROUTINE_HANDLER(CreateBsric02Info);
CUSPARSE_ROUTINE_HANDLER(DestroyBsric02Info);
CUSPARSE_ROUTINE_HANDLER(CreateBsrilu02Info);
CUSPARSE_ROUTINE_HANDLER(DestroyBsrilu02Info);
CUSPARSE_ROUTINE_HANDLER(CreateCsrgemm2Info);
CUSPARSE_ROUTINE_HANDLER(DestroyCsrgemm2Info);
CUSPARSE_ROUTINE_HANDLER(CreatePruneInfo);
CUSPARSE_ROUTINE_HANDLER(DestroyPruneInfo);
// LEVEL1
CUSPARSE_ROUTINE_HANDLER(Saxpyi);
CUSPARSE_ROUTINE_HANDLER(Daxpyi);
CUSPARSE_ROUTINE_HANDLER(Caxpyi);
CUSPARSE_ROUTINE_HANDLER(Zaxpyi);
CUSPARSE_ROUTINE_HANDLER(Sgthr);
CUSPARSE_ROUTINE_HANDLER(Dgthr);
CUSPARSE_ROUTINE_HANDLER(Cgthr);
CUSPARSE_ROUTINE_HANDLER(Zgthr);
CUSPARSE_ROUTINE_HANDLER(Sgthrz);
CUSPARSE_ROUTINE_HANDLER(Dgthrz);
CUSPARSE_ROUTINE_HANDLER(Cgthrz);
CUSPARSE_ROUTINE_HANDLER(Zgthrz);
CUSPARSE_ROUTINE_HANDLER(Sroti);
CUSPARSE_ROUTINE_HANDLER(Droti);
CUSPARSE_ROUTINE_HANDLER(Ssctr);
CUSPARSE_ROUTINE_HANDLER(Dsctr);
CUSPARSE_ROUTINE_HANDLER(Csctr);
CUSPARSE_ROUTINE_HANDLER(Zsctr);
// LEVEL2
CUSPARSE_ROUTINE_HANDLER(Sbsrmv);
CUSPARSE_ROUTINE_HANDLER(Dbsrmv);
CUSPARSE_ROUTINE_HANDLER(Cbsrmv);
CUSPARSE_ROUTINE_HANDLER(Zbsrmv);
CUSPARSE_ROUTINE_HANDLER(Sbsrxmv);
CUSPARSE_ROUTINE_HANDLER(Dbsrxmv);
CUSPARSE_ROUTINE_HANDLER(Cbsrxmv);
CUSPARSE_ROUTINE_HANDLER(Zbsrxmv);
// LEVEL3
//FORMAT CONVERSION REFERENCE
CUSPARSE_ROUTINE_HANDLER(Xcsr2bsrNnz);
CUSPARSE_ROUTINE_HANDLER(Scsr2bsr);
CUSPARSE_ROUTINE_HANDLER(Dcsr2bsr);
CUSPARSE_ROUTINE_HANDLER(Ccsr2bsr);
CUSPARSE_ROUTINE_HANDLER(Zcsr2bsr);
CUSPARSE_ROUTINE_HANDLER(Sdense2csr);
CUSPARSE_ROUTINE_HANDLER(Ddense2csr);
CUSPARSE_ROUTINE_HANDLER(Cdense2csr);
CUSPARSE_ROUTINE_HANDLER(Zdense2csr);
CUSPARSE_ROUTINE_HANDLER(Snnz);
CUSPARSE_ROUTINE_HANDLER(Dnnz);
CUSPARSE_ROUTINE_HANDLER(Cnnz);
CUSPARSE_ROUTINE_HANDLER(Znnz);

#endif  /* CUSPARSEHANDLER_H */
