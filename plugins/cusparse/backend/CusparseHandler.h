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
CUSPARSE_ROUTINE_HANDLER(Sbsrsv2_bufferSize);
CUSPARSE_ROUTINE_HANDLER(Dbsrsv2_bufferSize);
CUSPARSE_ROUTINE_HANDLER(Cbsrsv2_bufferSize);
CUSPARSE_ROUTINE_HANDLER(Zbsrsv2_bufferSize);
CUSPARSE_ROUTINE_HANDLER(Sbsrsv2_analysis);
CUSPARSE_ROUTINE_HANDLER(Dbsrsv2_analysis);
CUSPARSE_ROUTINE_HANDLER(Cbsrsv2_analysis);
CUSPARSE_ROUTINE_HANDLER(Zbsrsv2_analysis);
CUSPARSE_ROUTINE_HANDLER(Sbsrsv2_solve);
CUSPARSE_ROUTINE_HANDLER(Dbsrsv2_solve);
CUSPARSE_ROUTINE_HANDLER(Cbsrsv2_solve);
CUSPARSE_ROUTINE_HANDLER(Zbsrsv2_solve);
CUSPARSE_ROUTINE_HANDLER(Xbsrsv2_zeroPivot);
CUSPARSE_ROUTINE_HANDLER(CsrmvEx_bufferSize);
CUSPARSE_ROUTINE_HANDLER(CsrmvEx);
CUSPARSE_ROUTINE_HANDLER(Scsrsv2_bufferSize);
CUSPARSE_ROUTINE_HANDLER(Dcsrsv2_bufferSize);
CUSPARSE_ROUTINE_HANDLER(Ccsrsv2_bufferSize);
CUSPARSE_ROUTINE_HANDLER(Zcsrsv2_bufferSize);
CUSPARSE_ROUTINE_HANDLER(Scsrsv2_analysis);
CUSPARSE_ROUTINE_HANDLER(Dcsrsv2_analysis);
CUSPARSE_ROUTINE_HANDLER(Ccsrsv2_analysis);
CUSPARSE_ROUTINE_HANDLER(Zcsrsv2_analysis);
CUSPARSE_ROUTINE_HANDLER(Scsrsv2_solve);
CUSPARSE_ROUTINE_HANDLER(Dcsrsv2_solve);
CUSPARSE_ROUTINE_HANDLER(Ccsrsv2_solve);
CUSPARSE_ROUTINE_HANDLER(Zcsrsv2_solve);
CUSPARSE_ROUTINE_HANDLER(Xcsrsv2_zeroPivot);
CUSPARSE_ROUTINE_HANDLER(Sgemvi_bufferSize);
CUSPARSE_ROUTINE_HANDLER(Dgemvi_bufferSize);
CUSPARSE_ROUTINE_HANDLER(Cgemvi_bufferSize);
CUSPARSE_ROUTINE_HANDLER(Zgemvi_bufferSize);
CUSPARSE_ROUTINE_HANDLER(Sgemvi);
CUSPARSE_ROUTINE_HANDLER(Dgemvi);
CUSPARSE_ROUTINE_HANDLER(Cgemvi);
CUSPARSE_ROUTINE_HANDLER(Zgemvi);
// LEVEL3
CUSPARSE_ROUTINE_HANDLER(Sbsrmm);
CUSPARSE_ROUTINE_HANDLER(Dbsrmm);
CUSPARSE_ROUTINE_HANDLER(Cbsrmm);
CUSPARSE_ROUTINE_HANDLER(Zbsrmm);
CUSPARSE_ROUTINE_HANDLER(Sbsrsm2_bufferSize);
CUSPARSE_ROUTINE_HANDLER(Dbsrsm2_bufferSize);
CUSPARSE_ROUTINE_HANDLER(Cbsrsm2_bufferSize);
CUSPARSE_ROUTINE_HANDLER(Zbsrsm2_bufferSize);
CUSPARSE_ROUTINE_HANDLER(Sbsrsm2_analysis);
CUSPARSE_ROUTINE_HANDLER(Dbsrsm2_analysis);
CUSPARSE_ROUTINE_HANDLER(Cbsrsm2_analysis);
CUSPARSE_ROUTINE_HANDLER(Zbsrsm2_analysis);
CUSPARSE_ROUTINE_HANDLER(Sbsrsm2_solve);
CUSPARSE_ROUTINE_HANDLER(Dbsrsm2_solve);
CUSPARSE_ROUTINE_HANDLER(Cbsrsm2_solve);
CUSPARSE_ROUTINE_HANDLER(Zbsrsm2_solve);
CUSPARSE_ROUTINE_HANDLER(Xbsrsm2_zeroPivot);
CUSPARSE_ROUTINE_HANDLER(Scsrsm2_bufferSizeExt);
CUSPARSE_ROUTINE_HANDLER(Dcsrsm2_bufferSizeExt);
CUSPARSE_ROUTINE_HANDLER(Ccsrsm2_bufferSizeExt);
CUSPARSE_ROUTINE_HANDLER(Zcsrsm2_bufferSizeExt);
CUSPARSE_ROUTINE_HANDLER(Scsrsm2_analysis);
CUSPARSE_ROUTINE_HANDLER(Dcsrsm2_analysis);
CUSPARSE_ROUTINE_HANDLER(Ccsrsm2_analysis);
CUSPARSE_ROUTINE_HANDLER(Zcsrsm2_analysis);
CUSPARSE_ROUTINE_HANDLER(Scsrsm2_solve);
CUSPARSE_ROUTINE_HANDLER(Dcsrsm2_solve);
CUSPARSE_ROUTINE_HANDLER(Ccsrsm2_solve);
CUSPARSE_ROUTINE_HANDLER(Zcsrsm2_solve);
CUSPARSE_ROUTINE_HANDLER(Xcsrsm2_zeroPivot);
CUSPARSE_ROUTINE_HANDLER(Sgemmi);
CUSPARSE_ROUTINE_HANDLER(Dgemmi);
CUSPARSE_ROUTINE_HANDLER(Cgemmi);
CUSPARSE_ROUTINE_HANDLER(Zgemmi);
// EXTRA
CUSPARSE_ROUTINE_HANDLER(Scsrgeam2_bufferSizeExt);
CUSPARSE_ROUTINE_HANDLER(Dcsrgeam2_bufferSizeExt);
CUSPARSE_ROUTINE_HANDLER(Ccsrgeam2_bufferSizeExt);
CUSPARSE_ROUTINE_HANDLER(Zcsrgeam2_bufferSizeExt);
CUSPARSE_ROUTINE_HANDLER(Xcsrgeam2Nnz);
CUSPARSE_ROUTINE_HANDLER(Scsrgeam2);
CUSPARSE_ROUTINE_HANDLER(Dcsrgeam2);
CUSPARSE_ROUTINE_HANDLER(Ccsrgeam2);
CUSPARSE_ROUTINE_HANDLER(Zcsrgeam2);
CUSPARSE_ROUTINE_HANDLER(Scsrgemm2_bufferSizeExt);
CUSPARSE_ROUTINE_HANDLER(Dcsrgemm2_bufferSizeExt);
CUSPARSE_ROUTINE_HANDLER(Ccsrgemm2_bufferSizeExt);
CUSPARSE_ROUTINE_HANDLER(Zcsrgemm2_bufferSizeExt);
CUSPARSE_ROUTINE_HANDLER(Xcsrgemm2Nnz);
CUSPARSE_ROUTINE_HANDLER(Scsrgemm2);
CUSPARSE_ROUTINE_HANDLER(Dcsrgemm2);
CUSPARSE_ROUTINE_HANDLER(Ccsrgemm2);
CUSPARSE_ROUTINE_HANDLER(Zcsrgemm2);
// PRECONDITIONERS REFERENCE
CUSPARSE_ROUTINE_HANDLER(Scsric02_bufferSize);
CUSPARSE_ROUTINE_HANDLER(Dcsric02_bufferSize);
CUSPARSE_ROUTINE_HANDLER(Ccsric02_bufferSize);
CUSPARSE_ROUTINE_HANDLER(Zcsric02_bufferSize);
CUSPARSE_ROUTINE_HANDLER(Scsric02_analysis);
CUSPARSE_ROUTINE_HANDLER(Ccsric02_analysis);
CUSPARSE_ROUTINE_HANDLER(Dcsric02_analysis);
CUSPARSE_ROUTINE_HANDLER(Zcsric02_analysis);
CUSPARSE_ROUTINE_HANDLER(Scsric02);
CUSPARSE_ROUTINE_HANDLER(Dcsric02);
CUSPARSE_ROUTINE_HANDLER(Ccsric02);
CUSPARSE_ROUTINE_HANDLER(Zcsric02);
CUSPARSE_ROUTINE_HANDLER(Xcsric02_zeroPivot);
CUSPARSE_ROUTINE_HANDLER(Sbsric02_bufferSize);
CUSPARSE_ROUTINE_HANDLER(Dbsric02_bufferSize);
CUSPARSE_ROUTINE_HANDLER(Cbsric02_bufferSize);
CUSPARSE_ROUTINE_HANDLER(Zbsric02_bufferSize);
CUSPARSE_ROUTINE_HANDLER(Sbsric02_analysis);
CUSPARSE_ROUTINE_HANDLER(Dbsric02_analysis);
CUSPARSE_ROUTINE_HANDLER(Cbsric02_analysis);
CUSPARSE_ROUTINE_HANDLER(Zbsric02_analysis);
CUSPARSE_ROUTINE_HANDLER(Sbsric02);
CUSPARSE_ROUTINE_HANDLER(Dbsric02);
CUSPARSE_ROUTINE_HANDLER(Cbsric02);
CUSPARSE_ROUTINE_HANDLER(Zbsric02);
CUSPARSE_ROUTINE_HANDLER(Xbsric02_zeroPivot);
CUSPARSE_ROUTINE_HANDLER(Scsrilu02_numericBoost);
CUSPARSE_ROUTINE_HANDLER(Dcsrilu02_numericBoost);
CUSPARSE_ROUTINE_HANDLER(Ccsrilu02_numericBoost);
CUSPARSE_ROUTINE_HANDLER(Zcsrilu02_numericBoost);
CUSPARSE_ROUTINE_HANDLER(Scsrilu02_bufferSize);
CUSPARSE_ROUTINE_HANDLER(Dcsrilu02_bufferSize);
CUSPARSE_ROUTINE_HANDLER(Ccsrilu02_bufferSize);
CUSPARSE_ROUTINE_HANDLER(Zcsrilu02_bufferSize);
CUSPARSE_ROUTINE_HANDLER(Scsrilu02_analysis);
CUSPARSE_ROUTINE_HANDLER(Dcsrilu02_analysis);
CUSPARSE_ROUTINE_HANDLER(Ccsrilu02_analysis);
CUSPARSE_ROUTINE_HANDLER(Zcsrilu02_analysis);
CUSPARSE_ROUTINE_HANDLER(Scsrilu02);
CUSPARSE_ROUTINE_HANDLER(Dcsrilu02);
CUSPARSE_ROUTINE_HANDLER(Ccsrilu02);
CUSPARSE_ROUTINE_HANDLER(Zcsrilu02);
CUSPARSE_ROUTINE_HANDLER(Xcsrilu02_zeroPivot);
CUSPARSE_ROUTINE_HANDLER(Sbsrilu02_numericBoost);
CUSPARSE_ROUTINE_HANDLER(Dbsrilu02_numericBoost);
CUSPARSE_ROUTINE_HANDLER(Cbsrilu02_numericBoost);
CUSPARSE_ROUTINE_HANDLER(Zbsrilu02_numericBoost);
CUSPARSE_ROUTINE_HANDLER(Sbsrilu02_bufferSize);
CUSPARSE_ROUTINE_HANDLER(Dbsrilu02_bufferSize);
CUSPARSE_ROUTINE_HANDLER(Cbsrilu02_bufferSize);
CUSPARSE_ROUTINE_HANDLER(Zbsrilu02_bufferSize);
CUSPARSE_ROUTINE_HANDLER(Sbsrilu02_analysis);
CUSPARSE_ROUTINE_HANDLER(Dbsrilu02_analysis);
CUSPARSE_ROUTINE_HANDLER(Cbsrilu02_analysis);
CUSPARSE_ROUTINE_HANDLER(Zbsrilu02_analysis);
CUSPARSE_ROUTINE_HANDLER(Sbsrilu02);
CUSPARSE_ROUTINE_HANDLER(Dbsrilu02);
CUSPARSE_ROUTINE_HANDLER(Cbsrilu02);
CUSPARSE_ROUTINE_HANDLER(Zbsrilu02);
CUSPARSE_ROUTINE_HANDLER(Xbsrilu02_zeroPivot);
CUSPARSE_ROUTINE_HANDLER(Sgtsv2_bufferSizeExt);
CUSPARSE_ROUTINE_HANDLER(Dgtsv2_bufferSizeExt);
CUSPARSE_ROUTINE_HANDLER(Cgtsv2_bufferSizeExt);
CUSPARSE_ROUTINE_HANDLER(Zgtsv2_bufferSizeExt);
CUSPARSE_ROUTINE_HANDLER(Sgtsv2);
CUSPARSE_ROUTINE_HANDLER(Dgtsv2);
CUSPARSE_ROUTINE_HANDLER(Cgtsv2);
CUSPARSE_ROUTINE_HANDLER(Zgtsv2);
CUSPARSE_ROUTINE_HANDLER(Sgtsv2_nopivot_bufferSizeExt);
CUSPARSE_ROUTINE_HANDLER(Dgtsv2_nopivot_bufferSizeExt);
CUSPARSE_ROUTINE_HANDLER(Cgtsv2_nopivot_bufferSizeExt);
CUSPARSE_ROUTINE_HANDLER(Zgtsv2_nopivot_bufferSizeExt);
CUSPARSE_ROUTINE_HANDLER(Sgtsv2_nopivot);
CUSPARSE_ROUTINE_HANDLER(Dgtsv2_nopivot);
CUSPARSE_ROUTINE_HANDLER(Cgtsv2_nopivot);
CUSPARSE_ROUTINE_HANDLER(Zgtsv2_nopivot);
CUSPARSE_ROUTINE_HANDLER(Sgtsv2StridedBatch_bufferSizeExt);
CUSPARSE_ROUTINE_HANDLER(Dgtsv2StridedBatch_bufferSizeExt);
CUSPARSE_ROUTINE_HANDLER(Cgtsv2StridedBatch_bufferSizeExt);
CUSPARSE_ROUTINE_HANDLER(Zgtsv2StridedBatch_bufferSizeExt);
CUSPARSE_ROUTINE_HANDLER(Sgtsv2StridedBatch);
CUSPARSE_ROUTINE_HANDLER(Dgtsv2StridedBatch);
CUSPARSE_ROUTINE_HANDLER(Cgtsv2StridedBatch);
CUSPARSE_ROUTINE_HANDLER(Zgtsv2StridedBatch);
CUSPARSE_ROUTINE_HANDLER(SgtsvInterleavedBatch_bufferSizeExt);
CUSPARSE_ROUTINE_HANDLER(DgtsvInterleavedBatch_bufferSizeExt);
CUSPARSE_ROUTINE_HANDLER(CgtsvInterleavedBatch_bufferSizeExt);
CUSPARSE_ROUTINE_HANDLER(ZgtsvInterleavedBatch_bufferSizeExt);
CUSPARSE_ROUTINE_HANDLER(SgtsvInterleavedBatch);
CUSPARSE_ROUTINE_HANDLER(DgtsvInterleavedBatch);
CUSPARSE_ROUTINE_HANDLER(CgtsvInterleavedBatch);
CUSPARSE_ROUTINE_HANDLER(ZgtsvInterleavedBatch);
CUSPARSE_ROUTINE_HANDLER(SgpsvInterleavedBatch_bufferSizeExt);
CUSPARSE_ROUTINE_HANDLER(DgpsvInterleavedBatch_bufferSizeExt);
CUSPARSE_ROUTINE_HANDLER(CgpsvInterleavedBatch_bufferSizeExt);
CUSPARSE_ROUTINE_HANDLER(ZgpsvInterleavedBatch_bufferSizeExt);
CUSPARSE_ROUTINE_HANDLER(SgpsvInterleavedBatch);
CUSPARSE_ROUTINE_HANDLER(DgpsvInterleavedBatch);
CUSPARSE_ROUTINE_HANDLER(CgpsvInterleavedBatch);
CUSPARSE_ROUTINE_HANDLER(ZgpsvInterleavedBatch);
// REORDERINGS REFERENCE
CUSPARSE_ROUTINE_HANDLER(Scsrcolor);
CUSPARSE_ROUTINE_HANDLER(Dcsrcolor);
CUSPARSE_ROUTINE_HANDLER(Ccsrcolor);
CUSPARSE_ROUTINE_HANDLER(Zcsrcolor);
// FORMAT CONVERSION REFERENCE
CUSPARSE_ROUTINE_HANDLER(Sbsr2csr);
CUSPARSE_ROUTINE_HANDLER(Dbsr2csr);
CUSPARSE_ROUTINE_HANDLER(Cbsr2csr);
CUSPARSE_ROUTINE_HANDLER(Zbsr2csr);
CUSPARSE_ROUTINE_HANDLER(Sgebsr2gebsc_bufferSize);
CUSPARSE_ROUTINE_HANDLER(Dgebsr2gebsc_bufferSize);
CUSPARSE_ROUTINE_HANDLER(Cgebsr2gebsc_bufferSize);
CUSPARSE_ROUTINE_HANDLER(Zgebsr2gebsc_bufferSize);
CUSPARSE_ROUTINE_HANDLER(Sgebsr2gebsc);
CUSPARSE_ROUTINE_HANDLER(Dgebsr2gebsc);
CUSPARSE_ROUTINE_HANDLER(Cgebsr2gebsc);
CUSPARSE_ROUTINE_HANDLER(Zgebsr2gebsc);
CUSPARSE_ROUTINE_HANDLER(Sgebsr2gebsr_bufferSize);
CUSPARSE_ROUTINE_HANDLER(Dgebsr2gebsr_bufferSize);
CUSPARSE_ROUTINE_HANDLER(Cgebsr2gebsr_bufferSize);
CUSPARSE_ROUTINE_HANDLER(Zgebsr2gebsr_bufferSize);
CUSPARSE_ROUTINE_HANDLER(Xgebsr2gebsrNnz);
CUSPARSE_ROUTINE_HANDLER(Sgebsr2gebsr);
CUSPARSE_ROUTINE_HANDLER(Dgebsr2gebsr);
CUSPARSE_ROUTINE_HANDLER(Cgebsr2gebsr);
CUSPARSE_ROUTINE_HANDLER(Zgebsr2gebsr);
CUSPARSE_ROUTINE_HANDLER(Sgebsr2csr);
CUSPARSE_ROUTINE_HANDLER(Dgebsr2csr);
CUSPARSE_ROUTINE_HANDLER(Cgebsr2csr);
CUSPARSE_ROUTINE_HANDLER(Zgebsr2csr);
CUSPARSE_ROUTINE_HANDLER(Scsr2gebsr_bufferSize);
CUSPARSE_ROUTINE_HANDLER(Dcsr2gebsr_bufferSize);
CUSPARSE_ROUTINE_HANDLER(Ccsr2gebsr_bufferSize);
CUSPARSE_ROUTINE_HANDLER(Zcsr2gebsr_bufferSize);
CUSPARSE_ROUTINE_HANDLER(Xcsr2gebsrNnz);
CUSPARSE_ROUTINE_HANDLER(Scsr2gebsr);
CUSPARSE_ROUTINE_HANDLER(Dcsr2gebsr);
CUSPARSE_ROUTINE_HANDLER(Ccsr2gebsr);
CUSPARSE_ROUTINE_HANDLER(Zcsr2gebsr);
CUSPARSE_ROUTINE_HANDLER(Xcoo2csr);
CUSPARSE_ROUTINE_HANDLER(Scsc2dense);
CUSPARSE_ROUTINE_HANDLER(Dcsc2dense);
CUSPARSE_ROUTINE_HANDLER(Ccsc2dense);
CUSPARSE_ROUTINE_HANDLER(Zcsc2dense);
CUSPARSE_ROUTINE_HANDLER(Xcsr2bsrNnz);
CUSPARSE_ROUTINE_HANDLER(Scsr2bsr);
CUSPARSE_ROUTINE_HANDLER(Dcsr2bsr);
CUSPARSE_ROUTINE_HANDLER(Ccsr2bsr);
CUSPARSE_ROUTINE_HANDLER(Zcsr2bsr);
CUSPARSE_ROUTINE_HANDLER(Xcsr2coo);
CUSPARSE_ROUTINE_HANDLER(Csr2cscEx2_bufferSize);
CUSPARSE_ROUTINE_HANDLER(Csr2cscEx2);
CUSPARSE_ROUTINE_HANDLER(Scsr2dense);
CUSPARSE_ROUTINE_HANDLER(Dcsr2dense);
CUSPARSE_ROUTINE_HANDLER(Ccsr2dense);
CUSPARSE_ROUTINE_HANDLER(Zcsr2dense);
CUSPARSE_ROUTINE_HANDLER(Scsr2csr_compress);
CUSPARSE_ROUTINE_HANDLER(Dcsr2csr_compress);
CUSPARSE_ROUTINE_HANDLER(Ccsr2csr_compress);
CUSPARSE_ROUTINE_HANDLER(Zcsr2csr_compress);
CUSPARSE_ROUTINE_HANDLER(Sdense2csc);
CUSPARSE_ROUTINE_HANDLER(Ddense2csc);
CUSPARSE_ROUTINE_HANDLER(Cdense2csc);
CUSPARSE_ROUTINE_HANDLER(Zdense2csc);
CUSPARSE_ROUTINE_HANDLER(Sdense2csr);
CUSPARSE_ROUTINE_HANDLER(Ddense2csr);
CUSPARSE_ROUTINE_HANDLER(Cdense2csr);
CUSPARSE_ROUTINE_HANDLER(Zdense2csr);
CUSPARSE_ROUTINE_HANDLER(Snnz);
CUSPARSE_ROUTINE_HANDLER(Dnnz);
CUSPARSE_ROUTINE_HANDLER(Cnnz);
CUSPARSE_ROUTINE_HANDLER(Znnz);
CUSPARSE_ROUTINE_HANDLER(CreateIdentityPermutation);
CUSPARSE_ROUTINE_HANDLER(Xcoosort_bufferSizeExt);
CUSPARSE_ROUTINE_HANDLER(XcoosortByRow);
CUSPARSE_ROUTINE_HANDLER(XcoosortByColumn);
CUSPARSE_ROUTINE_HANDLER(Snnz_compress);
CUSPARSE_ROUTINE_HANDLER(Dnnz_compress);
CUSPARSE_ROUTINE_HANDLER(Cnnz_compress);
CUSPARSE_ROUTINE_HANDLER(Znnz_compress);
// GENERIC API REFERENCE - SPARSE MATRIX API
CUSPARSE_ROUTINE_HANDLER(CreateCsr);
CUSPARSE_ROUTINE_HANDLER(DestroySpMat);
// GENERIC API REFERENCE - DENSE VECTOR API
CUSPARSE_ROUTINE_HANDLER(CreateDnVec);
CUSPARSE_ROUTINE_HANDLER(DestroyDnVec);
// GENERIC API REFERENCE - GENERIC API FUNCTIONS
CUSPARSE_ROUTINE_HANDLER(SpMV_bufferSize);
CUSPARSE_ROUTINE_HANDLER(SpMV);

#endif  /* CUSPARSEHANDLER_H */
