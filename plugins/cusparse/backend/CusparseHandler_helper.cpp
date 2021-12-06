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
 *
 * Written by: Nicholas Piantadosi <nicholas.piantadosi@studenti.uniparthenope.it>,
 *             Department of Science and Technologies
 */

#include "CusparseHandler.h"

using namespace log4cplus;

using gvirtus::communicators::Buffer;
using gvirtus::communicators::Result;

CUSPARSE_ROUTINE_HANDLER(CreateColorInfo){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("CreateColorInfo"));
    cusparseColorInfo_t info;
    cusparseStatus_t cs = cusparseCreateColorInfo(&info);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->Add<cusparseColorInfo_t>(info);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(cs);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseCreateColorInfo Executed");
    return std::make_shared<Result>(cs,out);
}

#ifndef CUSPARSE_VERSION
#error CUSPARSE_VERSION not defined
#endif
