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

#include "Cusparse.h"

using namespace std;

extern "C" cusparseStatus_t CUSPARSEAPI cusparseSaxpyi(cusparseHandle_t handle, int nnz, const float* alpha, const float* xVal, const int* xInd, float* y, cusparseIndexBase_t idxBase) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CusparseFrontend::AddVariableForArguments<int>(nnz);
    CusparseFrontend::AddHostPointerForArguments(const_cast<float *>(alpha));
    CusparseFrontend::AddDevicePointerForArguments(xVal);
    CusparseFrontend::AddDevicePointerForArguments(xInd);
    CusparseFrontend::AddDevicePointerForArguments(y);
    CusparseFrontend::AddVariableForArguments<cusparseIndexBase_t>(idxBase);
    CusparseFrontend::Execute("cusparseSaxpyi");
    if (CusparseFrontend::Success()) {
        y = (float *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}
