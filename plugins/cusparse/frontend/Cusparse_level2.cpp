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
 * Department of Science and Technologies
 */

#include "Cusparse.h"

using namespace std;

extern "C" cusparseStatus_t CUSPARSEAPI cusparseSbsrmv(cusparseHandle_t handle, cusparseDirection_t dir, cusparseOperation_t trans, int mb, int nb, int nnzb, const float* alpha, const cusparseMatDescr_t descr, const float* bsrVal, const int* bsrRowPtr, const int* bsrColInd, int blockDim, const float* x, const float* beta, float* y) {
  CusparseFrontend::Prepare();
  CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
  CusparseFrontend::AddVariableForArguments<cusparseDirection_t>(dir);
  CusparseFrontend::AddVariableForArguments<cusparseOperation_t>(trans);
  CusparseFrontend::AddVariableForArguments<int>(mb);
  CusparseFrontend::AddVariableForArguments<int>(nb);
  CusparseFrontend::AddVariableForArguments<int>(nnzb);
  CusparseFrontend::AddHostPointerForArguments(const_cast<float *>(alpha));
  CusparseFrontend::AddVariableForArguments<size_t>((size_t)descr);
  CusparseFrontend::AddDevicePointerForArguments(bsrVal);
  CusparseFrontend::AddDevicePointerForArguments(bsrRowPtr);
  CusparseFrontend::AddDevicePointerForArguments(bsrColInd);
  CusparseFrontend::AddVariableForArguments<int>(blockDim);
  CusparseFrontend::AddDevicePointerForArguments(x);
  CusparseFrontend::AddHostPointerForArguments(const_cast<float *>(beta));
  CusparseFrontend::AddDevicePointerForArguments(y);
  CusparseFrontend::Execute("cusparseSbsrmv");
  if (CusparseFrontend::Success()) {
  y = (float *)CusparseFrontend::GetOutputDevicePointer();
  }
  return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseDbsrmv(cusparseHandle_t handle, cusparseDirection_t dir, cusparseOperation_t trans, int mb, int nb, int nnzb, const double* alpha, const cusparseMatDescr_t descr, const double* bsrVal, const int* bsrRowPtr, const int* bsrColInd, int blockDim, const double* x, const double* beta, double* y) {
  CusparseFrontend::Prepare();
  CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
  CusparseFrontend::AddVariableForArguments<cusparseDirection_t>(dir);
  CusparseFrontend::AddVariableForArguments<cusparseOperation_t>(trans);
  CusparseFrontend::AddVariableForArguments<int>(mb);
  CusparseFrontend::AddVariableForArguments<int>(nb);
  CusparseFrontend::AddVariableForArguments<int>(nnzb);
  CusparseFrontend::AddHostPointerForArguments(const_cast<double *>(alpha));
  CusparseFrontend::AddVariableForArguments<size_t>((size_t)descr);
  CusparseFrontend::AddDevicePointerForArguments(bsrVal);
  CusparseFrontend::AddDevicePointerForArguments(bsrRowPtr);
  CusparseFrontend::AddDevicePointerForArguments(bsrColInd);
  CusparseFrontend::AddVariableForArguments<int>(blockDim);
  CusparseFrontend::AddDevicePointerForArguments(x);
  CusparseFrontend::AddHostPointerForArguments(const_cast<double *>(beta));
  CusparseFrontend::AddDevicePointerForArguments(y);
  CusparseFrontend::Execute("cusparseDbsrmv");
  if (CusparseFrontend::Success()) {
  y = (double *)CusparseFrontend::GetOutputDevicePointer();
  }
  return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseCbsrmv(cusparseHandle_t handle, cusparseDirection_t dir, cusparseOperation_t trans, int mb, int nb, int nnzb, const cuComplex* alpha, const cusparseMatDescr_t descr, const cuComplex* bsrVal, const int* bsrRowPtr, const int* bsrColInd, int blockDim, const cuComplex* x, const cuComplex* beta, cuComplex* y) {
  CusparseFrontend::Prepare();
  CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
  CusparseFrontend::AddVariableForArguments<cusparseDirection_t>(dir);
  CusparseFrontend::AddVariableForArguments<cusparseOperation_t>(trans);
  CusparseFrontend::AddVariableForArguments<int>(mb);
  CusparseFrontend::AddVariableForArguments<int>(nb);
  CusparseFrontend::AddVariableForArguments<int>(nnzb);
  CusparseFrontend::AddHostPointerForArguments(const_cast<cuComplex *>(alpha));
  CusparseFrontend::AddVariableForArguments<size_t>((size_t)descr);
  CusparseFrontend::AddDevicePointerForArguments(bsrVal);
  CusparseFrontend::AddDevicePointerForArguments(bsrRowPtr);
  CusparseFrontend::AddDevicePointerForArguments(bsrColInd);
  CusparseFrontend::AddVariableForArguments<int>(blockDim);
  CusparseFrontend::AddDevicePointerForArguments(x);
  CusparseFrontend::AddHostPointerForArguments(const_cast<cuComplex *>(beta));
  CusparseFrontend::AddDevicePointerForArguments(y);
  CusparseFrontend::Execute("cusparseCbsrmv");
  if (CusparseFrontend::Success()) {
  y = (cuComplex *)CusparseFrontend::GetOutputDevicePointer();
  }
  return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseZbsrmv(cusparseHandle_t handle, cusparseDirection_t dir, cusparseOperation_t trans, int mb, int nb, int nnzb, const cuDoubleComplex* alpha, const cusparseMatDescr_t descr, const cuDoubleComplex* bsrVal, const int* bsrRowPtr, const int* bsrColInd, int blockDim, const cuDoubleComplex* x, const cuDoubleComplex* beta, cuDoubleComplex* y) {
  CusparseFrontend::Prepare();
  CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
  CusparseFrontend::AddVariableForArguments<cusparseDirection_t>(dir);
  CusparseFrontend::AddVariableForArguments<cusparseOperation_t>(trans);
  CusparseFrontend::AddVariableForArguments<int>(mb);
  CusparseFrontend::AddVariableForArguments<int>(nb);
  CusparseFrontend::AddVariableForArguments<int>(nnzb);
  CusparseFrontend::AddHostPointerForArguments(const_cast<cuDoubleComplex *>(alpha));
  CusparseFrontend::AddVariableForArguments<size_t>((size_t)descr);
  CusparseFrontend::AddDevicePointerForArguments(bsrVal);
  CusparseFrontend::AddDevicePointerForArguments(bsrRowPtr);
  CusparseFrontend::AddDevicePointerForArguments(bsrColInd);
  CusparseFrontend::AddVariableForArguments<int>(blockDim);
  CusparseFrontend::AddDevicePointerForArguments(x);
  CusparseFrontend::AddHostPointerForArguments(const_cast<cuDoubleComplex *>(beta));
  CusparseFrontend::AddDevicePointerForArguments(y);
  CusparseFrontend::Execute("cusparseZbsrmv");
  if (CusparseFrontend::Success()) {
  y = (cuDoubleComplex *)CusparseFrontend::GetOutputDevicePointer();
  }
  return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseSbsrxmv(cusparseHandle_t handle, cusparseDirection_t dir, cusparseOperation_t trans, int sizeOfMask, int mb, int nb, int nnzb, const float* alpha, const cusparseMatDescr_t descr, const float* bsrVal, const int* bsrMaskPtr, const int* bsrRowPtr, const int* bsrRowPtrEnd, const int* bsrColInd, int blockDim, const float* x, const float* beta, float* y) {
  CusparseFrontend::Prepare();
  CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
  CusparseFrontend::AddVariableForArguments<cusparseDirection_t>(dir);
  CusparseFrontend::AddVariableForArguments<cusparseOperation_t>(trans);
  CusparseFrontend::AddVariableForArguments<int>(sizeOfMask);
  CusparseFrontend::AddVariableForArguments<int>(mb);
  CusparseFrontend::AddVariableForArguments<int>(nb);
  CusparseFrontend::AddVariableForArguments<int>(nnzb);
  CusparseFrontend::AddHostPointerForArguments(const_cast<float *>(alpha));
  CusparseFrontend::AddVariableForArguments<size_t>((size_t)descr);
  CusparseFrontend::AddDevicePointerForArguments(bsrVal);
  CusparseFrontend::AddDevicePointerForArguments(bsrMaskPtr);
  CusparseFrontend::AddDevicePointerForArguments(bsrRowPtr);
  CusparseFrontend::AddDevicePointerForArguments(bsrRowPtrEnd);
  CusparseFrontend::AddDevicePointerForArguments(bsrColInd);
  CusparseFrontend::AddVariableForArguments<int>(blockDim);
  CusparseFrontend::AddDevicePointerForArguments(x);
  CusparseFrontend::AddHostPointerForArguments(const_cast<float *>(beta));
  CusparseFrontend::AddDevicePointerForArguments(y);
  CusparseFrontend::Execute("cusparseSbsrxmv");
  if (CusparseFrontend::Success()) {
  y = (float *)CusparseFrontend::GetOutputDevicePointer();
  }
  return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseDbsrxmv(cusparseHandle_t handle, cusparseDirection_t dir, cusparseOperation_t trans, int sizeOfMask, int mb, int nb, int nnzb, const double* alpha, const cusparseMatDescr_t descr, const double* bsrVal, const int* bsrMaskPtr, const int* bsrRowPtr, const int* bsrRowPtrEnd, const int* bsrColInd, int blockDim, const double* x, const double* beta, double* y) {
  CusparseFrontend::Prepare();
  CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
  CusparseFrontend::AddVariableForArguments<cusparseDirection_t>(dir);
  CusparseFrontend::AddVariableForArguments<cusparseOperation_t>(trans);
  CusparseFrontend::AddVariableForArguments<int>(sizeOfMask);
  CusparseFrontend::AddVariableForArguments<int>(mb);
  CusparseFrontend::AddVariableForArguments<int>(nb);
  CusparseFrontend::AddVariableForArguments<int>(nnzb);
  CusparseFrontend::AddHostPointerForArguments(const_cast<double *>(alpha));
  CusparseFrontend::AddVariableForArguments<size_t>((size_t)descr);
  CusparseFrontend::AddDevicePointerForArguments(bsrVal);
  CusparseFrontend::AddDevicePointerForArguments(bsrMaskPtr);
  CusparseFrontend::AddDevicePointerForArguments(bsrRowPtr);
  CusparseFrontend::AddDevicePointerForArguments(bsrRowPtrEnd);
  CusparseFrontend::AddDevicePointerForArguments(bsrColInd);
  CusparseFrontend::AddVariableForArguments<int>(blockDim);
  CusparseFrontend::AddDevicePointerForArguments(x);
  CusparseFrontend::AddHostPointerForArguments(const_cast<double *>(beta));
  CusparseFrontend::AddDevicePointerForArguments(y);
  CusparseFrontend::Execute("cusparseDbsrxmv");
  if (CusparseFrontend::Success()) {
  y = (double *)CusparseFrontend::GetOutputDevicePointer();
  }
  return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseCbsrxmv(cusparseHandle_t handle, cusparseDirection_t dir, cusparseOperation_t trans, int sizeOfMask, int mb, int nb, int nnzb, const cuComplex* alpha, const cusparseMatDescr_t descr, const cuComplex* bsrVal, const int* bsrMaskPtr, const int* bsrRowPtr, const int* bsrRowPtrEnd, const int* bsrColInd, int blockDim, const cuComplex* x, const cuComplex* beta, cuComplex* y) {
  CusparseFrontend::Prepare();
  CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
  CusparseFrontend::AddVariableForArguments<cusparseDirection_t>(dir);
  CusparseFrontend::AddVariableForArguments<cusparseOperation_t>(trans);
  CusparseFrontend::AddVariableForArguments<int>(sizeOfMask);
  CusparseFrontend::AddVariableForArguments<int>(mb);
  CusparseFrontend::AddVariableForArguments<int>(nb);
  CusparseFrontend::AddVariableForArguments<int>(nnzb);
  CusparseFrontend::AddHostPointerForArguments(const_cast<cuComplex *>(alpha));
  CusparseFrontend::AddVariableForArguments<size_t>((size_t)descr);
  CusparseFrontend::AddDevicePointerForArguments(bsrVal);
  CusparseFrontend::AddDevicePointerForArguments(bsrMaskPtr);
  CusparseFrontend::AddDevicePointerForArguments(bsrRowPtr);
  CusparseFrontend::AddDevicePointerForArguments(bsrRowPtrEnd);
  CusparseFrontend::AddDevicePointerForArguments(bsrColInd);
  CusparseFrontend::AddVariableForArguments<int>(blockDim);
  CusparseFrontend::AddDevicePointerForArguments(x);
  CusparseFrontend::AddHostPointerForArguments(const_cast<cuComplex *>(beta));
  CusparseFrontend::AddDevicePointerForArguments(y);
  CusparseFrontend::Execute("cusparseCbsrxmv");
  if (CusparseFrontend::Success()) {
  y = (cuComplex *)CusparseFrontend::GetOutputDevicePointer();
  }
  return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseZbsrxmv(cusparseHandle_t handle, cusparseDirection_t dir, cusparseOperation_t trans, int sizeOfMask, int mb, int nb, int nnzb, const cuDoubleComplex* alpha, const cusparseMatDescr_t descr, const cuDoubleComplex* bsrVal, const int* bsrMaskPtr, const int* bsrRowPtr, const int* bsrRowPtrEnd, const int* bsrColInd, int blockDim, const cuDoubleComplex* x, const cuDoubleComplex* beta, cuDoubleComplex* y) {
  CusparseFrontend::Prepare();
  CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
  CusparseFrontend::AddVariableForArguments<cusparseDirection_t>(dir);
  CusparseFrontend::AddVariableForArguments<cusparseOperation_t>(trans);
  CusparseFrontend::AddVariableForArguments<int>(sizeOfMask);
  CusparseFrontend::AddVariableForArguments<int>(mb);
  CusparseFrontend::AddVariableForArguments<int>(nb);
  CusparseFrontend::AddVariableForArguments<int>(nnzb);
  CusparseFrontend::AddHostPointerForArguments(const_cast<cuDoubleComplex *>(alpha));
  CusparseFrontend::AddVariableForArguments<size_t>((size_t)descr);
  CusparseFrontend::AddDevicePointerForArguments(bsrVal);
  CusparseFrontend::AddDevicePointerForArguments(bsrMaskPtr);
  CusparseFrontend::AddDevicePointerForArguments(bsrRowPtr);
  CusparseFrontend::AddDevicePointerForArguments(bsrRowPtrEnd);
  CusparseFrontend::AddDevicePointerForArguments(bsrColInd);
  CusparseFrontend::AddVariableForArguments<int>(blockDim);
  CusparseFrontend::AddDevicePointerForArguments(x);
  CusparseFrontend::AddHostPointerForArguments(const_cast<cuDoubleComplex *>(beta));
  CusparseFrontend::AddDevicePointerForArguments(y);
  CusparseFrontend::Execute("cusparseZbsrxmv");
  if (CusparseFrontend::Success()) {
  y = (cuDoubleComplex *)CusparseFrontend::GetOutputDevicePointer();
  }
  return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseSbsrsv2_bufferSize(cusparseHandle_t handle, cusparseDirection_t dir, cusparseOperation_t trans, int mb, int nnzb, const cusparseMatDescr_t descr, float* bsrVal, const int* bsrRowPtr, const int* bsrColInd, int blockDim, bsrsv2Info_t info, int* pBufferSizeInBytes) {
  CusparseFrontend::Prepare();
  CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
  CusparseFrontend::AddVariableForArguments<cusparseDirection_t>(dir);
  CusparseFrontend::AddVariableForArguments<cusparseOperation_t>(trans);
  CusparseFrontend::AddVariableForArguments<int>(mb);
  CusparseFrontend::AddVariableForArguments<int>(nnzb);
  CusparseFrontend::AddVariableForArguments<size_t>((size_t)descr);
  CusparseFrontend::AddDevicePointerForArguments(bsrVal);
  CusparseFrontend::AddDevicePointerForArguments(bsrRowPtr);
  CusparseFrontend::AddDevicePointerForArguments(bsrColInd);
  CusparseFrontend::AddVariableForArguments<int>(blockDim);
  CusparseFrontend::AddVariableForArguments<size_t>((size_t) info);
  CusparseFrontend::AddDevicePointerForArguments(pBufferSizeInBytes);
  CusparseFrontend::Execute("cusparseSbsrsv2_bufferSize");
  if (CusparseFrontend::Success()) {
  info = CusparseFrontend::GetOutputVariable<bsrsv2Info_t>();
  pBufferSizeInBytes = (int *)CusparseFrontend::GetOutputDevicePointer();
  }
  return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseDbsrsv2_bufferSize(cusparseHandle_t handle, cusparseDirection_t dir, cusparseOperation_t trans, int mb, int nnzb, const cusparseMatDescr_t descr, double* bsrVal, const int* bsrRowPtr, const int* bsrColInd, int blockDim, bsrsv2Info_t info, int* pBufferSizeInBytes) {
  CusparseFrontend::Prepare();
  CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
  CusparseFrontend::AddVariableForArguments<cusparseDirection_t>(dir);
  CusparseFrontend::AddVariableForArguments<cusparseOperation_t>(trans);
  CusparseFrontend::AddVariableForArguments<int>(mb);
  CusparseFrontend::AddVariableForArguments<int>(nnzb);
  CusparseFrontend::AddVariableForArguments<size_t>((size_t)descr);
  CusparseFrontend::AddDevicePointerForArguments(bsrVal);
  CusparseFrontend::AddDevicePointerForArguments(bsrRowPtr);
  CusparseFrontend::AddDevicePointerForArguments(bsrColInd);
  CusparseFrontend::AddVariableForArguments<int>(blockDim);
  CusparseFrontend::AddVariableForArguments<size_t>((size_t) info);
  CusparseFrontend::AddDevicePointerForArguments(pBufferSizeInBytes);
  CusparseFrontend::Execute("cusparseDbsrsv2_bufferSize");
  if (CusparseFrontend::Success()) {
  info = CusparseFrontend::GetOutputVariable<bsrsv2Info_t>();
  pBufferSizeInBytes = (int *)CusparseFrontend::GetOutputDevicePointer();
  }
  return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseCbsrsv2_bufferSize(cusparseHandle_t handle, cusparseDirection_t dir, cusparseOperation_t trans, int mb, int nnzb, const cusparseMatDescr_t descr, cuComplex* bsrVal, const int* bsrRowPtr, const int* bsrColInd, int blockDim, bsrsv2Info_t info, int* pBufferSizeInBytes) {
  CusparseFrontend::Prepare();
  CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
  CusparseFrontend::AddVariableForArguments<cusparseDirection_t>(dir);
  CusparseFrontend::AddVariableForArguments<cusparseOperation_t>(trans);
  CusparseFrontend::AddVariableForArguments<int>(mb);
  CusparseFrontend::AddVariableForArguments<int>(nnzb);
  CusparseFrontend::AddVariableForArguments<size_t>((size_t)descr);
  CusparseFrontend::AddDevicePointerForArguments(bsrVal);
  CusparseFrontend::AddDevicePointerForArguments(bsrRowPtr);
  CusparseFrontend::AddDevicePointerForArguments(bsrColInd);
  CusparseFrontend::AddVariableForArguments<int>(blockDim);
  CusparseFrontend::AddVariableForArguments<size_t>((size_t) info);
  CusparseFrontend::AddDevicePointerForArguments(pBufferSizeInBytes);
  CusparseFrontend::Execute("cusparseCbsrsv2_bufferSize");
  if (CusparseFrontend::Success()) {
  info = CusparseFrontend::GetOutputVariable<bsrsv2Info_t>();
  pBufferSizeInBytes = (int *)CusparseFrontend::GetOutputDevicePointer();
  }
  return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseZbsrsv2_bufferSize(cusparseHandle_t handle, cusparseDirection_t dir, cusparseOperation_t trans, int mb, int nnzb, const cusparseMatDescr_t descr, cuDoubleComplex* bsrVal, const int* bsrRowPtr, const int* bsrColInd, int blockDim, bsrsv2Info_t info, int* pBufferSizeInBytes) {
  CusparseFrontend::Prepare();
  CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
  CusparseFrontend::AddVariableForArguments<cusparseDirection_t>(dir);
  CusparseFrontend::AddVariableForArguments<cusparseOperation_t>(trans);
  CusparseFrontend::AddVariableForArguments<int>(mb);
  CusparseFrontend::AddVariableForArguments<int>(nnzb);
  CusparseFrontend::AddVariableForArguments<size_t>((size_t)descr);
  CusparseFrontend::AddDevicePointerForArguments(bsrVal);
  CusparseFrontend::AddDevicePointerForArguments(bsrRowPtr);
  CusparseFrontend::AddDevicePointerForArguments(bsrColInd);
  CusparseFrontend::AddVariableForArguments<int>(blockDim);
  CusparseFrontend::AddVariableForArguments<size_t>((size_t) info);
  CusparseFrontend::AddDevicePointerForArguments(pBufferSizeInBytes);
  CusparseFrontend::Execute("cusparseZbsrsv2_bufferSize");
  if (CusparseFrontend::Success()) {
  info = CusparseFrontend::GetOutputVariable<bsrsv2Info_t>();
  pBufferSizeInBytes = (int *)CusparseFrontend::GetOutputDevicePointer();
  }
  return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseSbsrsv2_analysis(cusparseHandle_t handle, cusparseDirection_t dir, cusparseOperation_t trans, int mb, int nnzb, cusparseMatDescr_t descr, const float* bsrVal, const int* bsrRowPtr, const int* bsrColInd, int blockDim, bsrsv2Info_t info, cusparseSolvePolicy_t policy, void* pBuffer) {
  CusparseFrontend::Prepare();
  CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
  CusparseFrontend::AddVariableForArguments<cusparseDirection_t>(dir);
  CusparseFrontend::AddVariableForArguments<cusparseOperation_t>(trans);
  CusparseFrontend::AddVariableForArguments<int>(mb);
  CusparseFrontend::AddVariableForArguments<int>(nnzb);
  CusparseFrontend::AddVariableForArguments<size_t>((size_t)descr);
  CusparseFrontend::AddDevicePointerForArguments(bsrVal);
  CusparseFrontend::AddDevicePointerForArguments(bsrRowPtr);
  CusparseFrontend::AddDevicePointerForArguments(bsrColInd);
  CusparseFrontend::AddVariableForArguments<int>(blockDim);
  CusparseFrontend::AddVariableForArguments<size_t>((size_t) info);
  CusparseFrontend::AddVariableForArguments<cusparseSolvePolicy_t>(policy);
  CusparseFrontend::AddDevicePointerForArguments(pBuffer);
  CusparseFrontend::Execute("cusparseSbsrsv2_analysis");
  if (CusparseFrontend::Success()) {
  info = CusparseFrontend::GetOutputVariable<bsrsv2Info_t>();
  }
  return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseDbsrsv2_analysis(cusparseHandle_t handle, cusparseDirection_t dir, cusparseOperation_t trans, int mb, int nnzb, cusparseMatDescr_t descr, const double* bsrVal, const int* bsrRowPtr, const int* bsrColInd, int blockDim, bsrsv2Info_t info, cusparseSolvePolicy_t policy, void* pBuffer) {
  CusparseFrontend::Prepare();
  CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
  CusparseFrontend::AddVariableForArguments<cusparseDirection_t>(dir);
  CusparseFrontend::AddVariableForArguments<cusparseOperation_t>(trans);
  CusparseFrontend::AddVariableForArguments<int>(mb);
  CusparseFrontend::AddVariableForArguments<int>(nnzb);
  CusparseFrontend::AddVariableForArguments<size_t>((size_t)descr);
  CusparseFrontend::AddDevicePointerForArguments(bsrVal);
  CusparseFrontend::AddDevicePointerForArguments(bsrRowPtr);
  CusparseFrontend::AddDevicePointerForArguments(bsrColInd);
  CusparseFrontend::AddVariableForArguments<int>(blockDim);
  CusparseFrontend::AddVariableForArguments<size_t>((size_t) info);
  CusparseFrontend::AddVariableForArguments<cusparseSolvePolicy_t>(policy);
  CusparseFrontend::AddDevicePointerForArguments(pBuffer);
  CusparseFrontend::Execute("cusparseDbsrsv2_analysis");
  if (CusparseFrontend::Success()) {
  info = CusparseFrontend::GetOutputVariable<bsrsv2Info_t>();
  }
  return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseCbsrsv2_analysis(cusparseHandle_t handle, cusparseDirection_t dir, cusparseOperation_t trans, int mb, int nnzb, cusparseMatDescr_t descr, const cuComplex* bsrVal, const int* bsrRowPtr, const int* bsrColInd, int blockDim, bsrsv2Info_t info, cusparseSolvePolicy_t policy, void* pBuffer) {
  CusparseFrontend::Prepare();
  CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
  CusparseFrontend::AddVariableForArguments<cusparseDirection_t>(dir);
  CusparseFrontend::AddVariableForArguments<cusparseOperation_t>(trans);
  CusparseFrontend::AddVariableForArguments<int>(mb);
  CusparseFrontend::AddVariableForArguments<int>(nnzb);
  CusparseFrontend::AddVariableForArguments<size_t>((size_t)descr);
  CusparseFrontend::AddDevicePointerForArguments(bsrVal);
  CusparseFrontend::AddDevicePointerForArguments(bsrRowPtr);
  CusparseFrontend::AddDevicePointerForArguments(bsrColInd);
  CusparseFrontend::AddVariableForArguments<int>(blockDim);
  CusparseFrontend::AddVariableForArguments<size_t>((size_t) info);
  CusparseFrontend::AddVariableForArguments<cusparseSolvePolicy_t>(policy);
  CusparseFrontend::AddDevicePointerForArguments(pBuffer);
  CusparseFrontend::Execute("cusparseCbsrsv2_analysis");
  if (CusparseFrontend::Success()) {
  info = CusparseFrontend::GetOutputVariable<bsrsv2Info_t>();
  }
  return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseZbsrsv2_analysis(cusparseHandle_t handle, cusparseDirection_t dir, cusparseOperation_t trans, int mb, int nnzb, cusparseMatDescr_t descr, const cuDoubleComplex* bsrVal, const int* bsrRowPtr, const int* bsrColInd, int blockDim, bsrsv2Info_t info, cusparseSolvePolicy_t policy, void* pBuffer) {
  CusparseFrontend::Prepare();
  CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
  CusparseFrontend::AddVariableForArguments<cusparseDirection_t>(dir);
  CusparseFrontend::AddVariableForArguments<cusparseOperation_t>(trans);
  CusparseFrontend::AddVariableForArguments<int>(mb);
  CusparseFrontend::AddVariableForArguments<int>(nnzb);
  CusparseFrontend::AddVariableForArguments<size_t>((size_t)descr);
  CusparseFrontend::AddDevicePointerForArguments(bsrVal);
  CusparseFrontend::AddDevicePointerForArguments(bsrRowPtr);
  CusparseFrontend::AddDevicePointerForArguments(bsrColInd);
  CusparseFrontend::AddVariableForArguments<int>(blockDim);
  CusparseFrontend::AddVariableForArguments<size_t>((size_t) info);
  CusparseFrontend::AddVariableForArguments<cusparseSolvePolicy_t>(policy);
  CusparseFrontend::AddDevicePointerForArguments(pBuffer);
  CusparseFrontend::Execute("cusparseZbsrsv2_analysis");
  if (CusparseFrontend::Success()) {
  info = CusparseFrontend::GetOutputVariable<bsrsv2Info_t>();
  }
  return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseSbsrsv2_solve(cusparseHandle_t handle, cusparseDirection_t dir, cusparseOperation_t trans, int mb, int nnzb, const float* alpha, const cusparseMatDescr_t descr, const float* bsrVal, const int* bsrRowPtr, const int* bsrColInd, int blockDim, bsrsv2Info_t info, const float* x, float* y, cusparseSolvePolicy_t policy, void* pBuffer) {
  CusparseFrontend::Prepare();
  CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
  CusparseFrontend::AddVariableForArguments<cusparseDirection_t>(dir);
  CusparseFrontend::AddVariableForArguments<cusparseOperation_t>(trans);
  CusparseFrontend::AddVariableForArguments<int>(mb);
  CusparseFrontend::AddVariableForArguments<int>(nnzb);
  CusparseFrontend::AddHostPointerForArguments(const_cast<float *>(alpha));
  CusparseFrontend::AddVariableForArguments<size_t>((size_t)descr);
  CusparseFrontend::AddDevicePointerForArguments(bsrVal);
  CusparseFrontend::AddDevicePointerForArguments(bsrRowPtr);
  CusparseFrontend::AddDevicePointerForArguments(bsrColInd);
  CusparseFrontend::AddVariableForArguments<int>(blockDim);
  CusparseFrontend::AddVariableForArguments<size_t>((size_t) info);
  CusparseFrontend::AddDevicePointerForArguments(x);
  CusparseFrontend::AddDevicePointerForArguments(y);
  CusparseFrontend::AddVariableForArguments<cusparseSolvePolicy_t>(policy);
  CusparseFrontend::AddDevicePointerForArguments(pBuffer);
  CusparseFrontend::Execute("cusparseSbsrsv2_solve");
  if (CusparseFrontend::Success()) {
  y = (float *)CusparseFrontend::GetOutputDevicePointer();
  }
  return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseDbsrsv2_solve(cusparseHandle_t handle, cusparseDirection_t dir, cusparseOperation_t trans, int mb, int nnzb, const double* alpha, const cusparseMatDescr_t descr, const double* bsrVal, const int* bsrRowPtr, const int* bsrColInd, int blockDim, bsrsv2Info_t info, const double* x, double* y, cusparseSolvePolicy_t policy, void* pBuffer) {
  CusparseFrontend::Prepare();
  CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
  CusparseFrontend::AddVariableForArguments<cusparseDirection_t>(dir);
  CusparseFrontend::AddVariableForArguments<cusparseOperation_t>(trans);
  CusparseFrontend::AddVariableForArguments<int>(mb);
  CusparseFrontend::AddVariableForArguments<int>(nnzb);
  CusparseFrontend::AddHostPointerForArguments(const_cast<double *>(alpha));
  CusparseFrontend::AddVariableForArguments<size_t>((size_t)descr);
  CusparseFrontend::AddDevicePointerForArguments(bsrVal);
  CusparseFrontend::AddDevicePointerForArguments(bsrRowPtr);
  CusparseFrontend::AddDevicePointerForArguments(bsrColInd);
  CusparseFrontend::AddVariableForArguments<int>(blockDim);
  CusparseFrontend::AddVariableForArguments<size_t>((size_t) info);
  CusparseFrontend::AddDevicePointerForArguments(x);
  CusparseFrontend::AddDevicePointerForArguments(y);
  CusparseFrontend::AddVariableForArguments<cusparseSolvePolicy_t>(policy);
  CusparseFrontend::AddDevicePointerForArguments(pBuffer);
  CusparseFrontend::Execute("cusparseDbsrsv2_solve");
  if (CusparseFrontend::Success()) {
  y = (double *)CusparseFrontend::GetOutputDevicePointer();
  }
  return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseCbsrsv2_solve(cusparseHandle_t handle, cusparseDirection_t dir, cusparseOperation_t trans, int mb, int nnzb, const cuComplex* alpha, const cusparseMatDescr_t descr, const cuComplex* bsrVal, const int* bsrRowPtr, const int* bsrColInd, int blockDim, bsrsv2Info_t info, const cuComplex* x, cuComplex* y, cusparseSolvePolicy_t policy, void* pBuffer) {
  CusparseFrontend::Prepare();
  CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
  CusparseFrontend::AddVariableForArguments<cusparseDirection_t>(dir);
  CusparseFrontend::AddVariableForArguments<cusparseOperation_t>(trans);
  CusparseFrontend::AddVariableForArguments<int>(mb);
  CusparseFrontend::AddVariableForArguments<int>(nnzb);
  CusparseFrontend::AddHostPointerForArguments(const_cast<cuComplex *>(alpha));
  CusparseFrontend::AddVariableForArguments<size_t>((size_t)descr);
  CusparseFrontend::AddDevicePointerForArguments(bsrVal);
  CusparseFrontend::AddDevicePointerForArguments(bsrRowPtr);
  CusparseFrontend::AddDevicePointerForArguments(bsrColInd);
  CusparseFrontend::AddVariableForArguments<int>(blockDim);
  CusparseFrontend::AddVariableForArguments<size_t>((size_t) info);
  CusparseFrontend::AddDevicePointerForArguments(x);
  CusparseFrontend::AddDevicePointerForArguments(y);
  CusparseFrontend::AddVariableForArguments<cusparseSolvePolicy_t>(policy);
  CusparseFrontend::AddDevicePointerForArguments(pBuffer);
  CusparseFrontend::Execute("cusparseCbsrsv2_solve");
  if (CusparseFrontend::Success()) {
  y = (cuComplex *)CusparseFrontend::GetOutputDevicePointer();
  }
  return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseZbsrsv2_solve(cusparseHandle_t handle, cusparseDirection_t dir, cusparseOperation_t trans, int mb, int nnzb, const cuDoubleComplex* alpha, const cusparseMatDescr_t descr, const cuDoubleComplex* bsrVal, const int* bsrRowPtr, const int* bsrColInd, int blockDim, bsrsv2Info_t info, const cuDoubleComplex* x, cuDoubleComplex* y, cusparseSolvePolicy_t policy, void* pBuffer) {
  CusparseFrontend::Prepare();
  CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
  CusparseFrontend::AddVariableForArguments<cusparseDirection_t>(dir);
  CusparseFrontend::AddVariableForArguments<cusparseOperation_t>(trans);
  CusparseFrontend::AddVariableForArguments<int>(mb);
  CusparseFrontend::AddVariableForArguments<int>(nnzb);
  CusparseFrontend::AddHostPointerForArguments(const_cast<cuDoubleComplex *>(alpha));
  CusparseFrontend::AddVariableForArguments<size_t>((size_t)descr);
  CusparseFrontend::AddDevicePointerForArguments(bsrVal);
  CusparseFrontend::AddDevicePointerForArguments(bsrRowPtr);
  CusparseFrontend::AddDevicePointerForArguments(bsrColInd);
  CusparseFrontend::AddVariableForArguments<int>(blockDim);
  CusparseFrontend::AddVariableForArguments<size_t>((size_t) info);
  CusparseFrontend::AddDevicePointerForArguments(x);
  CusparseFrontend::AddDevicePointerForArguments(y);
  CusparseFrontend::AddVariableForArguments<cusparseSolvePolicy_t>(policy);
  CusparseFrontend::AddDevicePointerForArguments(pBuffer);
  CusparseFrontend::Execute("cusparseZbsrsv2_solve");
  if (CusparseFrontend::Success()) {
  y = (cuDoubleComplex *)CusparseFrontend::GetOutputDevicePointer();
  }
  return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseXbsrsv2_zeroPivot(cusparseHandle_t handle, bsrsv2Info_t info, int* position) {
  CusparseFrontend::Prepare();
  CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
  CusparseFrontend::AddVariableForArguments<size_t>((size_t) info);
  CusparseFrontend::AddDevicePointerForArguments(position);
  CusparseFrontend::Execute("cusparseXbsrsv2_zeroPivot");
  if (CusparseFrontend::Success()) {
  position = (int *)CusparseFrontend::GetOutputDevicePointer();
  }
  return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseCsrmvEx_bufferSize(cusparseHandle_t handle, cusparseAlgMode_t alg, cusparseOperation_t transA, int m, int n, int nnz, const void* alpha, cudaDataType alphatype, const cusparseMatDescr_t descrA, const void* csrValA, cudaDataType csrValAtype, const int* csrRowPtrA, const int* csrColIndA, const void* x, cudaDataType xtype, const void* beta, cudaDataType betatype, void* y, cudaDataType ytype, cudaDataType executiontype, size_t* buffer) {
  CusparseFrontend::Prepare();
  CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
  CusparseFrontend::AddVariableForArguments<cusparseAlgMode_t>(alg);
  CusparseFrontend::AddVariableForArguments<cusparseOperation_t>(transA);
  CusparseFrontend::AddVariableForArguments<int>(m);
  CusparseFrontend::AddVariableForArguments<int>(n);
  CusparseFrontend::AddVariableForArguments<int>(nnz);
  CusparseFrontend::AddVariableForArguments<cudaDataType>(alphatype);
  switch(alphatype){
  case CUDA_R_32F:
  //float
  CusparseFrontend::AddVariableForArguments(*(float *)alpha);
  break;
  case CUDA_R_64F:
  //double
  CusparseFrontend::AddVariableForArguments(*(double *)alpha);
  break;
  case CUDA_C_32F:
  //cuComplex
  CusparseFrontend::AddVariableForArguments(*(cuComplex *)alpha);
  break;
  case CUDA_C_64F:
  //cuDoubleComplex
  CusparseFrontend::AddVariableForArguments(*(cuDoubleComplex *)alpha);
  break;
  default:
  throw "Type not supported by GVirtus!";
  }
  CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrA);
  CusparseFrontend::AddDevicePointerForArguments(csrValA);
  CusparseFrontend::AddVariableForArguments<cudaDataType>(csrValAtype);
  CusparseFrontend::AddDevicePointerForArguments(csrRowPtrA);
  CusparseFrontend::AddDevicePointerForArguments(csrColIndA);
  CusparseFrontend::AddDevicePointerForArguments(x);
  CusparseFrontend::AddVariableForArguments<cudaDataType>(xtype);
  CusparseFrontend::AddVariableForArguments<cudaDataType>(betatype);
  switch(betatype){
  case CUDA_R_32F:
  //float
  CusparseFrontend::AddVariableForArguments(*(float *)beta);
  break;
  case CUDA_R_64F:
  //double
  CusparseFrontend::AddVariableForArguments(*(double *)beta);
  break;
  case CUDA_C_32F:
  //cuComplex
  CusparseFrontend::AddVariableForArguments(*(cuComplex *)beta);
  break;
  case CUDA_C_64F:
  //cuDoubleComplex
  CusparseFrontend::AddVariableForArguments(*(cuDoubleComplex *)beta);
  break;
  default:
  throw "Type not supported by GVirtus!";
  }
  CusparseFrontend::AddDevicePointerForArguments(y);
  CusparseFrontend::AddVariableForArguments<cudaDataType>(ytype);
  CusparseFrontend::AddVariableForArguments<cudaDataType>(executiontype);
  CusparseFrontend::Execute("cusparseCsrmvEx_bufferSize");
  if (CusparseFrontend::Success()) {
  *buffer = *(CusparseFrontend::GetOutputHostPointer<size_t>());
  }
  return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseCsrmvEx(cusparseHandle_t handle, cusparseAlgMode_t alg, cusparseOperation_t transA, int m, int n, int nnz, const void* alpha, cudaDataType alphatype, const cusparseMatDescr_t descrA, const void* csrValA, cudaDataType csrValAtype, const int* csrRowPtrA, const int* csrColIndA, const void* x, cudaDataType xtype, const void* beta, cudaDataType betatype, void* y, cudaDataType ytype, cudaDataType executiontype, void* buffer) {
  CusparseFrontend::Prepare();
  CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
  CusparseFrontend::AddVariableForArguments<cusparseAlgMode_t>(alg);
  CusparseFrontend::AddVariableForArguments<cusparseOperation_t>(transA);
  CusparseFrontend::AddVariableForArguments<int>(m);
  CusparseFrontend::AddVariableForArguments<int>(n);
  CusparseFrontend::AddVariableForArguments<int>(nnz);
  CusparseFrontend::AddVariableForArguments<cudaDataType>(alphatype);
  switch(alphatype){
  case CUDA_R_32F:
  //float
  CusparseFrontend::AddVariableForArguments(*(float *)alpha);
  break;
  case CUDA_R_64F:
  //double
  CusparseFrontend::AddVariableForArguments(*(double *)alpha);
  break;
  case CUDA_C_32F:
  //cuComplex
  CusparseFrontend::AddVariableForArguments(*(cuComplex *)alpha);
  break;
  case CUDA_C_64F:
  //cuDoubleComplex
  CusparseFrontend::AddVariableForArguments(*(cuDoubleComplex *)alpha);
  break;
  default:
  throw "Type not supported by GVirtus!";
  }
  CusparseFrontend::AddVariableForArguments<size_t>((size_t)descrA);
  CusparseFrontend::AddDevicePointerForArguments(csrValA);
  CusparseFrontend::AddVariableForArguments<cudaDataType>(csrValAtype);
  CusparseFrontend::AddDevicePointerForArguments(csrRowPtrA);
  CusparseFrontend::AddDevicePointerForArguments(csrColIndA);
  CusparseFrontend::AddDevicePointerForArguments(x);
  CusparseFrontend::AddVariableForArguments<cudaDataType>(xtype);
  CusparseFrontend::AddVariableForArguments<cudaDataType>(betatype);
  switch(betatype){
  case CUDA_R_32F:
  //float
  CusparseFrontend::AddVariableForArguments(*(float *)beta);
  break;
  case CUDA_R_64F:
  //double
  CusparseFrontend::AddVariableForArguments(*(double *)beta);
  break;
  case CUDA_C_32F:
  //cuComplex
  CusparseFrontend::AddVariableForArguments(*(cuComplex *)beta);
  break;
  case CUDA_C_64F:
  //cuDoubleComplex
  CusparseFrontend::AddVariableForArguments(*(cuDoubleComplex *)beta);
  break;
  default:
  throw "Type not supported by GVirtus!";
  }
  CusparseFrontend::AddDevicePointerForArguments(y);
  CusparseFrontend::AddVariableForArguments<cudaDataType>(ytype);
  CusparseFrontend::AddVariableForArguments<cudaDataType>(executiontype);
  CusparseFrontend::AddDevicePointerForArguments(buffer);
  CusparseFrontend::Execute("cusparseCsrmvEx");
  if (CusparseFrontend::Success()) {
  y = (void *)CusparseFrontend::GetOutputDevicePointer();
  }
  return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseScsrsv2_bufferSize(cusparseHandle_t handle, cusparseOperation_t trans, int m, int nnz, const cusparseMatDescr_t descr, float* csrVal, const int* csrRowPtr, const int* csrColInd, csrsv2Info_t info, int* pBufferSizeInBytes) {
  CusparseFrontend::Prepare();
  CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
  CusparseFrontend::AddVariableForArguments<cusparseOperation_t>(trans);
  CusparseFrontend::AddVariableForArguments<int>(m);
  CusparseFrontend::AddVariableForArguments<int>(nnz);
  CusparseFrontend::AddVariableForArguments<size_t>((size_t)descr);
  CusparseFrontend::AddDevicePointerForArguments(csrVal);
  CusparseFrontend::AddDevicePointerForArguments(csrRowPtr);
  CusparseFrontend::AddDevicePointerForArguments(csrColInd);
  CusparseFrontend::AddVariableForArguments<size_t>((size_t) info);
  CusparseFrontend::AddDevicePointerForArguments(pBufferSizeInBytes);
  CusparseFrontend::Execute("cusparseScsrsv2_bufferSize");
  if (CusparseFrontend::Success()) {
  info = CusparseFrontend::GetOutputVariable<csrsv2Info_t>();
  pBufferSizeInBytes = (int *)CusparseFrontend::GetOutputDevicePointer();
  }
  return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseDcsrsv2_bufferSize(cusparseHandle_t handle, cusparseOperation_t trans, int m, int nnz, const cusparseMatDescr_t descr, double* csrVal, const int* csrRowPtr, const int* csrColInd, csrsv2Info_t info, int* pBufferSizeInBytes) {
  CusparseFrontend::Prepare();
  CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
  CusparseFrontend::AddVariableForArguments<cusparseOperation_t>(trans);
  CusparseFrontend::AddVariableForArguments<int>(m);
  CusparseFrontend::AddVariableForArguments<int>(nnz);
  CusparseFrontend::AddVariableForArguments<size_t>((size_t)descr);
  CusparseFrontend::AddDevicePointerForArguments(csrVal);
  CusparseFrontend::AddDevicePointerForArguments(csrRowPtr);
  CusparseFrontend::AddDevicePointerForArguments(csrColInd);
  CusparseFrontend::AddVariableForArguments<size_t>((size_t) info);
  CusparseFrontend::AddDevicePointerForArguments(pBufferSizeInBytes);
  CusparseFrontend::Execute("cusparseDcsrsv2_bufferSize");
  if (CusparseFrontend::Success()) {
  info = CusparseFrontend::GetOutputVariable<csrsv2Info_t>();
  pBufferSizeInBytes = (int *)CusparseFrontend::GetOutputDevicePointer();
  }
  return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseCcsrsv2_bufferSize(cusparseHandle_t handle, cusparseOperation_t trans, int m, int nnz, const cusparseMatDescr_t descr, cuComplex* csrVal, const int* csrRowPtr, const int* csrColInd, csrsv2Info_t info, int* pBufferSizeInBytes) {
  CusparseFrontend::Prepare();
  CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
  CusparseFrontend::AddVariableForArguments<cusparseOperation_t>(trans);
  CusparseFrontend::AddVariableForArguments<int>(m);
  CusparseFrontend::AddVariableForArguments<int>(nnz);
  CusparseFrontend::AddVariableForArguments<size_t>((size_t)descr);
  CusparseFrontend::AddDevicePointerForArguments(csrVal);
  CusparseFrontend::AddDevicePointerForArguments(csrRowPtr);
  CusparseFrontend::AddDevicePointerForArguments(csrColInd);
  CusparseFrontend::AddVariableForArguments<size_t>((size_t) info);
  CusparseFrontend::AddDevicePointerForArguments(pBufferSizeInBytes);
  CusparseFrontend::Execute("cusparseCcsrsv2_bufferSize");
  if (CusparseFrontend::Success()) {
  info = CusparseFrontend::GetOutputVariable<csrsv2Info_t>();
  pBufferSizeInBytes = (int *)CusparseFrontend::GetOutputDevicePointer();
  }
  return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseZcsrsv2_bufferSize(cusparseHandle_t handle, cusparseOperation_t trans, int m, int nnz, const cusparseMatDescr_t descr, cuDoubleComplex* csrVal, const int* csrRowPtr, const int* csrColInd, csrsv2Info_t info, int* pBufferSizeInBytes) {
  CusparseFrontend::Prepare();
  CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
  CusparseFrontend::AddVariableForArguments<cusparseOperation_t>(trans);
  CusparseFrontend::AddVariableForArguments<int>(m);
  CusparseFrontend::AddVariableForArguments<int>(nnz);
  CusparseFrontend::AddVariableForArguments<size_t>((size_t)descr);
  CusparseFrontend::AddDevicePointerForArguments(csrVal);
  CusparseFrontend::AddDevicePointerForArguments(csrRowPtr);
  CusparseFrontend::AddDevicePointerForArguments(csrColInd);
  CusparseFrontend::AddVariableForArguments<size_t>((size_t) info);
  CusparseFrontend::AddDevicePointerForArguments(pBufferSizeInBytes);
  CusparseFrontend::Execute("cusparseZcsrsv2_bufferSize");
  if (CusparseFrontend::Success()) {
  info = CusparseFrontend::GetOutputVariable<csrsv2Info_t>();
  pBufferSizeInBytes = (int *)CusparseFrontend::GetOutputDevicePointer();
  }
  return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseScsrsv2_analysis(cusparseHandle_t handle, cusparseOperation_t trans, int m, int nnz, cusparseMatDescr_t descr, const float* csrVal, const int* csrRowPtr, const int* csrColInd, csrsv2Info_t info, cusparseSolvePolicy_t policy, void* pBuffer) {
  CusparseFrontend::Prepare();
  CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
  CusparseFrontend::AddVariableForArguments<cusparseOperation_t>(trans);
  CusparseFrontend::AddVariableForArguments<int>(m);
  CusparseFrontend::AddVariableForArguments<int>(nnz);
  CusparseFrontend::AddVariableForArguments<size_t>((size_t)descr);
  CusparseFrontend::AddDevicePointerForArguments(csrVal);
  CusparseFrontend::AddDevicePointerForArguments(csrRowPtr);
  CusparseFrontend::AddDevicePointerForArguments(csrColInd);
  CusparseFrontend::AddVariableForArguments<size_t>((size_t) info);
  CusparseFrontend::AddVariableForArguments<cusparseSolvePolicy_t>(policy);
  CusparseFrontend::AddDevicePointerForArguments(pBuffer);
  CusparseFrontend::Execute("cusparseScsrsv2_analysis");
  if (CusparseFrontend::Success()) {
  info = CusparseFrontend::GetOutputVariable<csrsv2Info_t>();
  }
  return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseDcsrsv2_analysis(cusparseHandle_t handle, cusparseOperation_t trans, int m, int nnz, cusparseMatDescr_t descr, const double* csrVal, const int* csrRowPtr, const int* csrColInd, csrsv2Info_t info, cusparseSolvePolicy_t policy, void* pBuffer) {
  CusparseFrontend::Prepare();
  CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
  CusparseFrontend::AddVariableForArguments<cusparseOperation_t>(trans);
  CusparseFrontend::AddVariableForArguments<int>(m);
  CusparseFrontend::AddVariableForArguments<int>(nnz);
  CusparseFrontend::AddVariableForArguments<size_t>((size_t)descr);
  CusparseFrontend::AddDevicePointerForArguments(csrVal);
  CusparseFrontend::AddDevicePointerForArguments(csrRowPtr);
  CusparseFrontend::AddDevicePointerForArguments(csrColInd);
  CusparseFrontend::AddVariableForArguments<size_t>((size_t) info);
  CusparseFrontend::AddVariableForArguments<cusparseSolvePolicy_t>(policy);
  CusparseFrontend::AddDevicePointerForArguments(pBuffer);
  CusparseFrontend::Execute("cusparseDcsrsv2_analysis");
  if (CusparseFrontend::Success()) {
  info = CusparseFrontend::GetOutputVariable<csrsv2Info_t>();
  }
  return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseCcsrsv2_analysis(cusparseHandle_t handle, cusparseOperation_t trans, int m, int nnz, cusparseMatDescr_t descr, const cuComplex* csrVal, const int* csrRowPtr, const int* csrColInd, csrsv2Info_t info, cusparseSolvePolicy_t policy, void* pBuffer) {
  CusparseFrontend::Prepare();
  CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
  CusparseFrontend::AddVariableForArguments<cusparseOperation_t>(trans);
  CusparseFrontend::AddVariableForArguments<int>(m);
  CusparseFrontend::AddVariableForArguments<int>(nnz);
  CusparseFrontend::AddVariableForArguments<size_t>((size_t)descr);
  CusparseFrontend::AddDevicePointerForArguments(csrVal);
  CusparseFrontend::AddDevicePointerForArguments(csrRowPtr);
  CusparseFrontend::AddDevicePointerForArguments(csrColInd);
  CusparseFrontend::AddVariableForArguments<size_t>((size_t) info);
  CusparseFrontend::AddVariableForArguments<cusparseSolvePolicy_t>(policy);
  CusparseFrontend::AddDevicePointerForArguments(pBuffer);
  CusparseFrontend::Execute("cusparseCcsrsv2_analysis");
  if (CusparseFrontend::Success()) {
  info = CusparseFrontend::GetOutputVariable<csrsv2Info_t>();
  }
  return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseZcsrsv2_analysis(cusparseHandle_t handle, cusparseOperation_t trans, int m, int nnz, cusparseMatDescr_t descr, const cuDoubleComplex* csrVal, const int* csrRowPtr, const int* csrColInd, csrsv2Info_t info, cusparseSolvePolicy_t policy, void* pBuffer) {
  CusparseFrontend::Prepare();
  CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
  CusparseFrontend::AddVariableForArguments<cusparseOperation_t>(trans);
  CusparseFrontend::AddVariableForArguments<int>(m);
  CusparseFrontend::AddVariableForArguments<int>(nnz);
  CusparseFrontend::AddVariableForArguments<size_t>((size_t)descr);
  CusparseFrontend::AddDevicePointerForArguments(csrVal);
  CusparseFrontend::AddDevicePointerForArguments(csrRowPtr);
  CusparseFrontend::AddDevicePointerForArguments(csrColInd);
  CusparseFrontend::AddVariableForArguments<size_t>((size_t) info);
  CusparseFrontend::AddVariableForArguments<cusparseSolvePolicy_t>(policy);
  CusparseFrontend::AddDevicePointerForArguments(pBuffer);
  CusparseFrontend::Execute("cusparseZcsrsv2_analysis");
  if (CusparseFrontend::Success()) {
  info = CusparseFrontend::GetOutputVariable<csrsv2Info_t>();
  }
  return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseScsrsv2_solve(cusparseHandle_t handle, cusparseOperation_t trans, int m, int nnz, const float* alpha, const cusparseMatDescr_t descr, const float* csrVal, const int* csrRowPtr, const int* csrColInd, csrsv2Info_t info, const float* x, float* y, cusparseSolvePolicy_t policy, void* pBuffer) {
  CusparseFrontend::Prepare();
  CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
  CusparseFrontend::AddVariableForArguments<cusparseOperation_t>(trans);
  CusparseFrontend::AddVariableForArguments<int>(m);
  CusparseFrontend::AddVariableForArguments<int>(nnz);
  CusparseFrontend::AddHostPointerForArguments(const_cast<float *>(alpha));
  CusparseFrontend::AddVariableForArguments<size_t>((size_t)descr);
  CusparseFrontend::AddDevicePointerForArguments(csrVal);
  CusparseFrontend::AddDevicePointerForArguments(csrRowPtr);
  CusparseFrontend::AddDevicePointerForArguments(csrColInd);
  CusparseFrontend::AddVariableForArguments<size_t>((size_t) info);
  CusparseFrontend::AddDevicePointerForArguments(x);
  CusparseFrontend::AddDevicePointerForArguments(y);
  CusparseFrontend::AddVariableForArguments<cusparseSolvePolicy_t>(policy);
  CusparseFrontend::AddDevicePointerForArguments(pBuffer);
  CusparseFrontend::Execute("cusparseScsrsv2_solve");
  if (CusparseFrontend::Success()) {
  y = (float *)CusparseFrontend::GetOutputDevicePointer();
  }
  return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseDcsrsv2_solve(cusparseHandle_t handle, cusparseOperation_t trans, int m, int nnz, const double* alpha, const cusparseMatDescr_t descr, const double* csrVal, const int* csrRowPtr, const int* csrColInd, csrsv2Info_t info, const double* x, double* y, cusparseSolvePolicy_t policy, void* pBuffer) {
  CusparseFrontend::Prepare();
  CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
  CusparseFrontend::AddVariableForArguments<cusparseOperation_t>(trans);
  CusparseFrontend::AddVariableForArguments<int>(m);
  CusparseFrontend::AddVariableForArguments<int>(nnz);
  CusparseFrontend::AddHostPointerForArguments(const_cast<double *>(alpha));
  CusparseFrontend::AddVariableForArguments<size_t>((size_t)descr);
  CusparseFrontend::AddDevicePointerForArguments(csrVal);
  CusparseFrontend::AddDevicePointerForArguments(csrRowPtr);
  CusparseFrontend::AddDevicePointerForArguments(csrColInd);
  CusparseFrontend::AddVariableForArguments<size_t>((size_t) info);
  CusparseFrontend::AddDevicePointerForArguments(x);
  CusparseFrontend::AddDevicePointerForArguments(y);
  CusparseFrontend::AddVariableForArguments<cusparseSolvePolicy_t>(policy);
  CusparseFrontend::AddDevicePointerForArguments(pBuffer);
  CusparseFrontend::Execute("cusparseDcsrsv2_solve");
  if (CusparseFrontend::Success()) {
  y = (double *)CusparseFrontend::GetOutputDevicePointer();
  }
  return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseCcsrsv2_solve(cusparseHandle_t handle, cusparseOperation_t trans, int m, int nnz, const cuComplex* alpha, const cusparseMatDescr_t descr, const cuComplex* csrVal, const int* csrRowPtr, const int* csrColInd, csrsv2Info_t info, const cuComplex* x, cuComplex* y, cusparseSolvePolicy_t policy, void* pBuffer) {
  CusparseFrontend::Prepare();
  CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
  CusparseFrontend::AddVariableForArguments<cusparseOperation_t>(trans);
  CusparseFrontend::AddVariableForArguments<int>(m);
  CusparseFrontend::AddVariableForArguments<int>(nnz);
  CusparseFrontend::AddHostPointerForArguments(const_cast<cuComplex *>(alpha));
  CusparseFrontend::AddVariableForArguments<size_t>((size_t)descr);
  CusparseFrontend::AddDevicePointerForArguments(csrVal);
  CusparseFrontend::AddDevicePointerForArguments(csrRowPtr);
  CusparseFrontend::AddDevicePointerForArguments(csrColInd);
  CusparseFrontend::AddVariableForArguments<size_t>((size_t) info);
  CusparseFrontend::AddDevicePointerForArguments(x);
  CusparseFrontend::AddDevicePointerForArguments(y);
  CusparseFrontend::AddVariableForArguments<cusparseSolvePolicy_t>(policy);
  CusparseFrontend::AddDevicePointerForArguments(pBuffer);
  CusparseFrontend::Execute("cusparseCcsrsv2_solve");
  if (CusparseFrontend::Success()) {
  y = (cuComplex *)CusparseFrontend::GetOutputDevicePointer();
  }
  return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseZcsrsv2_solve(cusparseHandle_t handle, cusparseOperation_t trans, int m, int nnz, const cuDoubleComplex* alpha, const cusparseMatDescr_t descr, const cuDoubleComplex* csrVal, const int* csrRowPtr, const int* csrColInd, csrsv2Info_t info, const cuDoubleComplex* x, cuDoubleComplex* y, cusparseSolvePolicy_t policy, void* pBuffer) {
  CusparseFrontend::Prepare();
  CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
  CusparseFrontend::AddVariableForArguments<cusparseOperation_t>(trans);
  CusparseFrontend::AddVariableForArguments<int>(m);
  CusparseFrontend::AddVariableForArguments<int>(nnz);
  CusparseFrontend::AddHostPointerForArguments(const_cast<cuDoubleComplex *>(alpha));
  CusparseFrontend::AddVariableForArguments<size_t>((size_t)descr);
  CusparseFrontend::AddDevicePointerForArguments(csrVal);
  CusparseFrontend::AddDevicePointerForArguments(csrRowPtr);
  CusparseFrontend::AddDevicePointerForArguments(csrColInd);
  CusparseFrontend::AddVariableForArguments<size_t>((size_t) info);
  CusparseFrontend::AddDevicePointerForArguments(x);
  CusparseFrontend::AddDevicePointerForArguments(y);
  CusparseFrontend::AddVariableForArguments<cusparseSolvePolicy_t>(policy);
  CusparseFrontend::AddDevicePointerForArguments(pBuffer);
  CusparseFrontend::Execute("cusparseZcsrsv2_solve");
  if (CusparseFrontend::Success()) {
  y = (cuDoubleComplex *)CusparseFrontend::GetOutputDevicePointer();
  }
  return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseXcsrsv2_zeroPivot(cusparseHandle_t handle, csrsv2Info_t info, int* position) {
  CusparseFrontend::Prepare();
  CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
  CusparseFrontend::AddVariableForArguments<size_t>((size_t) info);
  CusparseFrontend::AddDevicePointerForArguments(position);
  CusparseFrontend::Execute("cusparseXcsrsv2_zeroPivot");
  if (CusparseFrontend::Success()) {
  position = (int *)CusparseFrontend::GetOutputDevicePointer();
  }
  return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseSgemvi_bufferSize(cusparseHandle_t handle, cusparseOperation_t trans, int m, int n, int nnz, int* pBufferSize) {
  CusparseFrontend::Prepare();
  CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
  CusparseFrontend::AddVariableForArguments<cusparseOperation_t>(trans);
  CusparseFrontend::AddVariableForArguments<int>(m);
  CusparseFrontend::AddVariableForArguments<int>(n);
  CusparseFrontend::AddVariableForArguments<int>(nnz);
  CusparseFrontend::AddDevicePointerForArguments(pBufferSize);
  CusparseFrontend::Execute("cusparseSgemvi_bufferSize");
  if (CusparseFrontend::Success()) {
  pBufferSize = (int *)CusparseFrontend::GetOutputDevicePointer();
  }
  return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseDgemvi_bufferSize(cusparseHandle_t handle, cusparseOperation_t trans, int m, int n, int nnz, int* pBufferSize) {
  CusparseFrontend::Prepare();
  CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
  CusparseFrontend::AddVariableForArguments<cusparseOperation_t>(trans);
  CusparseFrontend::AddVariableForArguments<int>(m);
  CusparseFrontend::AddVariableForArguments<int>(n);
  CusparseFrontend::AddVariableForArguments<int>(nnz);
  CusparseFrontend::Execute("cusparseDgemvi_bufferSize");
  if (CusparseFrontend::Success()) {
    pBufferSize = (int *)CusparseFrontend::GetOutputDevicePointer();
  }
  return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseCgemvi_bufferSize(cusparseHandle_t handle, cusparseOperation_t trans, int m, int n, int nnz, int* pBufferSize) {
  CusparseFrontend::Prepare();
  CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
  CusparseFrontend::AddVariableForArguments<cusparseOperation_t>(trans);
  CusparseFrontend::AddVariableForArguments<int>(m);
  CusparseFrontend::AddVariableForArguments<int>(n);
  CusparseFrontend::AddVariableForArguments<int>(nnz);
  CusparseFrontend::Execute("cusparseCgemvi_bufferSize");
  if (CusparseFrontend::Success()) {
    pBufferSize = (int *)CusparseFrontend::GetOutputDevicePointer();
  }
  return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseZgemvi_bufferSize(cusparseHandle_t handle, cusparseOperation_t trans, int m, int n, int nnz, int* pBufferSize) {
  CusparseFrontend::Prepare();
  CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
  CusparseFrontend::AddVariableForArguments<cusparseOperation_t>(trans);
  CusparseFrontend::AddVariableForArguments<int>(m);
  CusparseFrontend::AddVariableForArguments<int>(n);
  CusparseFrontend::AddVariableForArguments<int>(nnz);
  CusparseFrontend::Execute("cusparseZgemvi_bufferSize");
  if (CusparseFrontend::Success()) {
    pBufferSize = (int *)CusparseFrontend::GetOutputDevicePointer();
  }
  return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseSgemvi(cusparseHandle_t handle, cusparseOperation_t  transA, int m, int n, const float* alpha, const float* A, int lda, int nnz, const float* x, const int* xInd, const float* beta, float* y, cusparseIndexBase_t idxBase, void*  pBuffer) {
  CusparseFrontend::Prepare();
  CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
  CusparseFrontend::AddVariableForArguments<cusparseOperation_t>(transA);
  CusparseFrontend::AddVariableForArguments<int>(m);
  CusparseFrontend::AddVariableForArguments<int>(n);
  CusparseFrontend::AddHostPointerForArguments(const_cast<float *>(alpha));
  CusparseFrontend::AddDevicePointerForArguments(A);
  CusparseFrontend::AddVariableForArguments<int>(lda);
  CusparseFrontend::AddVariableForArguments<int>(nnz);
  CusparseFrontend::AddDevicePointerForArguments(x);
  CusparseFrontend::AddDevicePointerForArguments(xInd);
  CusparseFrontend::AddHostPointerForArguments(const_cast<float *>(beta));
  CusparseFrontend::AddDevicePointerForArguments(y);
  CusparseFrontend::AddVariableForArguments<cusparseIndexBase_t>(idxBase);
  CusparseFrontend::AddDevicePointerForArguments(pBuffer);
  CusparseFrontend::Execute("cusparseSgemvi");
  if (CusparseFrontend::Success()) {
      y = (float *)CusparseFrontend::GetOutputDevicePointer();
  }
  return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseDgemvi(cusparseHandle_t handle, cusparseOperation_t  transA, int m, int n, const double* alpha, const double* A, int lda, int nnz, const double* x, const int* xInd, const double* beta, double* y, cusparseIndexBase_t idxBase, void*  pBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<cusparseOperation_t>(transA);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddVariableForArguments<int>(n);
    CusparseFrontend::AddHostPointerForArguments(const_cast<double *>(alpha));
    CusparseFrontend::AddDevicePointerForArguments(A);
    CusparseFrontend::AddVariableForArguments<int>(lda);
    CusparseFrontend::AddVariableForArguments<int>(nnz);
    CusparseFrontend::AddDevicePointerForArguments(x);
    CusparseFrontend::AddDevicePointerForArguments(xInd);
    CusparseFrontend::AddHostPointerForArguments(const_cast<double *>(beta));
    CusparseFrontend::AddDevicePointerForArguments(y);
    CusparseFrontend::AddVariableForArguments<cusparseIndexBase_t>(idxBase);
    CusparseFrontend::AddDevicePointerForArguments(pBuffer);
    CusparseFrontend::Execute("cusparseDgemvi");
    if (CusparseFrontend::Success()) {
        y = (double *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseCgemvi(cusparseHandle_t handle, cusparseOperation_t  transA, int m, int n, const cuComplex* alpha, const cuComplex* A, int lda, int nnz, const cuComplex* x, const int* xInd, const cuComplex* beta, cuComplex* y, cusparseIndexBase_t idxBase, void*  pBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<cusparseOperation_t>(transA);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddVariableForArguments<int>(n);
    CusparseFrontend::AddHostPointerForArguments(const_cast<cuComplex *>(alpha));
    CusparseFrontend::AddDevicePointerForArguments(A);
    CusparseFrontend::AddVariableForArguments<int>(lda);
    CusparseFrontend::AddVariableForArguments<int>(nnz);
    CusparseFrontend::AddDevicePointerForArguments(x);
    CusparseFrontend::AddDevicePointerForArguments(xInd);
    CusparseFrontend::AddHostPointerForArguments(const_cast<cuComplex *>(beta));
    CusparseFrontend::AddDevicePointerForArguments(y);
    CusparseFrontend::AddVariableForArguments<cusparseIndexBase_t>(idxBase);
    CusparseFrontend::AddDevicePointerForArguments(pBuffer);
    CusparseFrontend::Execute("cusparseCgemvi");
    if (CusparseFrontend::Success()) {
        y = (cuComplex *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseZgemvi(cusparseHandle_t handle, cusparseOperation_t  transA, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, int nnz, const cuDoubleComplex* x, const int* xInd, const cuDoubleComplex* beta, cuDoubleComplex* y, cusparseIndexBase_t idxBase, void*  pBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<size_t>((size_t)handle);
    CusparseFrontend::AddVariableForArguments<cusparseOperation_t>(transA);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddVariableForArguments<int>(n);
    CusparseFrontend::AddHostPointerForArguments(const_cast<cuDoubleComplex *>(alpha));
    CusparseFrontend::AddDevicePointerForArguments(A);
    CusparseFrontend::AddVariableForArguments<int>(lda);
    CusparseFrontend::AddVariableForArguments<int>(nnz);
    CusparseFrontend::AddDevicePointerForArguments(x);
    CusparseFrontend::AddDevicePointerForArguments(xInd);
    CusparseFrontend::AddHostPointerForArguments(const_cast<cuDoubleComplex *>(beta));
    CusparseFrontend::AddDevicePointerForArguments(y);
    CusparseFrontend::AddVariableForArguments<cusparseIndexBase_t>(idxBase);
    CusparseFrontend::AddDevicePointerForArguments(pBuffer);
    CusparseFrontend::Execute("cusparseZgemvi");
    if (CusparseFrontend::Success()) {
        y = (cuDoubleComplex *)CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}