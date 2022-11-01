/// @file getri_uxlinv_recursive.hpp
/// @author Ali Lotfi, University of Colorado Denver, USA
//
// Copyright (c) 2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_GETRI_UXLINV_RECURSIVE_HH
#define TLAPACK_GETRI_UXLINV_RECURSIVE_HH

#include "tlapack/base/utils.hpp"

#include "tlapack/blas/gemv.hpp"
#include "tlapack/blas/dotu.hpp"
#include "tlapack/blas/copy.hpp"
#include "tlapack/blas/swap.hpp"
#include "tlapack/blas/gemm.hpp"
#include "tlapack/lapack/trtri_recursive.hpp"
//#include "tlapack/lapack/ismm.hpp"

namespace tlapack {

/** getri_uxli_recursive is the recursive version of getri_uxli, it was devleped by Dr. Weslley Pereira 
 *  getri_uxli_recursive computes the inverse of a general n-by-n matrix A recursively
 *  the L and U factors of a matrix is given on the input
 *  then we solve for X in the following equation
 * \[
 *   U X L = I
 * \]
 * Notice that from LU, we have PA=LU and as a result
 * \[
 *   U (A^{-1} P^{T}) L = I
 * \]
 * last equation means that $A^{-1} P^{T}=X$, therefore, to solve for $A^{-1}$
 * we just need to swap the columns of X according to $X=A^{-1} P$
 *
 * @return  0 if success
 * @return  -1 if matrix is not invertible
 *
 * @param[in,out] A n-by-n complex matrix.
 *      
 * @ingroup group_solve
 */
template< class matrix_t>
int getri_uxlinv_recursive( matrix_t& A){
    using idx_t = size_type< matrix_t >;
    using T = type_t<matrix_t>;

    // check arguments
    tlapack_check_false( access_denied( dense, write_policy(A) ) );
    tlapack_check( nrows(A)==ncols(A));
    
    // constant n, number of rows and also columns of A
    const idx_t n = ncols(A);

    trtri_recursive(Uplo::Lower, Diag::Unit, A);

    ipuxl(A);

    return 0;

    
    
} // 

} // lapack

#endif // TLAPACK_GETRI_UXLINV_RECURSIVE_HH





