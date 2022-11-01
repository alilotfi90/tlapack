/// @file getri_uxlinv_recursive.hpp
/// @author Ali Lotfi, University of Colorado Denver, USA
//
// Copyright (c) 2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_ISMM_HH
#define TLAPACK_ISMM_HH

#include "tlapack/base/utils.hpp"

#include "tlapack/blas/gemv.hpp"
#include "tlapack/blas/dotu.hpp"
#include "tlapack/blas/copy.hpp"
#include "tlapack/blas/swap.hpp"
#include "tlapack/blas/gemm.hpp"

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
    template< class alpha_t , class matrix_t , class work_t >
    void ismm( const alpha_t& alpha, matrix_t& A, matrix_t& B , work_t& work ){
        using idx_t = size_type< matrix_t >;
        using T = type_t<matrix_t>;

        // check arguments
        tlapack_check_false( access_denied( dense, write_policy(A) ) );
        tlapack_check( nrows(A)==ncols(A));
        tlapack_check( nrows(B)==ncols(B));
        tlapack_check( ncols(A)==nrows(B));
        tlapack_check( size(work) >= ncols(A) - 1);

        // constant n, number of rows and also columns of A
        const idx_t n = ncols(A);

        for (idx_t j = 0; j < n; ++j){
                auto a = tlapack::slice(A,j,tlapack::range<idx_t>(0,n));
            

                // first step of the algorithm, work1 holds x12
                tlapack::gemv(Op::Trans,alpha, B, a,T(0), work);
                tlapack::copy(work,a);

        }

    } // ismm

} // lapack

#endif // TLAPACK_ISMM_HH

