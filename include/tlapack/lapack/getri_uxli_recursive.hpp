/// @file getri_uxli_recursive.hpp
/// @author Ali Lotfi, University of Colorado Denver, USA
//
// Copyright (c) 2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_GETRI_UXLI_RECURSIVE_HH
#define TLAPACK_GETRI_UXLI_RECURSIVE_HH

#include "tlapack/base/utils.hpp"

#include "tlapack/blas/gemv.hpp"
#include "tlapack/blas/dotu.hpp"
#include "tlapack/blas/copy.hpp"
#include "tlapack/blas/swap.hpp"
#include "tlapack/blas/gemm.hpp"

namespace tlapack {

/** getri computes inverse of a general n-by-n matrix A
 *  using LU factorization. 
 *  we first run LU in place of A
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
 * @param[in,out] Piv Piv vector of size at least n.
 * 
 * @param[in,out] work a vector of size at least n-1.
 *      
 * @ingroup group_solve
 */
template< class matrix_t, class vector_t >
int getri_uxli_recursive( matrix_t& A, vector_t &Piv ){
    using idx_t = size_type< matrix_t >;
    using T = type_t<matrix_t>;

    // check arguments
    tlapack_check_false( access_denied( dense, write_policy(A) ) );
    tlapack_check( nrows(A)==ncols(A));
    
    // constant n, number of rows and also columns of A
    const idx_t n = ncols(A);

    if(n==1){
        if(A(0,0)==T(0))
            return -1;
        A(0,0) = T(1)/A(0,0); 
    }
    else{
        idx_t k0 = n/2;
        auto A00 = tlapack::slice(A,tlapack::range<idx_t>(0,k0),tlapack::range<idx_t>(0,k0));
        auto U01 = tlapack::slice(A,tlapack::range<idx_t>(0,k0),tlapack::range<idx_t>(k0,n));
        auto L10 = tlapack::slice(A,tlapack::range<idx_t>(k0,n),tlapack::range<idx_t>(0,k0));
        auto A11 = tlapack::slice(A,tlapack::range<idx_t>(k0,n),tlapack::range<idx_t>(k0,n));
        // Solve triangular system A0 X = A1 and update A1
        // trsm(Side::Left,Uplo::Lower,Op::NoTrans,Diag::Unit,T(1),A0,A1);
        // Solve triangular system X L00 = L10 and update L10
        trsm(Side::Right,Uplo::Lower,Op::NoTrans,Diag::Unit,T(1),A00,L10);

        // Solve triangular system L11 X = L10 and update L10
        trsm(Side::Left,Uplo::Lower,Op::NoTrans,Diag::Unit,T(1),A11,L10);

        // Solve triangular system U11 X = -L10 and update L10
        trsm(Side::Left,Uplo::Upper,Op::NoTrans,Diag::NonUnit,T(-1),A11,L10);

        
        // Solve triangular system U00 X = U01 and update U01
        trsm(Side::Left,Uplo::Upper,Op::NoTrans,Diag::NonUnit,T(1),A00,U01);
    
        getri_uxli_recursive(A00, Piv);

        // A00 <---- A00 - (U01 * L10)
        gemm(Op::NoTrans,Op::NoTrans,T(-1),U01,L10,T(1),A00);
    

        trsm(Side::Right,Uplo::Upper,Op::NoTrans,Diag::NonUnit,T(1),A11,U01);

        trsm(Side::Right,Uplo::Lower,Op::NoTrans,Diag::Unit,T(-1),A11,U01);
        
        getri_uxli_recursive(A11, Piv);
    
    
    
    
    
    
    
    
    
    }
        
    return 0;
    
} // getri_uxli_recursive

} // lapack

#endif // TLAPACK_GETRI_UXLI_RECURSIVE_HH



