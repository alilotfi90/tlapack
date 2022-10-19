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
int getri_uxli_recursive( matrix_t& A){
    using idx_t = size_type< matrix_t >;
    using T = type_t<matrix_t>;

    // check arguments
    tlapack_check_false( access_denied( dense, write_policy(A) ) );
    tlapack_check( nrows(A)==ncols(A));
    
    // constant n, number of rows and also columns of A
    const idx_t n = ncols(A);

    if(n==1){
        // 1-by-1 non-invertible case
        if(A(0,0)==T(0))
            return -1;
        // inverse of 1-by-1 case
        A(0,0) = T(1)/A(0,0); 
    }
    else{
        
        // n>1 case
        // breaking the matrix into four blocks
        idx_t k0 = n/2;
        auto A00 = tlapack::slice(A,tlapack::range<idx_t>(0,k0),tlapack::range<idx_t>(0,k0));
        auto U01 = tlapack::slice(A,tlapack::range<idx_t>(0,k0),tlapack::range<idx_t>(k0,n));
        auto L10 = tlapack::slice(A,tlapack::range<idx_t>(k0,n),tlapack::range<idx_t>(0,k0));
        auto A11 = tlapack::slice(A,tlapack::range<idx_t>(k0,n),tlapack::range<idx_t>(k0,n));
        
        //step1:
        // the following three lines calculates - U_11^{-1} (L_11^{-1} (L_10^{-1} L_00^{-1})) 
        // L00 is the subdiagonal of A_00 with 1 on the diagonal
        // L11 is the subdiagonal of A_11 with 1 on the diagonal
        // U11 is the superdiagonal and diagonal of A_11
        
        // compute L10 L00^{-1}
        trsm(Side::Right,Uplo::Lower,Op::NoTrans,Diag::Unit,T(1),A00,L10);

        // compute L11^{-1} L10
        trsm(Side::Left,Uplo::Lower,Op::NoTrans,Diag::Unit,T(1),A11,L10);

        // compute U11^{-1} L10
        trsm(Side::Left,Uplo::Upper,Op::NoTrans,Diag::NonUnit,T(-1),A11,L10);
        
        //step2: compute U00^{-1} U01
        // U00 is the diagonal and superdiagonal of A00  
        trsm(Side::Left,Uplo::Upper,Op::NoTrans,Diag::NonUnit,T(1),A00,U01);
    
        //step3: recursive call on A00
        getri_uxli_recursive(A00);

        //step4:
        // A00 <---- A00 - (U01 * L10)
        gemm(Op::NoTrans,Op::NoTrans,T(-1),U01,L10,T(1),A00);
    

        //step5: the two following computes -(U01 U11^{-1}) L_11^{-1}
        trsm(Side::Right,Uplo::Upper,Op::NoTrans,Diag::NonUnit,T(1),A11,U01);

        trsm(Side::Right,Uplo::Lower,Op::NoTrans,Diag::Unit,T(-1),A11,U01);
        
        //step6:recursive call on A11
        getri_uxli_recursive(A11);
    
    
    }
        
    return 0;
    
} // getri_uxli_recursive

} // lapack

#endif // TLAPACK_GETRI_UXLI_RECURSIVE_HH



