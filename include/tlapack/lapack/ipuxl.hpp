/// @file getri_uxlinv_recursive.hpp
/// @author Ali Lotfi, University of Colorado Denver, USA
//
// Copyright (c) 2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_IPUXL_HH
#define TLAPACK_IPUXL_HH

#include "tlapack/base/utils.hpp"

#include "tlapack/blas/gemv.hpp"
#include "tlapack/blas/dotu.hpp"
#include "tlapack/blas/copy.hpp"
#include "tlapack/blas/swap.hpp"
#include "tlapack/blas/gemm.hpp"
#include "tlapack/lapack/ismm.hpp"

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
int ipuxl( matrix_t& A){
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
            return 1;
        // inverse of 1-by-1 case
        A(0,0) = T(1)/A(0,0);
        return 0;
    }
    else{
        
        idx_t k0 = n/2;
        
        auto X00 = tlapack::slice(A,tlapack::range<idx_t>(0,k0),tlapack::range<idx_t>(0,k0));
        auto X01 = tlapack::slice(A,tlapack::range<idx_t>(0,k0),tlapack::range<idx_t>(k0,n));
        auto X10 = tlapack::slice(A,tlapack::range<idx_t>(k0,n),tlapack::range<idx_t>(0,k0));
        auto X11 = tlapack::slice(A,tlapack::range<idx_t>(k0,n),tlapack::range<idx_t>(k0,n));    
    
        //step1:
        // X10 <--- X11^{-1} X10
        trsm(Side::Left,Uplo::Upper,Op::NoTrans,Diag::NonUnit,T(1),X11,X10);

        //step2:
        // X01 <--- -X00^{-1}X01
        trsm(Side::Left,Uplo::Upper,Op::NoTrans,Diag::NonUnit,T(-1),X00,X01);
        

        //step3:
        // recursive call, solve UX=L where U is the upper part of X00(including the diagonal), and L is the part of X00
        // result is stored in X00
        int info=ipuxl(X00);
        if(info!=0){
            std::cout<<"error";
            return info;
        }

        //step4:
        // X00 <--X00 + X01*X10
        gemm(Op::NoTrans,Op::NoTrans,T(1),X01,X10,T(1),X00);

        //step5:
        //solve for X U(X11) = X01,    X01<---X
        trsm(Side::Right,Uplo::Upper,Op::NoTrans,Diag::NonUnit,T(1),X11,X01);
        
        //step6:
        trmm(Side::Right,Uplo::Lower,Op::NoTrans,Diag::Unit,T(1),X11,X01);

        //step7:
        // recursive call
        info=ipuxl(X11);
        if(info!=0){
            std::cout<<"issue";
            return info+k0;
        }

        return 0;


    
    }
        
    return 0;
    
} // ipuxl

} // lapack

#endif // TLAPACK_IPUXL_HH