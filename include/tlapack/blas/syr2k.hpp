// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_BLAS_SYR2K_HH
#define TLAPACK_BLAS_SYR2K_HH

#include "tlapack/base/utils.hpp"

namespace tlapack {

/**
 * Symmetric rank-k update:
 * \[
 *     C := \alpha A B^T + \alpha B A^T + \beta C,
 * \]
 * or
 * \[
 *     C := \alpha A^T B + \alpha B^T A + \beta C,
 * \]
 * where alpha and beta are scalars, C is an n-by-n symmetric matrix,
 * and A and B are n-by-k or k-by-n matrices.
 *
 * @param[in] uplo
 *     What part of the matrix C is referenced,
 *     the opposite triangle being assumed from symmetry:
 *     - Uplo::Lower: only the lower triangular part of C is referenced.
 *     - Uplo::Upper: only the upper triangular part of C is referenced.
 *
 * @param[in] trans
 *     The operation to be performed:
 *     - Op::NoTrans: $C = \alpha A B^T + \alpha B A^T + \beta C$.
 *     - Op::Trans:   $C = \alpha A^T B + \alpha B^T A + \beta C$.
 *
 * @param[in] alpha Scalar.
 * @param[in] A A n-by-k matrix.
 *     - If trans = NoTrans: a n-by-k matrix.
 *     - Otherwise:          a k-by-n matrix.
 * @param[in] B A n-by-k matrix.
 *     - If trans = NoTrans: a n-by-k matrix.
 *     - Otherwise:          a k-by-n matrix.
 * @param[in] beta Scalar.
 * @param[in,out] C A n-by-n symmetric matrix.
 *
 * @ingroup syr2k
 */
template<
    class matrixA_t, class matrixB_t, class matrixC_t, 
    class alpha_t, class beta_t,
    class T = type_t<matrixC_t>,
    disable_if_allow_optblas_t<
        pair< matrixA_t, T >,
        pair< matrixB_t, T >,
        pair< matrixC_t, T >,
        pair< alpha_t,   T >,
        pair< beta_t,    T >
    > = 0
>
void syr2k(
    Uplo uplo,
    Op trans,
    const alpha_t& alpha, const matrixA_t& A, const matrixB_t& B,
    const beta_t& beta, matrixC_t& C )
{
    // data traits
    using TA    = type_t< matrixA_t >;
    using TB    = type_t< matrixB_t >;
    using idx_t = size_type< matrixA_t >;

    // constants
    const idx_t n = (trans == Op::NoTrans) ? nrows(A) : ncols(A);
    const idx_t k = (trans == Op::NoTrans) ? ncols(A) : nrows(A);

    // check arguments
    tlapack_check_false( uplo != Uplo::Lower &&
                   uplo != Uplo::Upper &&
                   uplo != Uplo::General );
    tlapack_check_false( trans != Op::NoTrans &&
                   trans != Op::Trans );
    tlapack_check_false( nrows(B) != nrows(A) ||
                   ncols(B) != ncols(A) );
    tlapack_check_false( nrows(C) != ncols(C) );
    tlapack_check_false( nrows(C) != n );

    tlapack_check_false( access_denied( dense, read_policy(A) ) );
    tlapack_check_false( access_denied( dense, read_policy(B) ) );
    tlapack_check_false( access_denied( uplo, write_policy(C) ) );

    if (trans == Op::NoTrans) {
        if (uplo != Uplo::Lower) {
        // uplo == Uplo::Upper or uplo == Uplo::General
            for(idx_t j = 0; j < n; ++j) {

                for(idx_t i = 0; i <= j; ++i)
                    C(i,j) *= beta;

                for(idx_t l = 0; l < k; ++l) {
                    auto alphaBjl = alpha*B(j,l);
                    auto alphaAjl = alpha*A(j,l);
                    for(idx_t i = 0; i <= j; ++i)
                        C(i,j) += A(i,l)*alphaBjl + B(i,l)*alphaAjl;
                }
            }
        }
        else { // uplo == Uplo::Lower
            for(idx_t j = 0; j < n; ++j) {

                for(idx_t i = j; i < n; ++i)
                    C(i,j) *= beta;

                for(idx_t l = 0; l < k; ++l) {
                    auto alphaBjl = alpha*B(j,l);
                    auto alphaAjl = alpha*A(j,l);
                    for(idx_t i = j; i < n; ++i)
                        C(i,j) += A(i,l)*alphaBjl + B(i,l)*alphaAjl;
                }
            }
        }
    }
    else { // trans == Op::Trans
        using scalar_t = scalar_type<TA,TB>;

        if (uplo != Uplo::Lower) {
        // uplo == Uplo::Upper or uplo == Uplo::General
            for(idx_t j = 0; j < n; ++j) {
                for(idx_t i = 0; i <= j; ++i) {

                    scalar_t sum1( 0 );
                    scalar_t sum2( 0 );
                    for(idx_t l = 0; l < k; ++l) {
                        sum1 += A(l,i) * B(l,j);
                        sum2 += B(l,i) * A(l,j);
                    }
                    C(i,j) = alpha*sum1 + alpha*sum2 + beta*C(i,j);
                }
            }
        }
        else { // uplo == Uplo::Lower
            for(idx_t j = 0; j < n; ++j) {
                for(idx_t i = j; i < n; ++i) {

                    scalar_t sum1( 0 );
                    scalar_t sum2( 0 );
                    for(idx_t l = 0; l < k; ++l) {
                        sum1 +=  A(l,i) * B(l,j);
                        sum2 +=  B(l,i) * A(l,j);
                    }
                    C(i,j) = alpha*sum1 + alpha*sum2 + beta*C(i,j);
                }
            }
        }
    }

    if (uplo == Uplo::General) {
        for(idx_t j = 0; j < n; ++j) {
            for(idx_t i = j+1; i < n; ++i)
                C(i,j) = C(j,i);
        }
    }
}

/**
 * Symmetric rank-k update:
 * \[
 *     C := \alpha A B^T + \alpha B A^T,
 * \]
 * or
 * \[
 *     C := \alpha A^T B + \alpha B^T A,
 * \]
 * where alpha and beta are scalars, C is an n-by-n symmetric matrix,
 * and A and B are n-by-k or k-by-n matrices.
 *
 * @param[in] uplo
 *     What part of the matrix C is referenced,
 *     the opposite triangle being assumed from symmetry:
 *     - Uplo::Lower: only the lower triangular part of C is referenced.
 *     - Uplo::Upper: only the upper triangular part of C is referenced.
 *
 * @param[in] trans
 *     The operation to be performed:
 *     - Op::NoTrans: $C = \alpha A B^T + \alpha B A^T$.
 *     - Op::Trans:   $C = \alpha A^T B + \alpha B^T A$.
 *
 * @param[in] alpha Scalar.
 * @param[in] A A n-by-k matrix.
 *     - If trans = NoTrans: a n-by-k matrix.
 *     - Otherwise:          a k-by-n matrix.
 * @param[in] B A n-by-k matrix.
 *     - If trans = NoTrans: a n-by-k matrix.
 *     - Otherwise:          a k-by-n matrix.
 * @param[in] beta Scalar.
 * @param[out] C A n-by-n symmetric matrix.
 *
 * @ingroup syr2k
 */
template<
    class matrixA_t, class matrixB_t, class matrixC_t, 
    class alpha_t,
    class T = type_t<matrixC_t>,
    disable_if_allow_optblas_t<
        pair< matrixA_t, T >,
        pair< matrixB_t, T >,
        pair< matrixC_t, T >,
        pair< alpha_t,   T >
    > = 0
>
inline
void syr2k(
    Uplo uplo,
    Op trans,
    const alpha_t& alpha, const matrixA_t& A, const matrixB_t& B,
    matrixC_t& C )
{
    return syr2k( uplo, trans, alpha, A, B, internal::StrongZero(), C );
}

}  // namespace tlapack

#endif        //  #ifndef TLAPACK_BLAS_SYR2K_HH
