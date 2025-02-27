/// @file test_gelqf.cpp
/// @brief Test GELQF and UNGL2 and output a k-by-n orthogonal matrix Q.
//
// Copyright (c) 2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <tlapack/plugins/stdvector.hpp>
#include <tlapack/plugins/legacyArray.hpp>
#include <tlapack.hpp>
#include <testutils.hpp>
#include <testdefinitions.hpp>

using namespace tlapack;

TEMPLATE_LIST_TEST_CASE("LQ factorization of a general m-by-n matrix, blocked", "[lqf]", types_to_test)
{
    srand(1);

    using matrix_t = TestType;
    using T = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;
    using range = std::pair<idx_t, idx_t>;
    typedef real_type<T> real_t;

    const T zero(0);

    idx_t m, n, k, nb;

    m = GENERATE(10, 20, 30);
    n = GENERATE(10, 20, 30);
    k = GENERATE(8, 10, 20, 30); // k is the number of rows for output Q. Can personalize it.
    nb = GENERATE(2, 3, 7, 12);  // nb is the block height. Can personalize it.

    const real_t eps = ulp<real_t>();
    const real_t tol = max(m, n) * eps;

    std::unique_ptr<T[]> A_(new T[m * n]);
    std::unique_ptr<T[]> A_copy_(new T[m * n]);
    std::unique_ptr<T[]> TT_(new T[m * nb]);

    auto A = legacyMatrix<T, layout<matrix_t>>(m, n, &A_[0], layout<matrix_t> == Layout::ColMajor ? m : n);
    auto A_copy = legacyMatrix<T, layout<matrix_t>>(m, n, &A_copy_[0], layout<matrix_t> == Layout::ColMajor ? m : n);
    auto TT = legacyMatrix<T, layout<matrix_t>>(m, nb, &TT_[0], layout<matrix_t> == Layout::ColMajor ? m : nb);

    std::vector<T> work_gelqf(m);
    std::vector<T> work_ungl2(k);

    std::vector<T> tauw(min(m, n));

    for (idx_t j = 0; j < n; ++j)
        for (idx_t i = 0; i < m; ++i)
            A(i, j) = rand_helper<T>();

    lacpy(Uplo::General, A, A_copy);

    if (k <= n) // k must be less than or equal to n, because we cannot get a Q bigger than n-by-n
    {
        DYNAMIC_SECTION("m = " << m << " n = " << n << " k = " << k << " nb = " << nb)
        {
            gelqf(A, TT, work_gelqf, nb);

            // Build tauw vector from matrix TT
            for (idx_t j = 0; j < min(m,n); j += nb)
            {
                idx_t ib = std::min<idx_t>(nb, min(m,n) - j);
                
                for (idx_t i = 0; i < ib; i++)
                    tauw[i+j] = TT(i+j,i);
            }

            // Q is sliced down to the desired size of output Q (k-by-n).
            // It stores the desired number of Householder reflectors that UNGL2 will use.
            std::unique_ptr<T[]> Q_(new T[k * n]);
            auto Q = legacyMatrix<T, layout<matrix_t>>(k, n, &Q_[0], layout<matrix_t> == Layout::ColMajor ? k : n);
            lacpy(Uplo::General, slice(A, range(0, min(m, k)), range(0, n)), Q);

            ungl2(Q, tauw, work_ungl2);

            // Wq is the identity matrix to check the orthogonality of Q
            std::unique_ptr<T[]> Wq_(new T[k * k]);
            auto Wq = legacyMatrix<T, layout<matrix_t>>(k, k, &Wq_[0], k);
            auto orth_Q = check_orthogonality(Q, Wq);
            CHECK(orth_Q <= tol);

            // L is sliced from A after GELQ2
            std::unique_ptr<T[]> L_(new T[min(k, m) * k]);
            auto L = legacyMatrix<T, layout<matrix_t>>(min(k, m), k, &L_[0], layout<matrix_t> == Layout::ColMajor ? min(k, m) : k);
            laset(Uplo::Upper, zero, zero, L);
            lacpy(Uplo::Lower, slice(A, range(0, min(m, k)), range(0, k)), L);

            // R stores the product of L and Q
            std::unique_ptr<T[]> R_(new T[min(k, m) * n]);
            auto R = legacyMatrix<T, layout<matrix_t>>(min(k, m), n, &R_[0], layout<matrix_t> == Layout::ColMajor ? min(k, m) : n);

            // Test A = L * Q
            gemm(Op::NoTrans, Op::NoTrans, real_t(1.), L, Q, R);
            for (idx_t j = 0; j < n; ++j)
                for (idx_t i = 0; i < min(m, k); ++i)
                    A_copy(i, j) -= R(i, j);

            real_t repres = tlapack::lange(tlapack::Norm::Max, slice(A_copy, range(0, min(m, k)), range(0, n)));
            CHECK(repres <= tol);

        }
    }
}
