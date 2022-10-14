/// @file test_getrf.cpp
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
#include <tlapack/lapack/getrf_recursive.hpp>
#include <testutils.hpp>
#include <testdefinitions.hpp>

using namespace tlapack;
using namespace std;

TEMPLATE_LIST_TEST_CASE("LU factorization of a general m-by-n matrix, blocked", "[lqf]", types_to_test)
{
    srand(1);
    using matrix_t = TestType;
    using T = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;
    typedef real_type<T> real_t; // equivalent to using real_t = real_type<T>;
    
    // m and n represent no. rows and columns of the matrices we will be testing respectively
    idx_t m, n;
    m = GENERATE(5,10,20,30,100);
    n = m;

    // eps is the machine precision, and tol is the tolerance we accept for tests to pass
    const real_t eps = ulp<real_t>();
    const real_t tol = 10*max(m, n)*eps;
    
    // Initialize matrices A, and A_copy to run tests on
    std::unique_ptr<T[]> A_(new T[m * n]);
    std::unique_ptr<T[]> A_copy_(new T[m * n]);
    auto A = legacyMatrix<T, layout<matrix_t>>(m, n, &A_[0], layout<matrix_t> == Layout::ColMajor ? m : n);
    auto A_copy = legacyMatrix<T, layout<matrix_t>>(m, n, &A_copy_[0], layout<matrix_t> == Layout::ColMajor ? m : n);
    // forming A, a random matrix with diagonal of 1
    for (idx_t j = 0; j < n; ++j)
        for (idx_t i = 0; i < m; ++i){
            if(i==j){
                A(i, j) = rand_helper<T>();

            }
            else{
                A(i, j) = rand_helper<T>();

            }
            
        }
            

    
    // We will make a deep copy A
    // We intend to test A=LU, however, since after calling getrf, A will be udpated
    // then to test A=LU, we'll make a deep copy of A prior to calling getrf
    lacpy(Uplo::General, A, A_copy);
    double norma=tlapack::lange( tlapack::Norm::Max, A);
    
    // Put diagonal and super-diagonal of A into U and sub-diagonal in L  
    std::unique_ptr<T[]> L_(new T[m * n]);
    std::unique_ptr<T[]> U_copy_(new T[m * n]);
    auto L = legacyMatrix<T, layout<matrix_t>>(m, n, &L_[0], layout<matrix_t> == Layout::ColMajor ? m : n);
    auto U = legacyMatrix<T, layout<matrix_t>>(m, n, &U_copy_[0], layout<matrix_t> == Layout::ColMajor ? m : n);
    for (idx_t j = 0; j < n; ++j){
        for (idx_t i = 0; i < m; ++i){
            if(i==j){
                L(i, j) = T(1);
                U(i, j) = A(i,j);

            }
            else if (i>j)
            {
                L(i, j) = A(i,j);
                U(i, j) = T(0);
            }
            else{
                L(i, j) =T(0);
                U(i, j) = A(i,j);

            }
            
        }
    }
    // run ul_mult, which calculates U and L in place of A
    ul_mult(A);
    
    // store UL-A ---> A 
    gemm(Op::NoTrans,Op::NoTrans,T(1),U,L,T(-1),A);
    // getri_methodD(A);


    real_t error1 = tlapack::lange( tlapack::Norm::Max, A)/norma;
    CHECK(error1/tol <= 1);
    
}



