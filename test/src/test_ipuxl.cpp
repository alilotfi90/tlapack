/// @file test_getri.cpp
/// @brief Test functions that calculate inverse of matrices such as getri family.
// Copyright (c) 2022, University of Colorado Denver. All rights reserved.
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
    
    //n represent no. rows and columns of the square matrices we will performing tests on
    idx_t n;
    n = GENERATE(1,2,3,4,5,10,20,100);

    // eps is the machine precision, and tol is the tolerance we accept for tests to pass
    const real_t eps = ulp<real_t>();
    const real_t tol = n*n*eps;
    
    // Initialize matrices A, and A_copy to run tests on
    std::unique_ptr<T[]> A_(new T[n * n]);
    std::unique_ptr<T[]> A_copy_(new T[n * n]);
    auto A = legacyMatrix<T, layout<matrix_t>>(n, n, &A_[0], layout<matrix_t> == Layout::ColMajor ? n : n);
    //auto X = legacyMatrix<T, layout<matrix_t>>(n, n, &X_[0], layout<matrix_t> == Layout::ColMajor ? n : n);

    
    // forming A, a random matrix 
    for (idx_t j = 0; j < n; ++j)
        for (idx_t i = 0; i < n; ++i){
            A(i, j) = rand_helper<T>();
            // if(i==j){
            //     A(i,j)=T(0);
            // }
        }
    

    // calculate norm of A for later use in relative error
    double norma=tlapack::lange( tlapack::Norm::Max, A);

    // building identity matrix
    std::unique_ptr<T[]> U_(new T[n * n]);
    auto U = legacyMatrix<T, layout<matrix_t>>(n, n, &U_[0], layout<matrix_t> == Layout::ColMajor ? n : n);
    for (idx_t j = 0; j < n; ++j)
        for (idx_t i = 0; i < n; ++i){
            if(i<=j){
                U(i, j) = A(i,j);
            }
            else{
                U(i, j) = T(0);
            }
            
        }

    // tlapack::trtri_recursive( Uplo::Lower, Diag::Unit, A );
    
    // building identity matrix
    std::unique_ptr<T[]> L_(new T[n * n]);
    auto L = legacyMatrix<T, layout<matrix_t>>(n, n, &L_[0], layout<matrix_t> == Layout::ColMajor ? n : n);
    for (idx_t j = 0; j < n; ++j)
        for (idx_t i = 0; i < n; ++i){
            if(i>j){
                L(i, j) = A(i,j);
            }
            else{
                if(i==j){
                    L(i,j)=T(1);
                }
                else{
                    L(i, j) = T(0);
                }
                
            }
            
        }
    
    // make a deep copy A
    //lacpy(Uplo::General, A, A_copy);
    
    ipuxl(A);

    trsm(Side::Left,Uplo::Upper,Op::NoTrans,Diag::NonUnit,T(1),U,L);
    
    
    
    
    
    // identit1 -----> A * A_copy - ident1
    //gemm(Op::NoTrans,Op::NoTrans,real_t(1),A,A_copy,real_t(-1),L);
    for (idx_t j = 0; j < n; ++j)
        for (idx_t i = 0; i < n; ++i){
            L(i,j)=L(i,j)-A(i,j);
            
        }



    // error1 is  || A * A_copy - ident1 || / ||A||   
    real_t error1 = tlapack::lange( tlapack::Norm::Max, L)/norma;
    
    INFO( "n = " << n );

    // following tests if error1<=tol
    CHECK(error1 <= tol);
    
}



