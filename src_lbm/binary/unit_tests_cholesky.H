# include "LBM_binary.H"
#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MFParallelFor.H>
#include <AMReX_Print.H>

#include <cmath>
#include <math.h>

// in all test cases, success is measured by calculating whether the output array of the cholesky decomposition, labelled L, is identical to a 
// known result and compared to within single precision

// test case 1 looking at assigning positive and identical numbers to diagonal
void test_case_1(){
    GpuArray<Real, ndof*ndof> L_test = {};
    GpuArray<Real, ndof*ndof> L = {};
    
    L_test.fill(0);
    L.fill(0);
    Real atol = pow(10, -7);
    Print() << atol << " tolerance level\n";

    for (int i = ncons + 1; i < ndof; i++){
        L[i*ndof + i] = 2;
        L_test[i*ndof + i] = sqrt(2);
    }

    cholesky_decomp(L, ndof);
    bool test = true;

    Real actual, predict;
    for (int i = 0; i < ndof; i ++){
        actual = L[i*ndof + i];
        predict = L_test[i*ndof + i];
        if (fabs(actual - predict) >= atol){test = false;}
    }
    if(test){Print() << "Test case 1 passes\n";}
    else{Print() << "Test case 1 fails\n";}
}

// test case 2 looking at assigning positive but different numbers to diagonal
void test_case_2(){
    GpuArray<Real, ndof*ndof> L_test = {};
    GpuArray<Real, ndof*ndof> L = {};
    
    L_test.fill(0);
    L.fill(0);
    Real atol = pow(10, -7);

    for (int i = ncons + 1; i < ndof; i++){
        L[i*ndof + i] = i + 1;
        L_test[i*ndof + i] = sqrt(i + 1);
    }

    cholesky_decomp(L, ndof);
    bool test = true;

    Real actual, predict;
    for (int i = 0; i < ndof; i ++){
        actual = L[i*ndof + i];
        predict = L_test[i*ndof + i];
        if (fabs(actual - predict) >= atol){test = false;}
    }
    if(test){Print() << "Test case 2 passes\n";}
    else{Print() << "Test case 2 fails\n";}

}

// test case 3 looking at assigning random numbers to the diagonal
void test_case_3(){
    GpuArray<Real, ndof*ndof> L_test = {};
    GpuArray<Real, ndof*ndof> L = {};
    
    L_test.fill(0);
    L.fill(0);
    Real atol = pow(10, -7);

    for (int i = ncons + 1; i < ndof; i++){
        int val = Random_int(100);
        L[i*ndof + i] = val;
        L_test[i*ndof + i] = sqrt(val);
    }

    cholesky_decomp(L, ndof);
    bool test = true;

    Real actual, predict;
    for (int i = 0; i < ndof; i ++){
        actual = L[i*ndof + i];
        predict = L_test[i*ndof + i];
        if (fabs(actual - predict) >= atol){test = false;}
    }
    if(test){Print() << "Test case 3 passes\n";}
    else{Print() << "Test case 3 fails\n";}
}

// // test case 4 looking at assigning random numbers to an input matrix, ensuring it is positive definite and performing cholesky decomposition on that
// void test_case_4(){

//     GpuArray<Real, ndof*ndof> L_test = {};
//     GpuArray<Real, ndof*ndof> L = {};

//     L_test.fill(0);
//     L.fill(0);
//     Real atol = pow(10, -7);

//     for (int i = ncons + 1; i < ndof; i++){
//         int val = Random_int(100);
//         // L[i*ndof + i] = val;
//         // L_test[i*ndof + i] = sqrt(val);
//     }



//     cholesky_decomp(L, ndof);
//     bool test = true;

//     Real actual, predict;
//     for (int i = 0; i < ndof; i ++){
//         actual = L[i*ndof + i];
//         predict = L_test[i*ndof + i];
//         if (fabs(actual - predict) >= atol){test = false;}
//     }
//     if(test){Print() << "Test case 3 passes\n";}
//     else{Print() << "Test case 3 fails\n";}
// }