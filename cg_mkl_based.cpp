#include "mex.h"    // Required for MATLAB MEX interface
#include "matrix.h" // Required for MATLAB matrix operations
#include <mkl.h>    // Intel MKL library for optimized linear algebra operations

/**
 * @brief Implementation of the Conjugate Gradient method for solving sparse linear systems
 * 
 * The Conjugate Gradient (CG) method is an iterative algorithm for solving
 * symmetric positive-definite linear systems Ax = b. This implementation uses
 * Intel MKL for optimized sparse matrix operations and BLAS routines.
 * 
 * The algorithm works by generating a sequence of orthogonal vectors (conjugate
 * directions) and using them to iteratively improve the solution approximation.
 */
class CGSolver {
private:
    MKL_INT n;              // Size of the linear system
    double* x;              // Solution vector
    double* r;              // Residual vector (b - Ax)
    double* p;              // Search direction
    double* Ap;             // Matrix-vector product (A*p)
    sparse_matrix_t A;      // Sparse matrix in MKL format
    matrix_descr descr;     // Matrix descriptor for MKL sparse operations

public:
    /**
     * @brief Constructor - Initializes vectors and matrix descriptor
     * 
     * @param size Dimension of the linear system
     * 
     * Note: All vectors are aligned to 64-byte boundaries for optimal
     * performance with Intel MKL operations
     */
    CGSolver(MKL_INT size) : n(size) {
        // Allocate aligned memory for vectors using MKL
        x = (double*)mkl_malloc(n * sizeof(double), 64);
        r = (double*)mkl_malloc(n * sizeof(double), 64);
        p = (double*)mkl_malloc(n * sizeof(double), 64);
        Ap = (double*)mkl_malloc(n * sizeof(double), 64);
        
        // Configure matrix descriptor for symmetric upper triangular storage
        descr.type = SPARSE_MATRIX_TYPE_SYMMETRIC;
        descr.mode = SPARSE_FILL_MODE_UPPER;
        descr.diag = SPARSE_DIAG_NON_UNIT;
    }

    /**
     * @brief Destructor - Frees allocated memory
     */
    ~CGSolver() {
        mkl_free(x);
        mkl_free(r);
        mkl_free(p);
        mkl_free(Ap);
    }

    /**
     * @brief Creates the sparse matrix in MKL's internal format
     * 
     * @param ia Row pointers (CSR format)
     * @param ja Column indices (CSR format)
     * @param a  Nonzero values (CSR format)
     * 
     * Converts the input CSR matrix to MKL's optimized sparse format
     * and applies internal optimizations for subsequent operations
     */
    void createMatrix(const MKL_INT* ia, const MKL_INT* ja, const double* a) {
        mkl_sparse_d_create_csr(&A, SPARSE_INDEX_BASE_ZERO, n, n, 
                               const_cast<MKL_INT*>(ia),
                               const_cast<MKL_INT*>(ia + 1),
                               const_cast<MKL_INT*>(ja),
                               const_cast<double*>(a));
        mkl_sparse_optimize(A);  // Apply MKL's internal optimizations
    }

    /**
     * @brief Implements the Conjugate Gradient algorithm
     * 
     * @param b          Right-hand side vector
     * @param solution   Output solution vector
     * @param final_res  Final residual norm
     * @param iterations Number of iterations performed
     * @param tol        Convergence tolerance
     * @param maxit      Maximum number of iterations
     * 
     * Algorithm steps:
     * 1. Initialize x₀ = 0, r₀ = b - Ax₀ = b, p₀ = r₀
     * 2. For k = 0,1,... until convergence:
     *    - αₖ = (rₖᵀrₖ)/(pₖᵀApₖ)         // Step length
     *    - xₖ₊₁ = xₖ + αₖpₖ              // Update solution
     *    - rₖ₊₁ = rₖ - αₖApₖ             // Update residual
     *    - βₖ = (rₖ₊₁ᵀrₖ₊₁)/(rₖᵀrₖ)      // Conjugate direction update
     *    - pₖ₊₁ = rₖ₊₁ + βₖpₖ            // Update search direction
     */
    void solve(const double* b, double* solution, double* final_res, int* iterations,
              double tol, int maxit) {
        // Step 1: Initialize x = 0
        for(MKL_INT i = 0; i < n; i++) {
            x[i] = 0.0;
        }
        
        // Initialize r = b - Ax = b (since x = 0)
        cblas_dcopy(n, b, 1, r, 1);     // r = b
        cblas_dcopy(n, r, 1, p, 1);     // p = r
        
        // Calculate initial residual norm
        double r_norm = cblas_dnrm2(n, r, 1);
        double initial_r_norm = r_norm;
        double r_norm_old;
        
        // Main iteration loop
        int iter;
        for (iter = 0; iter < maxit; ++iter) {
            // Check convergence
            if (r_norm <= tol * initial_r_norm) {
                break;
            }
            
            // Compute Ap = A*p
            mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A, descr, p, 0.0, Ap);
            
            // Compute step length α = (r'r)/(p'Ap)
            double rr = cblas_ddot(n, r, 1, r, 1);    // r'r
            double pAp = cblas_ddot(n, p, 1, Ap, 1);  // p'Ap
            double alpha = rr / pAp;
            
            // Update solution: x = x + αp
            cblas_daxpy(n, alpha, p, 1, x, 1);
            // Update residual: r = r - αAp
            cblas_daxpy(n, -alpha, Ap, 1, r, 1);
            
            // Compute new residual norm
            r_norm_old = r_norm;
            r_norm = cblas_dnrm2(n, r, 1);
            
            // Compute β = (r_{k+1}'r_{k+1})/(r_k'r_k)
            double beta = (r_norm * r_norm) / (r_norm_old * r_norm_old);
            
            // Update search direction: p = r + βp
            cblas_dscal(n, beta, p, 1);      // p = βp
            cblas_daxpy(n, 1.0, r, 1, p, 1); // p = r + βp
        }

        // Copy final solution
        cblas_dcopy(n, x, 1, solution, 1);
        *final_res = r_norm;
        *iterations = iter;
    }

    /**
     * @brief Cleanup MKL sparse matrix
     */
    void cleanup() {
        mkl_sparse_destroy(A);
    }
};

/**
 * @brief MEX function interface for MATLAB
 * 
 * Input parameters:
 * - prhs[0]: ia - Row pointers (CSR format, 1-based)
 * - prhs[1]: ja - Column indices (CSR format, 1-based)
 * - prhs[2]: a  - Nonzero values
 * - prhs[3]: b  - Right-hand side vector
 * - prhs[4]: tol - Convergence tolerance
 * - prhs[5]: maxit - Maximum iterations
 * 
 * Output parameters:
 * - plhs[0]: x - Solution vector
 * - plhs[1]: iterations - Number of iterations performed
 * - plhs[2]: residual - Final residual norm
 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    // Validate input/output argument count
    if (nrhs != 6) {
        mexErrMsgTxt("Six inputs required: ia, ja, a, b, tol, maxit");
    }
    if (nlhs != 3) {
        mexErrMsgTxt("Three outputs required: x, iterations, residual");
    }
    
    // Extract input parameters
    double *ia_pr = mxGetPr(prhs[0]);  // Row pointers
    double *ja_pr = mxGetPr(prhs[1]);  // Column indices
    double *a_pr = mxGetPr(prhs[2]);   // Nonzero values
    double *b = mxGetPr(prhs[3]);      // Right-hand side
    double tol = mxGetScalar(prhs[4]); // Tolerance
    int maxit = (int)mxGetScalar(prhs[5]); // Max iterations
    
    // Get problem dimensions
    mwSize n = mxGetM(prhs[3]);   // System size from b vector
    mwSize nnz = mxGetM(prhs[2]); // Number of nonzeros
    
    // Convert CSR indices from 1-based (MATLAB) to 0-based (C++)
    MKL_INT *ia = new MKL_INT[n + 1];
    MKL_INT *ja = new MKL_INT[nnz];
    
    for (mwSize i = 0; i < n + 1; i++) {
        ia[i] = (MKL_INT)(ia_pr[i] - 1);
    }
    for (mwSize i = 0; i < nnz; i++) {
        ja[i] = (MKL_INT)(ja_pr[i] - 1);
    }
    
    // Create output arrays
    plhs[0] = mxCreateDoubleMatrix(n, 1, mxREAL);   // Solution vector
    plhs[1] = mxCreateDoubleMatrix(1, 1, mxREAL);   // Iteration count
    plhs[2] = mxCreateDoubleMatrix(1, 1, mxREAL);   // Residual
    
    double *x = mxGetPr(plhs[0]);
    double *iterations = mxGetPr(plhs[1]);
    double *residual = mxGetPr(plhs[2]);
    
    // Create solver and solve the system
    CGSolver solver(n);
    solver.createMatrix(ia, ja, a_pr);
    
    int iter;
    solver.solve(b, x, residual, &iter, tol, maxit);
    *iterations = (double)iter;
    
    // Cleanup
    solver.cleanup();
    delete[] ia;
    delete[] ja;
}