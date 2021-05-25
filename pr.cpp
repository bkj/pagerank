


#pragma GCC diagnostic ignored "-Wunused-result"

#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <string.h>
#include <omp.h>

using namespace std::chrono;
// #define VERBOSE

// --
// Global defs

typedef int Int;
typedef float Real;

// params
Real alpha   = 0.85;
Real tol     = 1e-6;
Int max_iter = 1000;

// graph
Int n_rows, n_cols, n_nnz;
Int* indptr;
Int* indices;
Real* data;

Int n_nodes;
Int n_edges;

// output
Real* x;

// --
// IO

void load_data(std::string inpath) {
    FILE *ptr;
    ptr = fopen(inpath.c_str(), "rb");

    fread(&n_rows,   sizeof(Int), 1, ptr);
    fread(&n_cols,   sizeof(Int), 1, ptr);
    fread(&n_nnz,    sizeof(Int), 1, ptr);

    indptr   = (Int*)  malloc(sizeof(Int)  * (n_rows + 1)  );
    indices  = (Int*)  malloc(sizeof(Int)  * n_nnz         );
    data = (Real*) malloc(sizeof(Real) * n_nnz         );

    fread(indptr,  sizeof(Int),   n_rows + 1 , ptr);  // send directy to the memory since thats what the thing is.
    fread(indices, sizeof(Int),   n_nnz      , ptr);
    fread(data, sizeof(Real),     n_nnz      , ptr);

#ifdef VERBOSE
        printf("----------------------------\n");
        printf("n_rows   = %d\n", n_rows);
        printf("n_cols   = %d\n", n_cols);
        printf("n_nnz    = %d\n", n_nnz);
        printf("----------------------------\n");
#endif
}

// --
// Run

void run_app(Real* x) {
    Real* p     = (Real*)malloc(n_nodes * sizeof(Real));
    Real* xlast = (Real*)malloc(n_nodes * sizeof(Real));
    
    Real init = 1.0 / n_nodes;
    for(Int i = 0; i < n_nodes; i++) p[i]     = init;
    for(Int i = 0; i < n_nodes; i++) x[i]     = init;
    for(Int i = 0; i < n_nodes; i++) xlast[i] = init;
    
    auto t1 = high_resolution_clock::now();
    for(Int it = 0; it < max_iter; it++) {
        printf("it=%d\n", it);
        
        #pragma omp parallel for
        for(Int dst = 0; dst < n_nodes; dst++) {
            Real acc = (1 - alpha) * p[dst];
            for(Int offset = indptr[dst]; offset < indptr[dst + 1]; offset++) {
                Int src  = indices[offset];
                Real val = data[offset];
                acc += alpha * xlast[src] * val;
            }
            
            x[dst] = acc;
        }
        
        bool done = true;
        for(Int i = 0; i < n_nodes; i++) {
            Real err = abs(x[i] - xlast[i]);
            if(err > tol) {
                done = false;
                break;
            }
        }
        
        if(done) break;
        
        Real* tmp_ptr = xlast;
        xlast = x;
        x     = tmp_ptr;
    }
    auto elapsed = high_resolution_clock::now() - t1;
    long long ms = duration_cast<microseconds>(elapsed).count();
    std::cout << "elapsed=" << ms << std::endl;
}

int main(int n_args, char** argument_array) {
    // ---------------- INPUT ----------------

    unsigned int n_runs = 1;
    if(n_args > 2) {
        n_runs = (unsigned int)atoi(argument_array[2]);
    }

    load_data(argument_array[1]);

    n_nodes = n_rows;
    n_edges = n_nnz;
    
    // ---------------- RUN ----------------
    auto t1 = high_resolution_clock::now();
    
    x = (Real*)malloc(n_nodes * sizeof(Real));
    for(unsigned int run = 0; run < n_runs; run++)
        run_app(x);
    
    auto elapsed = high_resolution_clock::now() - t1;
    long long ms = duration_cast<microseconds>(elapsed).count();
    
    ms /= n_runs;
    
    for(int i = 0; i < 40; i++) {
        printf("%f ", x[i]);
    }
    printf("\n");
    
    std::cout << "elapsed=" << ms << std::endl;
    
    return 0;
}
