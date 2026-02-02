#include <mpi.h>

#include <iostream>

#include "insitu.hpp"

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int ret = 0;
    try {
        ret = run_comp_on_mpi(argc, argv);
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        ret = 1;
    }

    MPI_Finalize();
    return ret;
}
