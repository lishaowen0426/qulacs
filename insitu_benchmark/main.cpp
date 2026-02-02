#include <mpi.h>

#include <chrono>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "insitu.hpp"
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/spdlog.h"

namespace {
std::string make_log_filename(const char* argv0) {
    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    std::ostringstream oss;
    oss << "insitu_benchmark_";
    oss << std::put_time(&tm, "%Y%m%d_%H%M%S") << ".log";
    std::filesystem::path dir = std::filesystem::path(argv0).parent_path();
    if (dir.empty()) {
        dir = std::filesystem::current_path();
    }
    return (dir / oss.str()).string();
}
}  // namespace

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    auto logger =
        spdlog::basic_logger_mt("insitu_file", make_log_filename(argv[0]));
    spdlog::set_default_logger(logger);
    spdlog::set_pattern("%v");

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
