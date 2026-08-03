#pragma once
#include <cstdio>
namespace hobot::hlog {
enum class LogLevel { log_trace, log_debug, log_info, log_warn, log_err };
}
#define HLOG_SET_LOGLEVEL(level) ((void)0)
#define HFLOGM_T(tag, fmt, ...) std::fprintf(stderr, "[T][%s] %s\n", tag, fmt)
#define HFLOGM_D(tag, fmt, ...) std::fprintf(stderr, "[D][%s] %s\n", tag, fmt)
#define HFLOGM_I(tag, fmt, ...) std::fprintf(stderr, "[I][%s] %s\n", tag, fmt)
#define HFLOGM_W(tag, fmt, ...) std::fprintf(stderr, "[W][%s] %s\n", tag, fmt)
#define HFLOGM_E(tag, fmt, ...) std::fprintf(stderr, "[E][%s] %s\n", tag, fmt)
