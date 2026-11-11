#pragma once
// Alerts CSV of the Python reference (log_event): header, numeric formatting.
#include <string>

#include "adasvp/adasvp.hpp"

namespace adasvp::example {

inline constexpr const char* kAlertsCsvHeader =
    "time_s,frame,type,level_side,ttc_s,line_dist_m,lat_vel_mps,tlc_s,lead_id,lead_dist_m,red_for_s";

// Full CSV row (no newline): time .2f, frame, type, level/side, ttc/line_dist/lat_vel/tlc .2f, lead id,
// lead distance .1f, red_for .1f; absent values are empty.
std::string format_alert_row(const Alert& a);
// The row without its first four fields (what the "[alert]" log line prints after the level/side).
std::string format_alert_detail(const Alert& a);

}  // namespace adasvp::example
