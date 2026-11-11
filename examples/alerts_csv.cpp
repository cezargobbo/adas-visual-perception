#include "alerts_csv.hpp"

#include <cstdio>
#include <optional>

namespace adasvp::example {
namespace {

std::string fixed(const std::optional<double>& x, int digits) {
  if (!x) return "";
  char b[64];
  std::snprintf(b, sizeof b, "%.*f", digits, *x);
  return b;
}

}  // namespace

std::string format_alert_detail(const Alert& a) {
  std::string s = fixed(a.ttc_s, 2) + "," + fixed(a.line_dist_m, 2) + "," + fixed(a.lat_vel_mps, 2) + "," +
                  fixed(a.tlc_s, 2) + ",";
  if (a.lead_id >= 0) s += std::to_string(a.lead_id);
  s += ",";
  s += a.lead_id >= 0 ? fixed(a.lead_dist_m, 1) : "";
  s += "," + fixed(a.red_for_s, 1);
  return s;
}

std::string format_alert_row(const Alert& a) {
  return fixed(a.t, 2) + "," + std::to_string(a.frame) + "," + a.type + "," + a.level_side + "," + format_alert_detail(a);
}

}  // namespace adasvp::example
