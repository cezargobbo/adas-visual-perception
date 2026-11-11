#include "alerts_csv.hpp"

#include <algorithm>
#include <cstdio>
#include <utility>
#include <vector>
#include <optional>

namespace adasvp::example {
namespace {

std::string fmt_pct(double v) {
  char b[64];
  std::snprintf(b, sizeof b, "%.0f", v);
  return b;
}

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

std::string format_alerts_summary(bool ldw_on, const AdasStats& st) {
  auto pct = [](long long v, long long total) { return fmt_pct(100.0 * static_cast<double>(v) / static_cast<double>(total)); };
  std::string msg = "[info] alerts: FCW " + std::to_string(st.fcw_events) + ", LDW " + std::to_string(st.ldw_events) +
                    ", RLR " + std::to_string(st.rlr_events);
  if (!ldw_on) return msg;
  long long total = 0;
  for (const auto& kv : st.ldw_frames) total += kv.second;
  total = std::max(total, 1LL);
  auto it = st.ldw_frames.find("armed");
  msg += " | LDW armed in " + pct(it == st.ldw_frames.end() ? 0 : it->second, total) +
         "% of frames, crossings suppressed " + std::to_string(st.ldw_crossings_suppressed) + "; disarm reasons: ";
  std::vector<std::pair<std::string, long long>> reasons;
  auto add = [&](const std::string& k) {
    auto f = st.ldw_frames.find(k);
    if (k != "armed" && f != st.ldw_frames.end()) reasons.emplace_back(k, f->second);
  };
  for (const auto& k : st.ldw_frames_order) add(k);
  for (const auto& kv : st.ldw_frames)  // keys missing from the order list (never with the Pipeline): alphabetical
    if (std::find(st.ldw_frames_order.begin(), st.ldw_frames_order.end(), kv.first) == st.ldw_frames_order.end()) add(kv.first);
  std::stable_sort(reasons.begin(), reasons.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
  for (size_t k = 0; k < reasons.size(); ++k)
    msg += (k ? ", " : "") + reasons[k].first + " " + pct(reasons[k].second, total) + "%";
  return msg;
}

}  // namespace adasvp::example
