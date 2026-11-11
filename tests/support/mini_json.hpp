#pragma once
// Tiny helpers to read flat values from the fixture .json files (first occurrence of a key).
#include <fstream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace adasvp::test {

struct MiniJson {
  std::string text;
  explicit MiniJson(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::stringstream ss;
    ss << f.rdbuf();
    text = ss.str();
  }
  double number(const std::string& key) const {
    std::smatch m;
    std::regex re("\"" + key + "\"\\s*:\\s*(-?[0-9.eE+-]+)");
    if (!std::regex_search(text, m, re)) throw std::runtime_error("missing key " + key);
    return std::stod(m[1]);
  }
  std::string string(const std::string& key) const {
    std::smatch m;
    std::regex re("\"" + key + "\"\\s*:\\s*\"([^\"]*)\"");
    if (!std::regex_search(text, m, re)) throw std::runtime_error("missing key " + key);
    return m[1];
  }
  std::vector<double> array(const std::string& key) const {
    std::smatch m;
    std::regex re("\"" + key + "\"\\s*:\\s*\\[([^\\]]*)\\]");
    if (!std::regex_search(text, m, re)) throw std::runtime_error("missing key " + key);
    std::vector<double> out;
    std::stringstream ss(m[1]);
    std::string tok;
    while (std::getline(ss, tok, ',')) out.push_back(std::stod(tok));
    return out;
  }
};

}  // namespace adasvp::test
