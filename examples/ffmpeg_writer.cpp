#include "ffmpeg_writer.hpp"

#include <fcntl.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace adasvp::example {
namespace {

// Shortest representation that reads back as the same double (like Python's f"{x}").
std::string num(double v) {
  char b[64];
  auto r = std::to_chars(b, b + sizeof b, v);
  return std::string(b, r.ptr);
}

// Runs ffmpeg with `args`, stdin from `stdin_fd` (or /dev/null) and stdout to `stdout_fd` (or /dev/null).
// Returns the pid; the caller owns the fds. Throws if ffmpeg cannot be executed (reported by the child through
// a close-on-exec pipe, so a missing binary is an error here and not a silent exit code 127).
pid_t spawn(const std::vector<std::string>& args, int stdin_fd, int stdout_fd) {
  std::vector<char*> argv;
  std::vector<std::string> store = args;
  store.insert(store.begin(), "ffmpeg");
  for (auto& s : store) argv.push_back(s.data());
  argv.push_back(nullptr);
  int ep[2];
  if (pipe2(ep, O_CLOEXEC) != 0) throw std::runtime_error("pipe failed");
  const pid_t pid = fork();
  if (pid < 0) {
    ::close(ep[0]);
    ::close(ep[1]);
    throw std::runtime_error(std::string("fork failed: ") + std::strerror(errno));
  }
  if (pid == 0) {
    const int devnull = open("/dev/null", O_RDWR);
    dup2(stdin_fd >= 0 ? stdin_fd : devnull, 0);
    dup2(stdout_fd >= 0 ? stdout_fd : devnull, 1);
    if (stdin_fd > 2) ::close(stdin_fd);
    if (stdout_fd > 2) ::close(stdout_fd);
    if (devnull > 2) ::close(devnull);
    execvp("ffmpeg", argv.data());
    const int err = errno;
    const ssize_t ignored = ::write(ep[1], &err, sizeof err);
    (void)ignored;
    _exit(127);
  }
  ::close(ep[1]);
  int err = 0;
  ssize_t n;
  do {
    n = read(ep[0], &err, sizeof err);
  } while (n < 0 && errno == EINTR);
  ::close(ep[0]);
  if (n == static_cast<ssize_t>(sizeof err)) {
    int st = 0;
    waitpid(pid, &st, 0);
    throw std::runtime_error(std::string("cannot run ffmpeg: ") + std::strerror(err));
  }
  return pid;
}

}  // namespace

std::vector<std::string> ffmpeg_args(const WriterSpec& s) {
  std::vector<std::string> a = {"-y",      "-loglevel", "error", "-f",  "rawvideo", "-pix_fmt", "bgr24",
                                "-s",      std::to_string(s.width) + "x" + std::to_string(s.height),
                                "-r",      num(s.fps), "-i", "-"};
  if (!s.audio_from.empty()) {
    a.push_back("-ss");
    a.push_back(num(s.start_s));
    a.push_back("-t");
    a.push_back(num(s.duration_s));
    a.push_back("-i");
    a.push_back(s.audio_from);
    // -max_muxing_queue_size: the system ffmpeg (3.4) otherwise overflows its muxing queue when the audio
    // runs ahead of the piped video
    for (const char* x : {"-map", "0:v", "-map", "1:a?", "-c:a", "aac", "-shortest", "-max_muxing_queue_size", "9999"})
      a.push_back(x);
  }
  if (s.encoder == "nvenc")
    for (const char* x : {"-c:v", "h264_nvenc", "-preset", "medium", "-rc", "vbr", "-cq", "21", "-b:v", "0"}) a.push_back(x);
  else
    for (const char* x : {"-c:v", "libx264", "-preset", "medium", "-crf", "20"}) a.push_back(x);
  for (const char* x : {"-pix_fmt", "yuv420p"}) a.push_back(x);
  a.push_back(s.path);
  return a;
}

bool ffmpeg_available() {
  const char* path = std::getenv("PATH");
  std::string p = path ? path : "";
  size_t pos = 0;
  while (pos <= p.size()) {
    size_t e = p.find(':', pos);
    if (e == std::string::npos) e = p.size();
    const std::string dir = p.substr(pos, e - pos);
    const std::string cand = (dir.empty() ? "." : dir) + "/ffmpeg";
    if (access(cand.c_str(), X_OK) == 0) return true;
    pos = e + 1;
  }
  return false;
}

std::string resolve_encoder(const std::string& requested) {
  if (requested != "auto") return requested;
  int fds[2];
  if (pipe(fds) != 0) return "x264";
  pid_t pid;
  try {
    pid = spawn({"-hide_banner", "-encoders"}, -1, fds[1]);
  } catch (const std::exception&) {
    close(fds[0]);
    close(fds[1]);
    return "x264";
  }
  close(fds[1]);
  std::string out;
  char buf[4096];
  ssize_t n;
  while ((n = read(fds[0], buf, sizeof buf)) > 0) out.append(buf, static_cast<size_t>(n));
  close(fds[0]);
  int st = 0;
  waitpid(pid, &st, 0);
  return out.find("h264_nvenc") != std::string::npos ? "nvenc" : "x264";
}

FfmpegWriter::FfmpegWriter(const WriterSpec& s)
    : frame_bytes_(static_cast<size_t>(s.width) * static_cast<size_t>(s.height) * 3), path_(s.path) {
  signal(SIGPIPE, SIG_IGN);  // a dying ffmpeg must surface as a write error, not kill the process
  int fds[2];
  if (pipe(fds) != 0) throw std::runtime_error("pipe failed");
  fcntl(fds[1], F_SETFD, FD_CLOEXEC);
  try {
    pid_ = spawn(ffmpeg_args(s), fds[0], -1);
  } catch (...) {
    ::close(fds[0]);
    ::close(fds[1]);
    throw;
  }
  ::close(fds[0]);
  fd_ = fds[1];
}

FfmpegWriter::~FfmpegWriter() {
  try {
    close();
  } catch (...) {
  }
}

void FfmpegWriter::write(const cv::Mat& bgr) {
  if (fd_ < 0) throw std::runtime_error("ffmpeg writer is closed: " + path_);
  if (!bgr.isContinuous() || bgr.total() * bgr.elemSize() != frame_bytes_)
    throw std::runtime_error("unexpected frame size for " + path_);
  const char* p = reinterpret_cast<const char*>(bgr.data);
  size_t left = frame_bytes_;
  while (left > 0) {
    const ssize_t n = ::write(fd_, p, left);
    if (n < 0) {
      if (errno == EINTR) continue;
      throw std::runtime_error("ffmpeg stopped while writing " + path_);
    }
    p += n;
    left -= static_cast<size_t>(n);
  }
}

int FfmpegWriter::close() {
  if (fd_ >= 0) {
    ::close(fd_);
    fd_ = -1;
  }
  int code = -1;
  if (pid_ > 0) {
    int st = 0;
    while (waitpid(pid_, &st, 0) < 0 && errno == EINTR) {
    }
    pid_ = -1;
    if (WIFEXITED(st)) code = WEXITSTATUS(st);
    if (code != 0)
      std::fprintf(stderr, "[warn] ffmpeg exited with %s while writing %s\n",
                   code < 0 ? "an abnormal status" : ("status " + std::to_string(code)).c_str(), path_.c_str());
  }
  return code;
}

}  // namespace adasvp::example
