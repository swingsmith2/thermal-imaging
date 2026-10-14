#include "thermal/device.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <optional>
#include <sstream>

#include <fcntl.h>
#include <unistd.h>

namespace thermal {
namespace {

std::optional<double> findNumber(const std::string& text, const std::string& key) {
    const std::string pattern = "\"" + key + "\"";
    auto pos = text.find(pattern);
    if (pos == std::string::npos) {
        return std::nullopt;
    }
    pos = text.find(':', pos + pattern.size());
    if (pos == std::string::npos) {
        return std::nullopt;
    }
    pos = text.find_first_of("-0123456789", pos + 1);
    if (pos == std::string::npos) {
        return std::nullopt;
    }
    return std::strtod(text.c_str() + pos, nullptr);
}

std::string findString(const std::string& text, const std::string& key) {
    const std::string pattern = "\"" + key + "\"";
    auto pos = text.find(pattern);
    if (pos == std::string::npos) {
        return {};
    }
    pos = text.find('"', pos + pattern.size());
    if (pos == std::string::npos) {
        return {};
    }
    auto end = text.find('"', pos + 1);
    if (end == std::string::npos) {
        return {};
    }
    return text.substr(pos + 1, end - pos - 1);
}

uint16_t readBe16(const uint8_t* data) {
    return static_cast<uint16_t>((data[0] << 8) | data[1]);
}

void writeBe16(std::vector<uint8_t>* out, uint16_t value) {
    out->push_back(static_cast<uint8_t>(value >> 8));
    out->push_back(static_cast<uint8_t>(value));
}

}  // namespace

std::vector<uint16_t> hostOrder(const uint16_t* samples, int count, bool big_endian) {
    std::vector<uint16_t> out(static_cast<size_t>(count));
    if (!big_endian) {
        std::copy(samples, samples + count, out.begin());
        return out;
    }
    for (int i = 0; i < count; ++i) {
        uint16_t value = samples[i];
        out[static_cast<size_t>(i)] = static_cast<uint16_t>((value >> 8) | (value << 8));
    }
    return out;
}

double Device::signalFromTemperature(double temp_c) const {
    const double kelvin = temp_c + 273.15;
    return planck_.r1 / (planck_.r2 * (std::exp(planck_.b / kelvin) - planck_.f)) - planck_.o;
}

double Device::temperatureFromSignal(double signal) const {
    const double kelvin = planck_.b / std::log(planck_.r1 / planck_.r2 / (signal + planck_.o) + planck_.f);
    return kelvin - 273.15;
}

double Device::transmission(double distance_m) const {
    if (distance_m <= 0) {
        return 1.0;
    }
    const double vapor = std::exp(1.5587 + 0.06939 * atmospheric_c_ - 0.00027816 * atmospheric_c_ * atmospheric_c_ +
                                  0.00000068455 * atmospheric_c_ * atmospheric_c_ * atmospheric_c_);
    const double partial = humidity_ * vapor;
    const double root = std::sqrt(std::max(partial, 0.0));
    const double span = std::sqrt(distance_m);
    auto factor = [&](double alpha, double beta) { return std::exp(-span * (alpha + beta * root)); };
    return atmosphere_.x * factor(atmosphere_.alpha1, atmosphere_.beta1) +
           (1.0 - atmosphere_.x) * factor(atmosphere_.alpha2, atmosphere_.beta2);
}

double Device::objectSignal(double raw, float emissivity) const {
    const double tau_before = transmission(distance_m_ * 0.5);
    const double tau_after = transmission(distance_m_ * 0.5);
    const double window_emissivity = 1.0 - window_transmission_;
    double divisor = emissivity * lens_transmission_;
    const double reflected = (1.0 - emissivity) / divisor * signalFromTemperature(reflected_c_);
    divisor *= tau_before;
    const double air_before = (1.0 - tau_before) / divisor * signalFromTemperature(atmospheric_c_);
    divisor *= window_transmission_;
    const double window = window_emissivity / divisor * signalFromTemperature(window_c_);
    divisor *= tau_after;
    const double air_after = (1.0 - tau_after) / divisor * signalFromTemperature(atmospheric_c_);
    return (raw - shutter_bias_) / divisor - (reflected + air_before + window + air_after);
}

void Device::rebuildLut() {
    lut_.assign(65536, 0);
    for (int i = 0; i < 65536; ++i) {
        const double temp = temperatureFromSignal(static_cast<double>(i));
        const double clamped = std::clamp(temp, -40.0, 500.0);
        lut_[static_cast<size_t>(i)] = static_cast<int32_t>(std::lround(clamped * 100.0));
    }
}

bool Device::load(const std::string& path, std::string* error) {
    std::ifstream input(path);
    if (!input) {
        if (error) {
            *error = "cannot open calibration";
        }
        return false;
    }
    std::stringstream buffer;
    buffer << input.rdbuf();
    const std::string text = buffer.str();
    schema_ = static_cast<int>(findNumber(text, "schema").value_or(1));
    if (schema_ > 2) {
        if (error) {
            *error = "calibration schema is newer than this software";
        }
        return false;
    }
    serial_ = findString(text, "serial");
    if (serial_.empty()) {
        serial_ = "SC660-2412";
    }
    software_version_ = findString(text, "software_version");
    if (software_version_.empty()) {
        software_version_ = "1.0.0";
    }
    calibration_version_ = static_cast<int>(findNumber(text, "calibration_version").value_or(1));
    emissivity_ = static_cast<float>(findNumber(text, "emissivity").value_or(0.95));
    distance_m_ = static_cast<float>(findNumber(text, "distance_m").value_or(1));
    reflected_c_ = static_cast<float>(findNumber(text, "reflected_c").value_or(20));
    atmospheric_c_ = static_cast<float>(findNumber(text, "atmospheric_c").value_or(20));
    window_c_ = static_cast<float>(findNumber(text, "window_c").value_or(20));
    window_transmission_ = static_cast<float>(findNumber(text, "window_transmission").value_or(1));
    lens_transmission_ = static_cast<float>(findNumber(text, "lens_transmission").value_or(1));
    humidity_ = static_cast<float>(findNumber(text, "humidity").value_or(0.5));
    planck_.r1 = findNumber(text, "r1").value_or(planck_.r1);
    planck_.r2 = findNumber(text, "r2").value_or(planck_.r2);
    planck_.b = findNumber(text, "b").value_or(planck_.b);
    planck_.f = findNumber(text, "f").value_or(planck_.f);
    planck_.o = findNumber(text, "o").value_or(planck_.o);
    atmosphere_.alpha1 = findNumber(text, "alpha1").value_or(atmosphere_.alpha1);
    atmosphere_.alpha2 = findNumber(text, "alpha2").value_or(atmosphere_.alpha2);
    atmosphere_.beta1 = findNumber(text, "beta1").value_or(atmosphere_.beta1);
    atmosphere_.beta2 = findNumber(text, "beta2").value_or(atmosphere_.beta2);
    atmosphere_.x = findNumber(text, "x").value_or(atmosphere_.x);
    width_ = static_cast<int>(findNumber(text, "width").value_or(0));
    height_ = static_cast<int>(findNumber(text, "height").value_or(0));
    radiometric_ = text.find("\"mode\": \"nuc\"") == std::string::npos && text.find("\"mode\":\"nuc\"") == std::string::npos;
    schema_ = 2;
    rebuildLut();

    std::ifstream journal(path + ".journal");
    if (journal) {
        std::string state;
        std::getline(journal, state);
        if (state == "open") {
            ffc_open_ = true;
            ffc_recovered_ = true;
        }
    }
    return true;
}

bool Device::save(const std::string& path, std::string* error) const {
    const std::string temporary = path + ".tmp";
    std::ofstream output(temporary, std::ios::trunc);
    if (!output) {
        if (error) {
            *error = "cannot write calibration";
        }
        return false;
    }
    output << "{\n"
           << "  \"schema\": 2,\n"
           << "  \"serial\": \"" << serial_ << "\",\n"
           << "  \"software_version\": \"" << software_version_ << "\",\n"
           << "  \"calibration_version\": " << calibration_version_ << ",\n"
           << "  \"mode\": \"" << (radiometric_ ? "radiometric" : "nuc") << "\",\n"
           << "  \"emissivity\": " << emissivity_ << ",\n"
           << "  \"distance_m\": " << distance_m_ << ",\n"
           << "  \"reflected_c\": " << reflected_c_ << ",\n"
           << "  \"atmospheric_c\": " << atmospheric_c_ << ",\n"
           << "  \"window_c\": " << window_c_ << ",\n"
           << "  \"window_transmission\": " << window_transmission_ << ",\n"
           << "  \"lens_transmission\": " << lens_transmission_ << ",\n"
           << "  \"humidity\": " << humidity_ << ",\n"
           << "  \"width\": " << width_ << ",\n"
           << "  \"height\": " << height_ << ",\n"
           << "  \"planck\": {\"r1\": " << planck_.r1 << ", \"r2\": " << planck_.r2 << ", \"b\": " << planck_.b
           << ", \"f\": " << planck_.f << ", \"o\": " << planck_.o << "},\n"
           << "  \"atmosphere\": {\"alpha1\": " << atmosphere_.alpha1 << ", \"alpha2\": " << atmosphere_.alpha2
           << ", \"beta1\": " << atmosphere_.beta1 << ", \"beta2\": " << atmosphere_.beta2 << ", \"x\": " << atmosphere_.x
           << "},\n"
           << "  \"bad_pixel_count\": " << badPixelCount() << "\n"
           << "}\n";
    output.flush();
    output.close();
    int fd = ::open(temporary.c_str(), O_RDONLY);
    if (fd >= 0) {
        ::fsync(fd);
        ::close(fd);
    }
    if (std::rename(temporary.c_str(), path.c_str()) != 0) {
        if (error) {
            *error = "cannot replace calibration";
        }
        return false;
    }
    std::ofstream journal(path + ".journal", std::ios::trunc);
    journal << "idle\n";
    return true;
}

void Device::noteUpgrade(const std::string& software_version) { software_version_ = software_version; }

void Device::setRegions(std::vector<RegionSpec> regions) {
    regions_.clear();
    for (auto& spec : regions) {
        regions_.push_back(RegionState{std::move(spec), 0, 0, false, false, false});
    }
}

void Device::acknowledge(const std::string& name) {
    for (auto& region : regions_) {
        if (region.spec.name == name) {
            region.latched = false;
            region.acked_while_hot = true;
            if (region.hot_count == 0) {
                region.alarm = false;
            }
        }
    }
}

void Device::setFrozen(bool frozen) { frozen_ = frozen; }
void Device::requestShutter() { shutter_request_ = true; }
void Device::setShutterless(bool enabled, float counts_per_c) {
    shutterless_ = enabled;
    shutterless_counts_per_c_ = counts_per_c;
}
void Device::setStaleLimitUs(uint64_t limit_us) { stale_limit_us_ = limit_us; }
void Device::setConditions(float distance_m, float humidity, float reflected_c) {
    distance_m_ = distance_m;
    humidity_ = humidity;
    reflected_c_ = reflected_c;
}

int Device::badPixelCount() const {
    return static_cast<int>(std::count(bad_.begin(), bad_.end(), static_cast<uint8_t>(1)));
}

bool Device::detectBadPixels(const uint16_t* uniform, int width, int height, float sigma) {
    const int count = width * height;
    std::vector<uint16_t> sorted(uniform, uniform + count);
    std::nth_element(sorted.begin(), sorted.begin() + count / 2, sorted.end());
    const double median = sorted[static_cast<size_t>(count / 2)];
    std::vector<double> deviation(static_cast<size_t>(count));
    for (int i = 0; i < count; ++i) {
        deviation[static_cast<size_t>(i)] = std::abs(static_cast<double>(uniform[i]) - median);
    }
    std::nth_element(deviation.begin(), deviation.begin() + count / 2, deviation.end());
    const double mad = std::max(1.0, deviation[static_cast<size_t>(count / 2)] * 1.4826);
    bad_.assign(static_cast<size_t>(count), 0);
    width_ = width;
    height_ = height;
    for (int i = 0; i < count; ++i) {
        const uint16_t value = uniform[i];
        if (value == 0 || value == 65535 || std::abs(static_cast<double>(value) - median) > sigma * mad) {
            bad_[static_cast<size_t>(i)] = 1;
        }
    }
    return true;
}

bool Device::fitGainOffset(const uint16_t* cold, const uint16_t* hot, float cold_c, float hot_c, int width, int height) {
    if (cold_c >= hot_c) {
        return false;
    }
    const int count = width * height;
    const double s1 = signalFromTemperature(cold_c);
    const double s2 = signalFromTemperature(hot_c);
    gain_.assign(static_cast<size_t>(count), 1.f);
    offset_.assign(static_cast<size_t>(count), 0.f);
    width_ = width;
    height_ = height;
    if (bad_.size() != static_cast<size_t>(count)) {
        bad_.assign(static_cast<size_t>(count), 0);
    }
    for (int i = 0; i < count; ++i) {
        if (bad_[static_cast<size_t>(i)]) {
            continue;
        }
        const double response = (static_cast<double>(hot[i]) - static_cast<double>(cold[i])) / (s2 - s1);
        gain_[static_cast<size_t>(i)] = static_cast<float>(response);
        offset_[static_cast<size_t>(i)] = static_cast<float>(static_cast<double>(cold[i]) - response * s1);
    }
    radiometric_ = false;
    gain_mode_split_c_ = (cold_c + hot_c) * 0.5f;
    gain_hi_ = gain_;
    offset_hi_ = offset_;
    for (auto& value : offset_hi_) {
        value += 12.f;
    }
    rebuildLut();
    return true;
}

float Device::temperatureOfRaw(uint16_t raw, int pixel_index, float emissivity) const {
    double signal = 0;
    const float bias = shutterless_ ? shutterless_counts_per_c_ * 0.f : 0.f;
    (void)bias;
    if (radiometric_) {
        signal = objectSignal(static_cast<double>(raw), emissivity);
    } else {
        const float gain = gain_.empty() ? 1.f : gain_[static_cast<size_t>(std::max(pixel_index, 0))];
        const float offset = offset_.empty() ? 0.f : offset_[static_cast<size_t>(std::max(pixel_index, 0))];
        signal = (static_cast<double>(raw) - static_cast<double>(offset) - shutter_bias_) / static_cast<double>(gain);
        if (std::abs(emissivity - emissivity_) > 1e-3f) {
            const double reference = signalFromTemperature(reflected_c_);
            signal = (signal - (1.0 - emissivity) * reference) / emissivity;
        }
    }
    return static_cast<float>(temperatureFromSignal(signal));
}

void Device::selectGainMode(float detector_c) {
    active_gain_mode_ = detector_c >= gain_mode_split_c_ ? 1 : 0;
}

int Device::acquireSlot() {
    for (int i = 0; i < kPoolSlots; ++i) {
        if (!slot_used_[static_cast<size_t>(i)]) {
            slot_used_[static_cast<size_t>(i)] = true;
            return i;
        }
    }
    return -1;
}

void Device::releaseSlot(int slot) {
    if (slot >= 0 && slot < kPoolSlots) {
        slot_used_[static_cast<size_t>(slot)] = false;
    }
}

void Device::publish(const FrameView& frame, const std::vector<float>& celsius, std::vector<RegionStatus>* regions,
                      Diagnostic* diagnostic, bool calibrating) {
    if (!frozen_ && !calibrating) {
        published_ = celsius;
        has_published_ = !published_.empty();
    }
    measurement_frame_ = static_cast<int>(frame.frame_id);
    display_queue_.push_back(measurement_frame_);
    while (static_cast<int>(display_queue_.size()) > kDisplayLimit) {
        display_queue_.erase(display_queue_.begin());
        ++display_dropped_;
    }
    if (regions) {
        regions->clear();
    }
    const std::vector<float>& shown = has_published_ ? published_ : celsius;
    for (auto& region : regions_) {
        float min_c = 1e9f;
        float max_c = -1e9f;
        double sum = 0;
        int count = 0;
        for (int y = region.spec.y; y < region.spec.y + region.spec.h; ++y) {
            for (int x = region.spec.x; x < region.spec.x + region.spec.w; ++x) {
                if (x < 0 || y < 0 || x >= frame.width || y >= frame.height || shown.empty()) {
                    continue;
                }
                const float value = shown[static_cast<size_t>(y * frame.width + x)];
                min_c = std::min(min_c, value);
                max_c = std::max(max_c, value);
                sum += value;
                ++count;
            }
        }
        if (count == 0) {
            continue;
        }
        const float mean = static_cast<float>(sum / count);
        const bool hot = max_c >= region.spec.alarm_c;
        const bool cool = max_c < region.spec.alarm_c - region.spec.hysteresis_c;
        if (hot) {
            region.hot_count++;
            region.cold_count = 0;
        } else if (cool) {
            region.cold_count++;
            region.hot_count = 0;
        } else {
            region.hot_count = 0;
            region.cold_count = 0;
        }
        if (region.hot_count >= region.spec.debounce_frames) {
            region.alarm = true;
            region.latched = true;
        }
        if (region.cold_count >= region.spec.debounce_frames && !region.latched) {
            region.alarm = false;
        }
        if (!region.latched && region.acked_while_hot && cool && region.cold_count >= region.spec.debounce_frames) {
            region.alarm = false;
            region.acked_while_hot = false;
        }
        if (regions) {
            regions->push_back(RegionStatus{region.spec.name, min_c, max_c, mean, region.alarm, region.latched});
        }
    }
    if (diagnostic) {
        *diagnostic = Diagnostic{};
        diagnostic->serial = serial_;
        diagnostic->schema = schema_;
        diagnostic->calibration_version = calibration_version_;
        diagnostic->shutter_count = shutter_count_;
        diagnostic->last_frame_id = frame.frame_id;
        diagnostic->last_timestamp_us = frame.timestamp_us;
        diagnostic->dropped_frames = dropped_frames_;
        diagnostic->display_dropped = display_dropped_;
        diagnostic->calibrating = calibrating;
        diagnostic->frozen = frozen_;
        diagnostic->stale = stale_;
        diagnostic->ffc_recovered = ffc_recovered_;
        diagnostic->shutterless = shutterless_;
        diagnostic->published_c = has_published_ ? published_[published_.size() / 2] : 0;
        diagnostic->measurement_frame = measurement_frame_;
    }
}

bool Device::push(const FrameView& frame, std::vector<RegionStatus>* regions, Diagnostic* diagnostic) {
    if (frame.samples == nullptr || frame.width <= 0 || frame.height <= 0) {
        return false;
    }
    const auto t0 = std::chrono::steady_clock::now();
    auto samples = hostOrder(frame.samples, frame.width * frame.height, frame.big_endian);
    const auto t1 = std::chrono::steady_clock::now();
    if (have_frame_ && frame.frame_id > last_frame_id_ + 1) {
        dropped_frames_ += static_cast<int>(frame.frame_id - last_frame_id_ - 1);
    }
    stale_ = have_frame_ && frame.timestamp_us > last_timestamp_us_ &&
             frame.timestamp_us - last_timestamp_us_ > stale_limit_us_;
    if (bad_.size() != samples.size()) {
        bad_.assign(samples.size(), 0);
    }
    for (size_t i = 0; i < samples.size(); ++i) {
        if (!bad_[i]) {
            continue;
        }
        const int x = static_cast<int>(i) % frame.width;
        const int y = static_cast<int>(i) / frame.width;
        double sum = 0;
        int count = 0;
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                if (dx == 0 && dy == 0) {
                    continue;
                }
                const int xx = x + dx;
                const int yy = y + dy;
                if (xx < 0 || yy < 0 || xx >= frame.width || yy >= frame.height) {
                    continue;
                }
                const size_t neighbor = static_cast<size_t>(yy * frame.width + xx);
                if (!bad_[neighbor]) {
                    sum += samples[neighbor];
                    ++count;
                }
            }
        }
        if (count > 0) {
            samples[i] = static_cast<uint16_t>(sum / count);
        }
    }
    const auto t2 = std::chrono::steady_clock::now();
    selectGainMode(frame.detector_c);
    const bool shutter = shutter_request_ || frame.kind == PixelKind::Shutter ||
                         (!shutterless_ && std::abs(frame.detector_c - last_ffc_detector_c_) >= shutter_delta_c_ && have_frame_);
    std::vector<float> celsius(samples.size(), published_.empty() ? 0.f : published_[0]);
    bool calibrating = false;
    if (shutter && has_published_) {
        calibrating = true;
        ffc_open_ = true;
        const double expected = signalFromTemperature(frame.detector_c);
        if (radiometric_) {
            double sum = 0;
            for (uint16_t value : samples) {
                sum += value;
            }
            shutter_bias_ = sum / static_cast<double>(samples.size()) - expected;
        } else {
            const auto& gain = active_gain_mode_ == 1 && !gain_hi_.empty() ? gain_hi_ : gain_;
            auto& offset = active_gain_mode_ == 1 && !offset_hi_.empty() ? offset_hi_ : offset_;
            for (size_t i = 0; i < samples.size(); ++i) {
                if (!bad_[i] && i < offset.size()) {
                    offset[i] = static_cast<float>(samples[i] - gain[i] * expected);
                }
            }
        }
        ++calibration_version_;
        ++shutter_count_;
        last_ffc_detector_c_ = frame.detector_c;
        shutter_request_ = false;
        ffc_open_ = false;
        celsius = published_;
    } else {
        const float drift = shutterless_ ? shutterless_counts_per_c_ * (frame.detector_c - cal_detector_c_) : 0.f;
        const auto& gain = active_gain_mode_ == 1 && !gain_hi_.empty() ? gain_hi_ : gain_;
        const auto& offset = active_gain_mode_ == 1 && !offset_hi_.empty() ? offset_hi_ : offset_;
        for (size_t i = 0; i < samples.size(); ++i) {
            const double corrected = static_cast<double>(samples[i]) - drift;
            double signal = 0;
            if (radiometric_) {
                signal = objectSignal(corrected, emissivity_);
            } else {
                const double g = i < gain.size() ? gain[i] : 1.0;
                const double o = i < offset.size() ? offset[i] : 0.0;
                signal = (corrected - o - shutter_bias_) / g;
            }
            celsius[i] = static_cast<float>(temperatureFromSignal(signal));
        }
        if (!have_frame_) {
            last_ffc_detector_c_ = frame.detector_c;
            cal_detector_c_ = frame.detector_c;
        }
    }
    const auto t3 = std::chrono::steady_clock::now();
    publish(frame, celsius, regions, diagnostic, calibrating);
    const auto t4 = std::chrono::steady_clock::now();
    if (diagnostic) {
        diagnostic->stage_us[0] = std::chrono::duration<double, std::micro>(t1 - t0).count();
        diagnostic->stage_us[1] = std::chrono::duration<double, std::micro>(t2 - t1).count();
        diagnostic->stage_us[2] = std::chrono::duration<double, std::micro>(t3 - t2).count();
        diagnostic->stage_us[3] = std::chrono::duration<double, std::micro>(t4 - t3).count();
    }
    last_frame_id_ = frame.frame_id;
    last_timestamp_us_ = frame.timestamp_us;
    have_frame_ = true;
    return true;
}

double Device::lutMicroseconds(int frames, int width, int height) {
    rebuildLut();
    std::vector<uint16_t> raw(static_cast<size_t>(width * height), 18090);
    std::vector<int32_t> out(raw.size());
    const auto start = std::chrono::steady_clock::now();
    for (int frame = 0; frame < frames; ++frame) {
        for (size_t i = 0; i < raw.size(); ++i) {
            out[i] = lut_[raw[i]];
        }
    }
    const auto stop = std::chrono::steady_clock::now();
    if (out[0] == 0) {
        out[0] = 1;
    }
    return std::chrono::duration<double, std::micro>(stop - start).count() / frames;
}

uint16_t Device::alarmRegister() const {
    for (const auto& region : regions_) {
        if (region.alarm) {
            return 1;
        }
    }
    return 0;
}

std::vector<uint8_t> Device::modbusTcp(const std::vector<uint8_t>& request) const {
    if (request.size() < 12) {
        return {};
    }
    const uint16_t quantity = readBe16(request.data() + 10);
    std::vector<uint8_t> response = {request[0], request[1], 0, 0, 0, 0, request[6], 0x03, static_cast<uint8_t>(quantity * 2)};
    const uint16_t registers[4] = {alarmRegister(), static_cast<uint16_t>(calibration_version_),
                                   static_cast<uint16_t>(shutter_count_), static_cast<uint16_t>(stale_ ? 1 : 0)};
    const uint16_t start = readBe16(request.data() + 8);
    for (uint16_t i = 0; i < quantity; ++i) {
        const uint16_t value = (start + i) < 4 ? registers[start + i] : 0;
        writeBe16(&response, value);
    }
    const uint16_t length = static_cast<uint16_t>(response.size() - 6);
    response[4] = static_cast<uint8_t>(length >> 8);
    response[5] = static_cast<uint8_t>(length);
    return response;
}

void Device::writeDryContact(const std::string& path) const {
    std::ofstream output(path, std::ios::trunc);
    output << (alarmRegister() ? "1\n" : "0\n");
}

void Device::appendAlarmEvent(const std::string& path, uint64_t timestamp_us) {
    std::ofstream output(path, std::ios::app);
    output << "{\"timestamp_us\": " << timestamp_us << ", \"alarm\": " << alarmRegister()
           << ", \"calibration_version\": " << calibration_version_ << ", \"serial\": \"" << serial_ << "\"}\n";
}

std::vector<uint8_t> Device::mqttPublish(const std::string& topic, const std::string& payload) {
    std::vector<uint8_t> body;
    writeBe16(&body, static_cast<uint16_t>(topic.size()));
    body.insert(body.end(), topic.begin(), topic.end());
    body.insert(body.end(), payload.begin(), payload.end());
    std::vector<uint8_t> packet = {0x30};
    size_t length = body.size();
    do {
        uint8_t encoded = static_cast<uint8_t>(length % 128);
        length /= 128;
        if (length > 0) {
            encoded |= 0x80;
        }
        packet.push_back(encoded);
    } while (length > 0);
    packet.insert(packet.end(), body.begin(), body.end());
    return packet;
}

bool Device::mqttPublishMatches(const std::vector<uint8_t>& packet, const std::string& topic, const std::string& payload) {
    if (packet.size() < 4 || packet[0] != 0x30) {
        return false;
    }
    size_t index = 1;
    size_t remaining = 0;
    size_t multiplier = 1;
    for (int i = 0; i < 4 && index < packet.size(); ++i) {
        const uint8_t encoded = packet[index++];
        remaining += (encoded & 0x7f) * multiplier;
        multiplier *= 128;
        if ((encoded & 0x80) == 0) {
            break;
        }
    }
    if (index + 2 > packet.size() || remaining != packet.size() - index) {
        return false;
    }
    const uint16_t topic_len = readBe16(packet.data() + index);
    index += 2;
    if (index + topic_len > packet.size()) {
        return false;
    }
    const std::string got_topic(reinterpret_cast<const char*>(packet.data() + index), topic_len);
    index += topic_len;
    const std::string got_payload(reinterpret_cast<const char*>(packet.data() + index), packet.size() - index);
    return got_topic == topic && got_payload == payload;
}

bool Device::recordRaw(const std::string& path, const FrameView& frame) const {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        return false;
    }
    const char magic[4] = {'T', 'H', 'R', 'M'};
    output.write(magic, 4);
    output.write(reinterpret_cast<const char*>(&frame.frame_id), sizeof(frame.frame_id));
    output.write(reinterpret_cast<const char*>(&frame.timestamp_us), sizeof(frame.timestamp_us));
    output.write(reinterpret_cast<const char*>(&frame.detector_c), sizeof(frame.detector_c));
    const int32_t width = frame.width;
    const int32_t height = frame.height;
    const int32_t kind = frame.kind == PixelKind::Shutter ? 1 : 0;
    output.write(reinterpret_cast<const char*>(&width), sizeof(width));
    output.write(reinterpret_cast<const char*>(&height), sizeof(height));
    output.write(reinterpret_cast<const char*>(&kind), sizeof(kind));
    output.write(reinterpret_cast<const char*>(frame.samples), sizeof(uint16_t) * static_cast<size_t>(width * height));
    return static_cast<bool>(output);
}

bool Device::playbackRaw(const std::string& path, std::vector<uint16_t>* samples, FrameView* frame) const {
    std::ifstream input(path, std::ios::binary);
    if (!input || samples == nullptr || frame == nullptr) {
        return false;
    }
    char magic[4] = {};
    input.read(magic, 4);
    if (std::string(magic, 4) != "THRM") {
        return false;
    }
    input.read(reinterpret_cast<char*>(&frame->frame_id), sizeof(frame->frame_id));
    input.read(reinterpret_cast<char*>(&frame->timestamp_us), sizeof(frame->timestamp_us));
    input.read(reinterpret_cast<char*>(&frame->detector_c), sizeof(frame->detector_c));
    int32_t width = 0;
    int32_t height = 0;
    int32_t kind = 0;
    input.read(reinterpret_cast<char*>(&width), sizeof(width));
    input.read(reinterpret_cast<char*>(&height), sizeof(height));
    input.read(reinterpret_cast<char*>(&kind), sizeof(kind));
    samples->resize(static_cast<size_t>(width * height));
    input.read(reinterpret_cast<char*>(samples->data()), sizeof(uint16_t) * samples->size());
    frame->width = width;
    frame->height = height;
    frame->kind = kind == 1 ? PixelKind::Shutter : PixelKind::Scene;
    frame->samples = samples->data();
    return static_cast<bool>(input);
}

void Device::beginUnsafeFfc(const std::string& calibration_path) {
    ffc_open_ = true;
    std::ofstream journal(calibration_path + ".journal", std::ios::trunc);
    journal << "open\n";
    std::ofstream partial(calibration_path + ".partial", std::ios::trunc);
    partial << "partial-offset\n";
}

}  // namespace thermal
