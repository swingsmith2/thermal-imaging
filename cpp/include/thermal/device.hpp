#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace thermal {

enum class PixelKind { Scene, Shutter };
enum class Quantity { RawCounts, Celsius };

struct FrameView {
    uint64_t frame_id = 0;
    uint64_t timestamp_us = 0;
    float detector_c = 25.f;
    int gain_mode = -1;
    PixelKind kind = PixelKind::Scene;
    bool big_endian = false;
    const uint16_t* samples = nullptr;
    int width = 0;
    int height = 0;
};

struct RegionSpec {
    std::string name;
    int x = 0;
    int y = 0;
    int w = 0;
    int h = 0;
    float emissivity = 0.95f;
    float alarm_c = 90.f;
    float hysteresis_c = 2.f;
    int debounce_frames = 2;
};

struct RegionStatus {
    std::string name;
    float min_c = 0;
    float max_c = 0;
    float mean_c = 0;
    bool alarm = false;
    bool latched = false;
};

struct Diagnostic {
    std::string serial;
    int schema = 0;
    int calibration_version = 0;
    int shutter_count = 0;
    uint64_t last_frame_id = 0;
    uint64_t last_timestamp_us = 0;
    int dropped_frames = 0;
    int display_dropped = 0;
    bool calibrating = false;
    bool frozen = false;
    bool stale = false;
    bool ffc_recovered = false;
    bool shutterless = false;
    std::array<double, 4> stage_us{};
    float published_c = 0;
    int measurement_frame = 0;
};

struct BlackbodyPoint {
    float setpoint_c = 0;
    float measured_c = 0;
    float error_c = 0;
};

class Device {
public:
    static constexpr int kPoolSlots = 4;
    static constexpr int kDisplayLimit = 2;

    bool load(const std::string& path, std::string* error);
    bool save(const std::string& path, std::string* error) const;
    void noteUpgrade(const std::string& software_version);

    void setRegions(std::vector<RegionSpec> regions);
    void acknowledge(const std::string& name);
    void setFrozen(bool frozen);
    void requestShutter();
    void setShutterless(bool enabled, float counts_per_c);
    void setStaleLimitUs(uint64_t limit_us);
    void setConditions(float distance_m, float humidity, float reflected_c);

    bool push(const FrameView& frame, std::vector<RegionStatus>* regions, Diagnostic* diagnostic);

    bool detectBadPixels(const uint16_t* uniform, int width, int height, float sigma);
    bool fitGainOffset(const uint16_t* cold, const uint16_t* hot, float cold_c, float hot_c, int width, int height);
    float temperatureOfRaw(uint16_t raw, int pixel_index, float emissivity) const;

    int acquireSlot();
    void releaseSlot(int slot);

    uint16_t alarmRegister() const;
    std::vector<uint8_t> modbusTcp(const std::vector<uint8_t>& request) const;
    void writeDryContact(const std::string& path) const;
    void appendAlarmEvent(const std::string& path, uint64_t timestamp_us);
    static std::vector<uint8_t> mqttPublish(const std::string& topic, const std::string& payload);
    static bool mqttPublishMatches(const std::vector<uint8_t>& packet, const std::string& topic, const std::string& payload);

    bool recordRaw(const std::string& path, const FrameView& frame) const;
    bool playbackRaw(const std::string& path, std::vector<uint16_t>* samples, FrameView* frame) const;

    int calibrationVersion() const { return calibration_version_; }
    const std::string& serial() const { return serial_; }
    const std::string& softwareVersion() const { return software_version_; }
    int badPixelCount() const;
    bool ffcJournalOpen() const { return ffc_open_; }
    void beginUnsafeFfc(const std::string& calibration_path);
    int schema() const { return schema_; }
    double lutMicroseconds(int frames, int width, int height);
    const std::vector<float>& published() const { return published_; }

private:
    struct Planck {
        double r1 = 21106.77;
        double r2 = 0.012545258;
        double b = 1501;
        double f = 1;
        double o = -7340;
    };
    struct Atmosphere {
        double alpha1 = 0.006569;
        double alpha2 = 0.01262;
        double beta1 = -0.002276;
        double beta2 = -0.00667;
        double x = 1.9;
    };
    struct RegionState {
        RegionSpec spec;
        int hot_count = 0;
        int cold_count = 0;
        bool alarm = false;
        bool latched = false;
        bool acked_while_hot = false;
    };

    double signalFromTemperature(double temp_c) const;
    double temperatureFromSignal(double signal) const;
    double transmission(double distance_m) const;
    double objectSignal(double raw, float emissivity) const;
    void rebuildLut();
    void publish(const FrameView& frame, const std::vector<float>& celsius, std::vector<RegionStatus>* regions, Diagnostic* diagnostic, bool calibrating);
    void selectGainMode(float detector_c);

    std::string serial_ = "UNSET";
    std::string software_version_ = "0";
    int schema_ = 2;
    int calibration_version_ = 1;
    Planck planck_{};
    Atmosphere atmosphere_{};
    float emissivity_ = 0.95f;
    float reflected_c_ = 20.f;
    float atmospheric_c_ = 20.f;
    float window_c_ = 20.f;
    float window_transmission_ = 1.f;
    float lens_transmission_ = 1.f;
    float distance_m_ = 1.f;
    float humidity_ = 0.5f;
    float shutter_delta_c_ = 0.4f;
    int width_ = 0;
    int height_ = 0;
    std::vector<float> gain_;
    std::vector<float> offset_;
    std::vector<uint8_t> bad_;
    std::vector<int32_t> lut_;
    bool radiometric_ = true;

    std::vector<RegionState> regions_;
    bool frozen_ = false;
    bool shutter_request_ = false;
    bool shutterless_ = false;
    float shutterless_counts_per_c_ = 0.f;
    float cal_detector_c_ = 25.f;
    float last_ffc_detector_c_ = 25.f;
    int shutter_count_ = 0;
    bool calibrating_ = false;
    bool has_published_ = false;
    std::vector<float> published_;
    uint64_t last_frame_id_ = 0;
    uint64_t last_timestamp_us_ = 0;
    bool have_frame_ = false;
    int dropped_frames_ = 0;
    int display_dropped_ = 0;
    std::vector<int> display_queue_;
    int measurement_frame_ = 0;
    uint64_t stale_limit_us_ = 2'000'000;
    bool stale_ = false;
    bool ffc_open_ = false;
    bool ffc_recovered_ = false;
    double shutter_bias_ = 0;
    std::array<bool, kPoolSlots> slot_used_{};
    int active_gain_mode_ = 0;
    float gain_mode_split_c_ = 35.f;
    std::vector<float> gain_hi_;
    std::vector<float> offset_hi_;
};

std::vector<uint16_t> hostOrder(const uint16_t* samples, int count, bool big_endian);

}  // namespace thermal
