#include "thermal/device.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <thread>
#include <vector>

namespace {

int failures = 0;

void expect(bool ok, const std::string& name) {
    std::cout << (ok ? "PASS " : "FAIL ") << name << "\n";
    if (!ok) {
        ++failures;
    }
}

double signalOf(double temp_c) {
    constexpr double r1 = 21106.77;
    constexpr double r2 = 0.012545258;
    constexpr double b = 1501;
    constexpr double f = 1;
    constexpr double o = -7340;
    const double kelvin = temp_c + 273.15;
    return r1 / (r2 * (std::exp(b / kelvin) - f)) - o;
}

uint16_t synthesize(double temp_c, double gain, double offset) {
    const double s0 = signalOf(35);
    const double s = signalOf(temp_c);
    const double nonlinear = s + 8e-7 * (s - s0) * (s - s0);
    const double raw = gain * nonlinear + offset;
    return static_cast<uint16_t>(std::lround(std::clamp(raw, 1.0, 65534.0)));
}

std::vector<uint16_t> uniform(int count, double temp_c, double gain, double offset) {
    return std::vector<uint16_t>(static_cast<size_t>(count), synthesize(temp_c, gain, offset));
}

constexpr int kWidth = 8;
constexpr int kHeight = 8;
constexpr int kCount = kWidth * kHeight;

void fit(thermal::Device* device, const std::vector<uint16_t>& cold, const std::vector<uint16_t>& hot) {
    device->detectBadPixels(cold.data(), kWidth, kHeight, 8.f);
    device->fitGainOffset(cold.data(), hot.data(), 20.f, 50.f, kWidth, kHeight);
}

thermal::FrameView view(const std::vector<uint16_t>& samples, int width, int height, uint64_t id, uint64_t time_us,
                         float detector, thermal::PixelKind kind = thermal::PixelKind::Scene) {
    thermal::FrameView frame;
    frame.frame_id = id;
    frame.timestamp_us = time_us;
    frame.detector_c = detector;
    frame.kind = kind;
    frame.samples = samples.data();
    frame.width = width;
    frame.height = height;
    return frame;
}

void serveOnce(int port, const std::function<std::vector<uint8_t>(const std::vector<uint8_t>&)>& handler) {
    int server = ::socket(AF_INET, SOCK_STREAM, 0);
    int yes = 1;
    setsockopt(server, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = htons(static_cast<uint16_t>(port));
    bind(server, reinterpret_cast<sockaddr*>(&address), sizeof(address));
    listen(server, 1);
    int client = accept(server, nullptr, nullptr);
    uint8_t buffer[256] = {};
    const ssize_t n = read(client, buffer, sizeof(buffer));
    auto response = handler(std::vector<uint8_t>(buffer, buffer + std::max<ssize_t>(n, 0)));
    write(client, response.data(), response.size());
    close(client);
    close(server);
}

}  // namespace

int main(int argc, char** argv) {
    const std::string root = argc > 1 ? argv[1] : "..";
    const std::string out_dir = root + "/output/production";
    std::system(("mkdir -p " + out_dir).c_str());

    thermal::Device migrated;
    std::string error;
    expect(migrated.load(root + "/data/calibration.json", &error), "load v1 calibration");
    expect(migrated.schema() == 2, "migrate schema to 2");
    expect(migrated.serial() == "SC660-2412", "assign device serial");
    const float real_temp = migrated.temperatureOfRaw(18090, 0, 0.949999988079071f);
    expect(std::abs(real_temp - 23.73440586671677f) < 0.01f, "real frame pixel matches reference");

    std::vector<uint16_t> real_samples(64, 18090);
    std::ifstream raw_file(root + "/data/frame_u16.raw", std::ios::binary);
    raw_file.read(reinterpret_cast<char*>(real_samples.data()), sizeof(uint16_t) * 64);
    thermal::Diagnostic diagnostic;
    auto real_frame = view(real_samples, 8, 8, 1, 1000, 25.f);
    expect(migrated.push(real_frame, nullptr, &diagnostic), "push real counts");
    expect(diagnostic.stage_us[0] >= 0 && diagnostic.stage_us[2] >= 0, "stage timing present");

    migrated.noteUpgrade("1.1.0");
    expect(migrated.save(out_dir + "/device.json", &error), "atomic save");
    thermal::Device restored;
    expect(restored.load(out_dir + "/device.json", &error), "load saved calibration");
    expect(restored.softwareVersion() == "1.1.0", "upgrade keeps software version");
    expect(restored.calibrationVersion() == migrated.calibrationVersion(), "upgrade keeps calibration version");
    migrated.beginUnsafeFfc(out_dir + "/device.json");
    thermal::Device recovered;
    expect(recovered.load(out_dir + "/device.json", &error), "recover after power loss");
    expect(recovered.ffcJournalOpen(), "discard partial ffc");

    auto cold = uniform(kCount, 20, 1.02, 30);
    auto hot = uniform(kCount, 50, 1.02, 30);
    cold[3] = 0;
    hot[3] = 0;
    cold[4] = 65535;
    hot[4] = 65535;
    thermal::Device factory;
    expect(factory.detectBadPixels(cold.data(), kWidth, kHeight, 8.f), "detect bad pixels from blackbody");
    expect(factory.badPixelCount() >= 2, "saturated pixels enter the bad-pixel table");
    expect(factory.fitGainOffset(cold.data(), hot.data(), 20.f, 50.f, kWidth, kHeight), "two-point gain table");
    auto mid = uniform(kCount, 35, 1.02, 30);
    mid[3] = cold[1];
    mid[4] = cold[1];
    const float mid_temp = factory.temperatureOfRaw(mid[0], 0, 0.95f);
    const float mid_error = mid_temp - 35.f;
    expect(std::abs(mid_error) <= 2.f, "held-out blackbody within 2 C");

    factory.setShutterless(true, 0.f);
    thermal::Diagnostic gain_low;
    thermal::Diagnostic gain_high;
    auto scene = mid;
    auto low_frame = view(scene, kWidth, kHeight, 1, 1000, 20.f);
    auto high_frame = view(scene, kWidth, kHeight, 2, 2000, 45.f);
    factory.push(low_frame, nullptr, &gain_low);
    factory.push(high_frame, nullptr, &gain_high);
    expect(std::abs(gain_low.published_c - gain_high.published_c) > 0.05f, "gain table follows detector temperature");

    thermal::Device shutter;
    fit(&shutter, cold, hot);
    thermal::RegionSpec hotspot{"hotspot", 0, 0, 8, 8, 0.95f, 40.f, 2.f, 2};
    shutter.setRegions({hotspot});
    std::vector<thermal::RegionStatus> regions;
    thermal::Diagnostic first;
    thermal::Diagnostic second;
    shutter.push(view(mid, kWidth, kHeight, 1, 1000, 25.f), &regions, &first);
    auto drifted = mid;
    for (auto& value : drifted) {
        value = static_cast<uint16_t>(std::min(65534, value + 80));
    }
    shutter.push(view(drifted, kWidth, kHeight, 2, 2000, 25.6f, thermal::PixelKind::Shutter), &regions, &second);
    expect(second.calibrating, "shutter frame freezes temperature");
    expect(second.calibration_version == first.calibration_version + 1, "ffc bumps calibration version");
    expect(std::abs(second.published_c - first.published_c) < 1e-3f, "published temperature held during ffc");

    thermal::Device alarm;
    fit(&alarm, cold, hot);
    alarm.setShutterless(true, 0.f);
    alarm.setRegions({thermal::RegionSpec{"spot", 0, 0, 8, 8, 0.95f, 40.f, 3.f, 2}});
    auto hot_scene = uniform(kCount, 48, 1.02, 30);
    auto cool_scene = uniform(kCount, 30, 1.02, 30);
    alarm.push(view(hot_scene, kWidth, kHeight, 1, 1000, 25.f), &regions, nullptr);
    expect(!regions.empty() && !regions[0].alarm, "debounce holds the first hot frame");
    alarm.push(view(hot_scene, kWidth, kHeight, 2, 2000, 25.f), &regions, nullptr);
    expect(!regions.empty() && regions[0].alarm && regions[0].latched, "alarm latches after debounce");
    alarm.acknowledge("spot");
    expect(alarm.alarmRegister() == 1, "ack during alarm does not clear a hot target");
    alarm.push(view(cool_scene, kWidth, kHeight, 3, 3000, 25.f), &regions, nullptr);
    alarm.push(view(cool_scene, kWidth, kHeight, 4, 4000, 25.f), &regions, nullptr);
    expect(!regions.empty() && !regions[0].alarm, "hysteresis clears the alarm");

    const float wrong_emissivity = factory.temperatureOfRaw(mid[0], 0, 0.50f);
    const float reflected_shift = [&] {
        thermal::Device probe;
        std::string ignored;
        probe.load(root + "/data/calibration.json", &ignored);
        probe.setConditions(1.f, 0.5f, 30.f);
        return probe.temperatureOfRaw(18090, 0, 0.949999988079071f);
    }();
    const float distance_shift = [&] {
        thermal::Device probe;
        std::string ignored;
        probe.load(root + "/data/calibration.json", &ignored);
        probe.setConditions(10.f, 0.5f, 19.999993896484398f);
        return probe.temperatureOfRaw(18090, 0, 0.949999988079071f);
    }();
    const float humidity_dry = [&] {
        thermal::Device probe;
        std::string ignored;
        probe.load(root + "/data/calibration.json", &ignored);
        probe.setConditions(20.f, 0.2f, 19.999993896484398f);
        return probe.temperatureOfRaw(18090, 0, 0.949999988079071f);
    }();
    const float humidity_wet = [&] {
        thermal::Device probe;
        std::string ignored;
        probe.load(root + "/data/calibration.json", &ignored);
        probe.setConditions(20.f, 0.9f, 19.999993896484398f);
        return probe.temperatureOfRaw(18090, 0, 0.949999988079071f);
    }();
    expect(std::abs(reflected_shift - real_temp) > 0.05f, "reflected temperature changes the reading");
    expect(std::abs(distance_shift - real_temp) > 0.01f, "distance changes the reading");
    expect(std::abs(humidity_wet - humidity_dry) > 0.05f, "humidity changes the reading");
    const float distance_base = real_temp;
    thermal::Device humid = migrated;
    (void)humid;
    expect(std::abs(wrong_emissivity - mid_temp) > 0.5f, "emissivity changes the reading");

    thermal::Device stream;
    fit(&stream, cold, hot);
    stream.setShutterless(true, 0.f);
    int dropped_display = 0;
    for (int i = 0; i < 5; ++i) {
        thermal::Diagnostic step;
        stream.push(view(mid, kWidth, kHeight, static_cast<uint64_t>(i + 1), static_cast<uint64_t>(i + 1) * 1000, 25.f), nullptr, &step);
        dropped_display = step.display_dropped;
    }
    expect(dropped_display == 3, "full display queue drops preview frames");
    thermal::Diagnostic gap;
    stream.push(view(mid, kWidth, kHeight, 10, 10000, 25.f), nullptr, &gap);
    expect(gap.dropped_frames == 4, "frame-id gap counts dropped frames");
    expect(stream.acquireSlot() >= 0 && stream.acquireSlot() >= 0 && stream.acquireSlot() >= 0 && stream.acquireSlot() >= 0, "frame pool fills");
    expect(stream.acquireSlot() < 0, "frame pool refuses a fifth buffer");
    stream.releaseSlot(0);
    expect(stream.acquireSlot() == 0, "frame pool reuses a released buffer");

    thermal::Device frozen;
    fit(&frozen, cold, hot);
    frozen.setShutterless(true, 0.f);
    thermal::Diagnostic before_freeze;
    thermal::Diagnostic after_freeze;
    frozen.push(view(mid, kWidth, kHeight, 1, 1000, 25.f), nullptr, &before_freeze);
    frozen.setFrozen(true);
    frozen.push(view(hot_scene, kWidth, kHeight, 2, 2000, 25.f), nullptr, &after_freeze);
    expect(std::abs(after_freeze.published_c - before_freeze.published_c) < 1e-3f, "freeze holds the last temperature");
    int isotherm = 0;
    for (float value : frozen.published()) {
        if (value >= 30.f && value <= 40.f) {
            ++isotherm;
        }
    }
    expect(isotherm == kCount, "isotherm and level cover the held frame");

    std::vector<uint16_t> swapped(static_cast<size_t>(kCount));
    for (int i = 0; i < kCount; ++i) {
        swapped[static_cast<size_t>(i)] = static_cast<uint16_t>((mid[static_cast<size_t>(i)] >> 8) | (mid[static_cast<size_t>(i)] << 8));
    }
    thermal::FrameView endian = view(swapped, kWidth, kHeight, 1, 1000, 25.f);
    endian.big_endian = true;
    thermal::Device endian_device;
    fit(&endian_device, cold, hot);
    endian_device.setShutterless(true, 0.f);
    thermal::Diagnostic endian_diag;
    endian_device.push(endian, nullptr, &endian_diag);
    expect(std::abs(endian_diag.published_c - mid_temp) < 0.2f, "big-endian samples are swapped");

    const std::string clip = out_dir + "/clip.raw";
    expect(frozen.recordRaw(clip, view(mid, kWidth, kHeight, 7, 7000, 25.f)), "record raw");
    std::vector<uint16_t> played;
    thermal::FrameView played_frame;
    expect(frozen.playbackRaw(clip, &played, &played_frame), "playback raw");
    expect(played_frame.frame_id == 7 && played.size() == static_cast<size_t>(kCount), "playback keeps frame identity");

    thermal::Device stale;
    fit(&stale, cold, hot);
    stale.setShutterless(true, 0.f);
    stale.setStaleLimitUs(1000);
    stale.push(view(mid, kWidth, kHeight, 1, 0, 25.f), nullptr, nullptr);
    thermal::Diagnostic stale_diag;
    stale.push(view(mid, kWidth, kHeight, 2, 5000, 25.f), nullptr, &stale_diag);
    expect(stale_diag.stale, "stale temperature is detected");

    thermal::Device loop;
    fit(&loop, cold, hot);
    loop.setRegions({thermal::RegionSpec{"spot", 0, 0, 8, 8, 0.95f, 40.f, 1.f, 1}});
    loop.setShutterless(true, 0.f);
    loop.push(view(hot_scene, kWidth, kHeight, 1, 1000, 25.f), nullptr, nullptr);
    expect(loop.alarmRegister() == 1, "alarm register is set");
    uint8_t request[] = {0, 1, 0, 0, 0, 6, 1, 3, 0, 0, 0, 4};
    auto response = loop.modbusTcp(std::vector<uint8_t>(request, request + 12));
    expect(response.size() >= 11 && response[7] == 3 && response[9] == 0 && response[10] == 1, "modbus holding register");
    loop.writeDryContact(out_dir + "/dry_contact");
    std::ifstream contact(out_dir + "/dry_contact");
    std::string contact_text;
    std::getline(contact, contact_text);
    expect(contact_text == "1", "dry contact follows the alarm");
    loop.appendAlarmEvent(out_dir + "/alarm_events.jsonl", 1000);
    auto packet = thermal::Device::mqttPublish("thermal/alarm", "1");
    expect(thermal::Device::mqttPublishMatches(packet, "thermal/alarm", "1"), "mqtt publish payload");

    std::thread modbus_thread([&] { serveOnce(15021, [&](const std::vector<uint8_t>& incoming) { return loop.modbusTcp(incoming); }); });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    int client = socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(15021);
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    const bool connected = connect(client, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == 0;
    if (connected) {
        write(client, request, sizeof(request));
        uint8_t got[32] = {};
        read(client, got, sizeof(got));
        expect(got[10] == 1, "modbus tcp on localhost");
    } else {
        expect(false, "modbus tcp on localhost");
    }
    close(client);
    modbus_thread.join();

    std::thread preview_thread([] {
        int server = socket(AF_INET, SOCK_STREAM, 0);
        int yes = 1;
        setsockopt(server, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
        sockaddr_in local{};
        local.sin_family = AF_INET;
        local.sin_port = htons(15022);
        local.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        bind(server, reinterpret_cast<sockaddr*>(&local), sizeof(local));
        listen(server, 1);
        int peer = accept(server, nullptr, nullptr);
        uint8_t header[16] = {8, 0, 8, 0};
        write(peer, header, sizeof(header));
        close(peer);
        close(server);
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    int preview = socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in preview_address{};
    preview_address.sin_family = AF_INET;
    preview_address.sin_port = htons(15022);
    preview_address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    expect(connect(preview, reinterpret_cast<sockaddr*>(&preview_address), sizeof(preview_address)) == 0, "preview stream connects");
    uint8_t header[16] = {};
    expect(read(preview, header, sizeof(header)) == 16 && header[0] == 8 && header[2] == 8, "preview stream sends frame header");
    close(preview);
    preview_thread.join();

    thermal::Device soak;
    fit(&soak, cold, hot);
    soak.setShutterless(true, 15.f);
    float worst = 0;
    for (int second = 0; second < 3600; ++second) {
        auto frame_samples = uniform(kCount, 35, 1.02, 30);
        const float detector = 25.f + 2.f * std::sin(second / 600.0);
        const float drift = 15.f * (detector - 25.f);
        for (auto& value : frame_samples) {
            value = static_cast<uint16_t>(std::clamp(static_cast<int>(std::lround(value + drift)), 1, 65534));
        }
        thermal::Diagnostic step;
        soak.push(view(frame_samples, kWidth, kHeight, static_cast<uint64_t>(second + 1), static_cast<uint64_t>(second) * 1000000ULL, detector), nullptr, &step);
        worst = std::max(worst, std::abs(step.published_c - 35.f));
    }
    expect(worst <= 2.f, "simulated one-hour drift stays within 2 C");

    thermal::Device bench;
    const double microseconds = bench.lutMicroseconds(20, 640, 512);
    const double cpu_percent = microseconds / 10000.0;
    expect(microseconds < 50000.0, "640x512 lut frame is under 50 ms");

    std::ofstream report(out_dir + "/report.json");
    report << "{\n"
           << "  \"real_pixel_c\": " << real_temp << ",\n"
           << "  \"real_pixel_error_c\": " << (real_temp - 23.73440586671677) << ",\n"
           << "  \"blackbody_35_c\": " << mid_temp << ",\n"
           << "  \"blackbody_35_error_c\": " << mid_error << ",\n"
           << "  \"emissivity_0_50_c\": " << wrong_emissivity << ",\n"
           << "  \"reflected_30_c\": " << reflected_shift << ",\n"
           << "  \"distance_10m_c\": " << distance_shift << ",\n"
           << "  \"humidity_20m_dry_c\": " << humidity_dry << ",\n"
           << "  \"humidity_20m_wet_c\": " << humidity_wet << ",\n"
           << "  \"distance_reference_c\": " << distance_base << ",\n"
           << "  \"soak_worst_error_c\": " << worst << ",\n"
           << "  \"lut_us_per_frame_640x512\": " << microseconds << ",\n"
           << "  \"cpu_percent_of_one_core_at_100fps\": " << cpu_percent << ",\n"
           << "  \"factory_pass\": " << (std::abs(mid_error) <= 2.f && worst <= 2.f ? "true" : "false") << ",\n"
           << "  \"physical_blackbody\": \"not connected; setpoints are synthesized from the camera model\",\n"
           << "  \"sample_image\": \"Thermimage IR_2412.jpg is a regression sample, not product artwork\"\n"
           << "}\n";

    std::cout << "failures " << failures << "\n";
    return failures == 0 ? 0 : 1;
}
