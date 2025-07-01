#!/bin/bash
set -e


# Clean previous builds
rm -rf ../xcframework-build
mkdir -p ../xcframework-build/device
mkdir -p ../xcframework-build/simulator

# Compile for iOS device (arm64)
echo "Building for iOS devices (arm64)..."
cargo build --target aarch64-apple-ios --release --features ios

# Compile for iOS simulator (arm64 and x86_64)
echo "Building for iOS simulator (arm64)..."
cargo build --target aarch64-apple-ios-sim --release --features ios

echo "Building for iOS simulator (x86_64)..."
cargo build --target x86_64-apple-ios --release --features ios

# Clean previous framework output
rm -rf ../Binaries

# Create device and simulator libraries
echo "Creating fat library for simulator..."
lipo -create \
  ../target/aarch64-apple-ios-sim/release/libdf.a \
  ../target/x86_64-apple-ios/release/libdf.a \
  -output ../xcframework-build/simulator/libdf.a

# Copy the device library
cp ../target/aarch64-apple-ios/release/libdf.a ../xcframework-build/device/

# Copy or create header files
mkdir -p ../xcframework-build/include
cat > ../xcframework-build/include/DeepFilterNet.h << EOL
#ifndef DEEPFILTERNET_H
#define DEEPFILTERNET_H

#include <stdint.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Opaque pointer to the Rust DFState struct
 */
typedef struct DFState DFState;

DFState* df_create(const uint8_t* modelBytes, int modelSize, int channels, float attenLim);

int df_get_frame_length(DFState* state);

/**
 * Set DeepFilterNet attenuation limit.
 *
 * @param state DFState pointer created via df_create()
 * @param limDb New attenuation limit in dB
 */
void df_set_atten_lim(DFState* state, float limDb);

/**
 * Set DeepFilterNet post filter beta. A beta of 0 disables the post filter.
 *
 * @param state DFState pointer created via df_create()
 * @param beta Post filter attenuation. Suitable range between 0.05 and 0
 */
void df_set_post_filter_beta(DFState* state, float beta);

/**
 * Processes a chunk of samples.
 *
 * @param state DFState pointer created via df_create()
 * @param input Input buffer of length df_get_frame_length()
 * @return Local SNR of the current frame
 */
float df_process_frame(
    DFState* state,
    const int16_t* input,
    int frameSize
);

/**
 * Destroy a DeepFilterNet instance and free its resources.
 *
 * @param state DFState pointer created via df_create()
 */
void df_free(DFState* state);

#ifdef __cplusplus
}
#endif

#endif // DEEPFILTERNET_H
EOL

# Create framework directories for iOS device
mkdir -p ../xcframework-build/out/device/DeepFilterNet.framework/Headers
mkdir -p ../xcframework-build/out/device/DeepFilterNet.framework/Modules
# Create framework directories for iOS simulator
mkdir -p ../xcframework-build/out/simulator/DeepFilterNet.framework/Headers
mkdir -p ../xcframework-build/out/simulator/DeepFilterNet.framework/Modules

# Copy libraries and headers
cp ../xcframework-build/device/libdf.a ../xcframework-build/out/device/DeepFilterNet.framework/DeepFilterNet
cp ../xcframework-build/simulator/libdf.a ../xcframework-build/out/simulator/DeepFilterNet.framework/DeepFilterNet
cp ../xcframework-build/include/DeepFilterNet.h ../xcframework-build/out/device/DeepFilterNet.framework/Headers/
cp ../xcframework-build/include/DeepFilterNet.h ../xcframework-build/out/simulator/DeepFilterNet.framework/Headers/

# Create module map file
cat > ../xcframework-build/out/device/DeepFilterNet.framework/Modules/module.modulemap << EOL
framework module DeepFilterNet {
  umbrella header "DeepFilterNet.h"
  export *
  module * { export * }
}
EOL

# Copy module map to simulator framework
cp ../xcframework-build/out/device/DeepFilterNet.framework/Modules/module.modulemap ../xcframework-build/out/simulator/DeepFilterNet.framework/Modules/

# Create Info.plist for iOS framework
cat > ../xcframework-build/out/device/DeepFilterNet.framework/Info.plist << EOL
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>en</string>
    <key>CFBundleExecutable</key>
    <string>DeepFilterNet</string>
    <key>CFBundleIdentifier</key>
    <string>com.deepfilternet</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>DeepFilterNet</string>
    <key>CFBundlePackageType</key>
    <string>FMWK</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
    <key>CFBundleVersion</key>
    <string>1</string>
    <key>MinimumOSVersion</key>
    <string>13.0</string>
    <key>UIDeviceFamily</key>
    <array>
      <integer>1</integer>
      <integer>2</integer>
    </array>
    <key>NSPrincipalClass</key>
    <string></string>
    <key>DTPlatformName</key>
    <string>iphoneos</string>
    <key>CFBundleSupportedPlatforms</key>
    <array>
      <string>iPhoneOS</string>
    </array>
</dict>
</plist>
EOL

# Create Info.plist for simulator framework
cp ../xcframework-build/out/device/DeepFilterNet.framework/Info.plist ../xcframework-build/out/simulator/DeepFilterNet.framework/

# Create XCFramework
echo "Creating XCFramework..."
xcodebuild -create-xcframework \
  -framework ../xcframework-build/out/device/DeepFilterNet.framework \
  -framework ../xcframework-build/out/simulator/DeepFilterNet.framework \
  -output ../Binaries/DeepFilterNet.xcframework

echo "XCFramework created at DeepFilterNet.xcframework"