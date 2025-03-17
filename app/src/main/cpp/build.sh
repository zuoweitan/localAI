set -e
cmake --preset android-release
cmake --build --preset android-release

mkdir -p lib
cp ./build/android/qnnlibs/*.so ./lib
cp ./build/android/bin/arm64-v8a/libstable_diffusion_core.so ./lib
mkdir -p ../jniLibs/arm64-v8a/
cp ./lib/*.so ../jniLibs/arm64-v8a/
