set -e

# 切换到脚本所在目录
cd "$(dirname "$0")"

cmake --preset android-release -DCMAKE_POLICY_VERSION_MINIMUM=3.5
# 添加-j2选项限制并行编译以解决内存不足问题
cmake --build --preset android-release -- -j2

mkdir -p lib
cp ./build/android/qnnlibs/*.so ./lib
cp ./build/android/bin/arm64-v8a/libstable_diffusion_core.so ./lib
mkdir -p ../jniLibs/arm64-v8a/
cp ./lib/*.so ../jniLibs/arm64-v8a/