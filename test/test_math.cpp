#include <glog/logging.h>
#include <gtest/gtest.h>
#include <armadillo>

// 测试用例：测试数学加法
TEST(test_math, add) {
    using namespace arma;
    arma::fmat f1 = "1,2,3;";
    arma::fmat f2 = "1,2,3;";
    arma::fmat f3 = f1 + f2;
    ASSERT_EQ(f3.at(0), 2); // 检查第一个元素是否为 2
    ASSERT_EQ(f3.at(1), 4); // 检查第二个元素是否为 4
    ASSERT_EQ(f3.at(2), 6); // 检查第三个元素是否为 6
    printf("it is ok!");
}

// 主函数：初始化 Google Test 和 Google Logging，并运行所有测试
int main(int argc, char* argv[]) {
//    LOG(INFO) << "tests...";
    // 初始化 Google Test
    testing::InitGoogleTest(&argc, argv);

    // 初始化 Google Logging，设置应用程序名称为 "ArmadilloTest"
    google::InitGoogleLogging(argv[0]);

    // 设置日志文件输出目录
//    FLAGS_log_dir = "./log/";

    // 同时将日志信息输出到标准错误输出
    FLAGS_alsologtostderr = true;

    // 输出启动测试的日志信息
    LOG(INFO) << "Starting Armadillo tests...";

    // 运行所有测试用例，并返回测试结果
    return RUN_ALL_TESTS();
}
