//
// Created by lz on 2024/8/18.
//

#ifndef LLAMA_INFER_TENSOR_H
#define LLAMA_INFER_TENSOR_H
#include <glog/logging.h>
#include <armadillo>
#include <memory>
#include <vector>
#include "base/base.h"
#include "base/buffer.h"
namespace tensor {
class Tensor{
public:
    explicit Tensor() = default;
    explicit Tensor(base::DataType data_type, int32_t dim0,
                    bool need_alloc = false,
                    std::shared_ptr<base::DeviceAllocator> alloc = nullptr,
                    void* ptr = nullptr);//构造函数中维度为1的情况
    explicit Tensor(base::DataType data_type, int32_t dim0, int32_t dim1,
                    bool need_alloc = false,
                    std::shared_ptr<base::DeviceAllocator> alloc = nullptr,
                    void* ptr = nullptr);//构造函数中维度为2的情况
    explicit Tensor(base::DataType data_type, int32_t dim0, int32_t dim1,int32_t dim2,
                    bool need_alloc = false,
                    std::shared_ptr<base::DeviceAllocator> alloc = nullptr,
                    void* ptr = nullptr);//构造函数中维度为3的情况
    explicit Tensor(base::DataType data_type, int32_t dim0, int32_t dim1,int32_t dim2,int32_t dim3,
                    bool need_alloc = false,
                    std::shared_ptr<base::DeviceAllocator> alloc = nullptr,
                    void* ptr = nullptr);//构造函数中维度为4的情况
    bool allocate(std::shared_ptr<base::DeviceAllocator> allocator,
                  bool need_realloc = false);
    //算tensor占用多少字节
    size_t byte_size() const;
    //返回tensor张量个数
    size_t size() const;
    template <typename T>
    T* ptr();
    //返回第index位置指针
    template <typename T>
    T* ptr(int64_t index);
    //获取维度
    int32_t get_dim(int32_t idx) const;
    bool is_empty() const;
    std::vector<size_t> strides() const;



private:
    size_t size_ = 0;//张量个数
    std::vector<int32_t> dims_;//张量维度
    std::shared_ptr<base::Buffer> buffer_;//内存管理器
    base::DataType data_type_ = base::DataType::kDataTypeUnknown;//数据类型
};

    template<typename T>
    T *Tensor::ptr() {
        if(!buffer_)
            return nullptr;
        return reinterpret_cast<T*>(buffer_->ptr());
    }

    template<typename T>
    T *Tensor::ptr(int64_t index) {
        CHECK(!buffer_ && !buffer_->ptr())
                        << "The data area buffer of this tensor is empty or it points to a null pointer.";
        return const_cast<T*>(reinterpret_cast<const T*>(buffer_->ptr())) + index;
    }
}




#endif //LLAMA_INFER_TENSOR_H
