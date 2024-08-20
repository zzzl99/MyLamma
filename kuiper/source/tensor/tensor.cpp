#include "tensor/tensor.h"
#include <glog/logging.h>
#include <numeric>
namespace tensor{
    template <typename T, typename Tp>
    static size_t reduce_dimension(T begin, T end, Tp init) {
        if (begin >= end) {
            return 0;
        }
    }
    static size_t data_type_size(base::DataType data_type) {
        switch (data_type) {
            case base::DataType::kDataTypeFp32: {
                return 4;
            }
            case base::DataType::kDataTypeInt8: {
                return 1;
            }
            case base::DataType::kDataTypeInt32: {
                return 4;
            }
            default: {
                LOG(FATAL) << "Unknown data type size for " << int(data_type);
                return 0;
            }
        }
    }
    //dims为1
    Tensor::Tensor(base::DataType data_type, int32_t dim0, bool need_alloc,
                   std::shared_ptr<base::DeviceAllocator> alloc, void *ptr)
            :data_type_(data_type){
        dims_.push_back(dim0);
        size_ = dim0;

        //如果need_alloc == true表示需要分配内存
        if(need_alloc && alloc){
            allocate(alloc);
        }else{
            //否则就要提供指向内存的指针ptr
            if(ptr != nullptr){
                CHECK(need_alloc == false)
                                << "The need_alloc is  true when ptr parameter is not a null pointer.";
                //下面初始化buffer
                //如果没有内存分配器，那tesor属于外部变量，不由buffer管理
                if(!alloc){
                    std::shared_ptr<base::Buffer> buffer = std::make_shared<base::Buffer>(
                            data_type_size(data_type)*size_, nullptr, ptr, true
                    );
                    this->buffer_ = buffer;
                }
                //下面初始化buffer
                //如果有内存分配器,就吧这个内存分配器给buffer，且该tensor将来由buffer释放，因为有内存分配器
                else {
                    std::shared_ptr<base::Buffer> buffer = std::make_shared<base::Buffer>(
                            data_type_size(data_type)*size_,alloc,ptr, false
                    );
                    this->buffer_ = buffer;
                }
            }
        }


    }
    //dims 为2
    Tensor::Tensor(base::DataType data_type, int32_t dim0, int32_t dim1, bool need_alloc,
                   std::shared_ptr<base::DeviceAllocator> alloc, void *ptr)
                   :data_type_(data_type){
        dims_.push_back(dim0);
        dims_.push_back(dim1);
        size_ = dim0 * dim1;
        //是否需要内存分配器以及内存分配器是否存在
        //如果need_alloc == true表示使用内存分配器来分配内存
        if(need_alloc && alloc){
            allocate(alloc);
        }else{
            //如果有指向数据的指针，只需要创建buffer进行管理
            if(ptr != nullptr){
                CHECK(need_alloc == false)
                << "The need_alloc is  true when ptr parameter is not a null pointer.";
                if(!alloc){
                    std::shared_ptr<base::Buffer> buffer = std::make_shared<base::Buffer>(
                            data_type_size(data_type)*size_, nullptr, ptr, true
                            );
                    this->buffer_ = buffer;
                } else {
                    std::shared_ptr<base::Buffer> buffer = std::make_shared<base::Buffer>(
                            data_type_size(data_type)*size_,alloc,ptr, false
                    );
                    this->buffer_ = buffer;
                }
            }
        }


    }
    //dims为3
    Tensor::Tensor(base::DataType data_type, int32_t dim0, int32_t dim1,int32_t dim2, bool need_alloc,
                   std::shared_ptr<base::DeviceAllocator> alloc, void *ptr)
            :data_type_(data_type){
        dims_.push_back(dim0);
        dims_.push_back(dim1);
        dims_.push_back(dim2);
        size_ = dim0 * dim1 * dim2;
        //是否需要内存分配器以及内存分配器是否存在
        //如果need_alloc == true表示使用内存分配器来分配内存
        if(need_alloc && alloc){
            allocate(alloc);
        }else{
            //如果有指向数据的指针，只需要创建buffer进行管理
            if(ptr != nullptr){
                CHECK(need_alloc == false)
                                << "The need_alloc is  true when ptr parameter is not a null pointer.";
                if(!alloc){
                    std::shared_ptr<base::Buffer> buffer = std::make_shared<base::Buffer>(
                            data_type_size(data_type)*size_, nullptr, ptr, true
                    );
                    this->buffer_ = buffer;
                } else {
                    std::shared_ptr<base::Buffer> buffer = std::make_shared<base::Buffer>(
                            data_type_size(data_type)*size_,alloc,ptr, false
                    );
                    this->buffer_ = buffer;
                }
            }
        }


    }
    //dims为4
    Tensor::Tensor(base::DataType data_type, int32_t dim0, int32_t dim1,int32_t dim2,int32_t dim3, bool need_alloc,
                   std::shared_ptr<base::DeviceAllocator> alloc, void *ptr)
            :data_type_(data_type){
        dims_.push_back(dim0);
        dims_.push_back(dim1);
        dims_.push_back(dim2);
        dims_.push_back(dim3);
        size_ = dim0 * dim1 * dim2 * dim3;
        //是否需要内存分配器以及内存分配器是否存在
        //如果need_alloc == true表示使用内存分配器来分配内存
        if(need_alloc && alloc){
            allocate(alloc);
        }else{
            //如果有指向数据的指针，只需要创建buffer进行管理
            if(ptr != nullptr){
                CHECK(need_alloc == false)
                                << "The need_alloc is  true when ptr parameter is not a null pointer.";
                if(!alloc){
                    std::shared_ptr<base::Buffer> buffer = std::make_shared<base::Buffer>(
                            data_type_size(data_type)*size_, nullptr, ptr, true
                    );
                    this->buffer_ = buffer;
                } else {
                    std::shared_ptr<base::Buffer> buffer = std::make_shared<base::Buffer>(
                            data_type_size(data_type)*size_,alloc,ptr, false
                    );
                    this->buffer_ = buffer;
                }
            }
        }


    }
    //返回值表示分配是否成功
    bool Tensor::allocate(std::shared_ptr<base::DeviceAllocator> allocator, bool need_realloc) {
        if(!allocator){
            LOG(ERROR) << "The allocator parameter in the allocate function is null "
                          "pointer!";
            return false;
        }
        size_t byte_size = this->byte_size();
        if (!byte_size) {
            LOG(ERROR) << "The byte_size parameter in the allocate function is equal to zero!";
            return false;
        }
        //若有buffer，则表明该tensor已经有内存管理器在管理了
        //所以need_realloc为false表示已分配有内存不需要分配
        if(buffer_ && byte_size <= buffer_->byte_size()){
            if(!need_realloc) return true;
        }
        //若无内存管理器，有内存分配器，则创建一个buffer_进行内存初始化和管理
        buffer_ = std::make_shared<base::Buffer>(byte_size, allocator, nullptr);
        if(!buffer_->ptr()){
            LOG(ERROR) << "The memory allocated is a null pointer!";
            return false;
        }
        return true;


    }

    size_t Tensor::size() const {return size_;}
    size_t Tensor::byte_size() const {return this->size() * DataTypeSize(data_type_);}

    int32_t Tensor::get_dim(int32_t idx) const {
        CHECK_GE(idx, 0);
        CHECK_LT(idx, this->dims_.size());
        //dims_.at会进行边界检查
        return this->dims_.at(idx);
    }

    bool Tensor::is_empty() const {
        return size_ == 0 || buffer_ == nullptr || buffer_->ptr() == nullptr;
    }
    std::vector<size_t> Tensor::strides() const {
        std::vector<size_t> strides;
        if (!dims_.empty()) {
            for (int32_t i = 0; i < dims_.size() - 1; ++i) {
                size_t stride = reduce_dimension(dims_.begin() + i + 1, dims_.end(), 1);
                strides.push_back(stride);
            }
            strides.push_back(1);
        }
        return strides;
    }
}