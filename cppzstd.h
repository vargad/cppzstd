// 2018 Daniel Varga (vargad88@gmail.com)
#pragma once

#include <zstd.h>

#include <vector>
#include <stdexcept>

namespace Zstd {

    struct ZError : std::runtime_error
    {
        using std::runtime_error::runtime_error;
    };

    struct ZUnknownSize : ZError
    {
        using ZError::ZError;
    };

    class TrainData
    {
    public:
        TrainData(std::size_t size=100*1024*1024)
            : mDictBuffer(size)
        {}

        template <typename SampleType>
        void train(std::vector<SampleType> const &samples) {
            std::vector<std::size_t> sizes(samples.size(), sizeof(SampleType));
            auto err = ZDICT_trainFromBuffer(static_cast<void*>(mDictBuffer.data()), size(),
                    static_cast<void const*>(samples.data()), sizes, samples.size());
            if (ZDICT_isError(err)) {
                throw ZError(std::string("train error: {}")+ZDICT_getErrorName(err));
            }
        }

        void const* data() const { return static_cast<void const*>(mDictBuffer.data()); }
        std::size_t size() const { return mDictBuffer.size(); }

    private:
        std::vector<char> mDictBuffer;
    };

    class CDict
    {
    public:
        CDict(CDict &&) = default;
        CDict& operator=(CDict &&) = default;
        CDict(TrainData const &data, int compressionLevel) {
            mDict = ZSTD_createCDict(data.data(), data.size(), compressionLevel);
            if (mDict == nullptr) throw std::bad_alloc();
        }
        CDict(CDict const&) = delete;
        CDict& operator=(CDict const&) = delete;

        ~CDict() {
            ZSTD_freeCDict(mDict);
        }

        static int maxCompressionLevel() { return ZSTD_maxCLevel(); }

        ZSTD_CDict const* get() const { return mDict; }

    private:
        ZSTD_CDict *mDict;
    };

    class Compress {
    public:
        Compress(Compress&) = delete;
        Compress(Compress&&) = delete;
        Compress& operator=(Compress&) = delete;
        Compress& operator=(Compress&&) = delete;

        static std::size_t compressbound(std::size_t srcSize) { return ZSTD_compressBound(srcSize); }

        static void compress(std::vector<char> &dst, const void* src, size_t srcSize,
                int compressionLevel)
        {
            dst.resize(compressbound(srcSize));
            auto res = ZSTD_compressCCtx(instance().mContext,
                    static_cast<void*>(dst.data()), dst.size(), src, srcSize,
                    compressionLevel);
            if (static_cast<bool>(ZSTD_isError(res))) {
                throw ZError(fmt::format(fmt("compress error: {}"), ZSTD_getErrorName(res)));
            }
            dst.resize(res);
        }

        static std::size_t compress(void* dst, size_t dstCapacity,
                    const void* src, size_t srcSize, int compressionLevel) {
            auto res = ZSTD_compressCCtx(instance().mContext, dst, dstCapacity, src, srcSize, compressionLevel);
            if (static_cast<bool>(ZSTD_isError(res))) {
                throw ZError(fmt::format(fmt("compress error: {}"), ZSTD_getErrorName(res)));
            }
            return res;
        }

        static void compress(CDict const &dict,
                    std::vector<char> &dst,
                    const void* src, size_t srcSize)
        {
            dst.resize(compressbound(srcSize));
            auto res = ZSTD_compress_usingCDict(instance().mContext,
                    static_cast<void*>(dst.data()), dst.size(), src, srcSize,
                    dict.get());
            if (static_cast<bool>(ZSTD_isError(res))) {
                throw ZError(fmt::format(fmt("compress error: {}"), ZSTD_getErrorName(res)));
            }
            dst.resize(res);
        }

        static std::size_t compress(CDict const &dict,
                    void* dst, size_t dstCapacity,
                    const void* src, size_t srcSize)
        {
            auto res = ZSTD_compress_usingCDict(instance().mContext, dst, dstCapacity, src, srcSize,
                    dict.get());
            if (static_cast<bool>(ZSTD_isError(res))) {
                throw ZError(fmt::format(fmt("compress error: {}"), ZSTD_getErrorName(res)));
            }
            return res;
        }
    private:
        static Compress& instance() {
            thread_local Compress instr;
            return instr;
        }

        Compress() {
            mContext = ZSTD_createCCtx();
            if (mContext == nullptr) throw std::bad_alloc();
        }

        ~Compress() {
            ZSTD_freeCCtx(mContext);
        }

        ZSTD_CCtx *mContext;
    };

    class DDict
    {
    public:
        DDict(DDict &&) = default;
        DDict& operator=(DDict &&) = default;
        DDict(TrainData const &data) {
            mDict = ZSTD_createDDict(data.data(), data.size());
            if (mDict == nullptr) throw std::bad_alloc();
        }
        DDict(DDict const&) = delete;
        DDict& operator=(DDict const&) = delete;

        ~DDict() {
            ZSTD_freeDDict(mDict);
        }

        ZSTD_DDict const* get() const { return mDict; }

    private:
        ZSTD_DDict *mDict;
    };

    class Decompress {
    public:
        Decompress(Decompress&) = delete;
        Decompress(Decompress&&) = delete;
        Decompress& operator=(Decompress&) = delete;
        Decompress& operator=(Decompress&&) = delete;

        static std::size_t frameContentSize(const void *src, size_t srcSize) {
            auto size = ZSTD_getFrameContentSize(src, srcSize);
            if (size == ZSTD_CONTENTSIZE_UNKNOWN)  throw ZUnknownSize("unknown size");
            if (size == ZSTD_CONTENTSIZE_ERROR)  throw ZError("cannot determine size");
            return size;
        }

        static std::size_t decompress(void* dst, size_t dstCapacity,
                    const void* src, size_t srcSize)
        {
            auto res = ZSTD_decompressDCtx(instance().mContext, dst, dstCapacity, src, srcSize);
            if (static_cast<bool>(ZSTD_isError(res))) {
                throw ZError(fmt::format(fmt("compress error: {}"), ZSTD_getErrorName(res)));
            }
            return res;
        }

        static std::size_t decompress(DDict const &dict,
                    void* dst, size_t dstCapacity,
                    const void* src, size_t srcSize)
        {
            auto res = ZSTD_decompress_usingDDict(instance().mContext, dst, dstCapacity, src, srcSize,
                    dict.get());
            if (static_cast<bool>(ZSTD_isError(res))) {
                throw ZError(fmt::format(fmt("compress error: {}"), ZSTD_getErrorName(res)));
            }
            return res;
        }
    private:

        static Decompress& instance() {
            thread_local Decompress instr;
            return instr;
        }

        Decompress() {
            mContext = ZSTD_createDCtx();
            if (mContext == nullptr) throw std::bad_alloc();
        }

        ~Decompress() {
            ZSTD_freeDCtx(mContext);
        }

        ZSTD_DCtx *mContext;
    };

}
