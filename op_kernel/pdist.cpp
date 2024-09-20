#include "kernel_operator.h"
using namespace AscendC;
template<typename T> __aicore__ inline void Reduce(const LocalTensor<T> &x, uint32_t length) {
    while (length > 32 / sizeof(T)) {
        length >>= 1;
        Add(x, x, x[length], length);
        PipeBarrier<PIPE_V>();
    }
    BlockReduceSum(x, x, 1, 32 / sizeof(T), 1, 1, 8);
}
template<typename T> class BruteForce {
public:
    __aicore__ inline BruteForce() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, float p, int n, int m) {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->p = p;
        this->n = n;
        this->m = m;
        this->copym = (m * sizeof(T) + 32 - 1) / 32 * 32 / sizeof(T);
        xGm.SetGlobalBuffer((__gm__ T*)x, n * m);
        yGm.SetGlobalBuffer((__gm__ T*)y, n * (n - 1) / 2);
        pipe.InitBuffer(QA, 2, 1024 * 4);
        pipe.InitBuffer(QB, 2, 1024 * 4);
        pipe.InitBuffer(QY, 2, 1024 * 4);
    }
    __aicore__ inline void Process() {
        int index = 0;
        for (int i = 0; i < n; ++i) {
            LocalTensor<T> a_tmp = QA.AllocTensor<T>();
            DataCopy(a_tmp, xGm[i * m], copym);
            QA.EnQue(a_tmp);
            LocalTensor<T> a = QA.DeQue<T>();
            LocalTensor<T> y = QY.AllocTensor<T>();
            for (int j = i + 1; j < n; ++j) {
                LocalTensor<T> b_tmp = QB.AllocTensor<T>();
                DataCopy(b_tmp, xGm[j * m], copym);
                QB.EnQue(b_tmp);
                LocalTensor<T> b = QB.DeQue<T>();
                Sub(b, a, b, copym);
                Abs(b, b, copym);
                Ln(b, b, copym);
                Muls(b, b, (T)p, copym);
                Exp(b, b, copym);
                Sum(y[j - (i + 1)], b, SumParams{1, copym, m});
                QB.FreeTensor(b);
            }
            int length = n - i - 1;
            length = (length * sizeof(T) + 32 - 1) / 32 * 32 / sizeof(T);
            Ln(y, y, length);
            Muls(y, y, (T)(1.0f / p), length);
            Exp(y, y, length);
            QY.EnQue<T>(y);
            QA.FreeTensor(a);
            LocalTensor<T> y_tmp = QY.DeQue<T>();

            DataCopy(yGm[index], y_tmp, length);
            QY.FreeTensor(y_tmp);
            index += n - i - 1;
        }
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 2> QA, QB;
    TQue<QuePosition::VECOUT, 2> QY;
    GlobalTensor<T> xGm, yGm;
    float p;
    uint32_t n;
    uint32_t m, copym;
};
template<> class BruteForce<float16_t> {
public:
    using T = float16_t;
    __aicore__ inline BruteForce() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, float p, int n, int m) {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->p = p;
        this->n = n;
        this->m = m;
        this->copym = (m * sizeof(T) + 32 - 1) / 32 * 32 / sizeof(T);
        xGm.SetGlobalBuffer((__gm__ T*)x, n * m);
        yGm.SetGlobalBuffer((__gm__ T*)y, n * (n - 1) / 2);
        pipe.InitBuffer(QA, 2, 1024 * 4);
        pipe.InitBuffer(QB, 2, 1024 * 4);
        pipe.InitBuffer(QY, 2, 1024 * 4);
        pipe.InitBuffer(BA, 1024 * 4);
        pipe.InitBuffer(BB, 1024 * 4);
        pipe.InitBuffer(BY, 1024 * 4);
    }
    __aicore__ inline void Process() {
        auto a = BA.Get<float>(), b = BB.Get<float>(), y = BY.Get<float>();
        int index = 0;
        for (int i = 0; i < n; ++i) {
            LocalTensor<T> a_tmp = QA.AllocTensor<T>();
            DataCopy(a_tmp, xGm[i * m], copym);
            QA.EnQue(a_tmp);
            LocalTensor<T> a16 = QA.DeQue<T>();
            Cast(a, a16, RoundMode::CAST_NONE, copym);
            LocalTensor<T> y16 = QY.AllocTensor<T>();
            for (int j = i + 1; j < n; ++j) {
                LocalTensor<T> b_tmp = QB.AllocTensor<T>();
                DataCopy(b_tmp, xGm[j * m], copym);
                QB.EnQue(b_tmp);
                LocalTensor<T> b16 = QB.DeQue<T>();
                Cast(b, b16, RoundMode::CAST_NONE, copym);
                Sub(b, a, b, copym);
                Abs(b, b, copym);
                Ln(b, b, copym);
                Muls(b, b, p, copym);
                Exp(b, b, copym);
                Sum(y[j - (i + 1)], b, SumParams{1, copym, m});
                QB.FreeTensor(b16);
            }
            int length = n - i - 1;
            length = (length * sizeof(T) + 32 - 1) / 32 * 32 / sizeof(T);
            Ln(y, y, length);
            Muls(y, y, (1.0f / p), length);
            Exp(y, y, length);
            Cast(y16, y, RoundMode::CAST_NONE, copym);
            QY.EnQue<T>(y16);
            QA.FreeTensor(a16);
            LocalTensor<T> y_tmp = QY.DeQue<T>();

            DataCopy(yGm[index], y_tmp, length);
            QY.FreeTensor(y_tmp);
            index += n - i - 1;
        }
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 2> QA, QB;
    TQue<QuePosition::VECOUT, 2> QY;
    TBuf<QuePosition::VECCALC> BA, BB, BY;
    GlobalTensor<T> xGm, yGm;
    float p;
    uint32_t n;
    uint32_t m, copym;
};

extern "C" __global__ __aicore__ void pdist(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    BruteForce<DTYPE_X> op;
    op.Init(x, y, tiling_data.p, tiling_data.n, tiling_data.m);
    for (int i = 0; i < 2; ++i)
        op.Process();
}