// Standard C++ includes
#include <algorithm>
#include <chrono>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <sstream>
#include <vector>

// Standard C includes
#include <cassert>
#include <cmath>

// CUDA includes
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>

//------------------------------------------------------------------------
// Macros
//------------------------------------------------------------------------
#define SEED 124

#define CHECK_CUDA_ERRORS(call) {                                                                   \
    cudaError_t error = call;                                                                       \
    if (error != cudaSuccess) {                                                                     \
            std::ostringstream errorMessageStream;                                                  \
            errorMessageStream << "cuda error:" __FILE__ << ": " << __LINE__ << " ";                \
            errorMessageStream << cudaGetErrorString(error) << "(" << error << ")" << std::endl;    \
            throw std::runtime_error(errorMessageStream.str());                                     \
        }                                                                                           \
    }

template<typename T>
using HostDeviceArray = std::pair < T*, T* > ;

enum Mode
{
    ModeSparsePost,
    ModeSparsePostBitfield,
    ModeMax,
};


const char *const s_ModeNames[] = {
    "Sparse Post",
    "Sparse Post Bitfield"};

//------------------------------------------------------------------------
// SparseMatrix
//------------------------------------------------------------------------
// Yale format sparse matrix
struct SparseMatrix
{
    unsigned int maxRowLength;
    HostDeviceArray<unsigned int> rowLength;
    HostDeviceArray<unsigned int> indices;
};

//------------------------------------------------------------------------
// Timer
//------------------------------------------------------------------------
template<typename A = std::milli>
class Timer
{
public:
    Timer(const std::string &title) : m_Start(std::chrono::high_resolution_clock::now()), m_Title(title)
    {
    }

    ~Timer()
    {
        std::cout << m_Title << get() << std::endl;
    }

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    double get() const
    {
        auto now = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = now - m_Start;
        return duration.count();
    }

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    std::chrono::time_point<std::chrono::high_resolution_clock> m_Start;
    std::string m_Title;
};

//! Divide two integers, rounding up i.e. effectively taking ceil
template<typename A, typename B, typename = std::enable_if_t<std::is_integral_v<A> && std::is_integral_v<B>>>
inline auto ceilDivide(A numerator, B denominator)
{
    return ((numerator + denominator - 1) / denominator);
}

//! Pad an integer to a multiple of another
template<typename A, typename B, typename = std::enable_if_t<std::is_integral_v<A>&& std::is_integral_v<B>>>
inline auto padSize(A size, B blockSize)
{
    return ceilDivide(size, blockSize) * blockSize;
}

//-----------------------------------------------------------------------------
// Device functions
//-----------------------------------------------------------------------------
//  Von Neumann'synBlk exponential distribution generator from Ripley p.230
//  Mean number of U(0,1) per call = 5.2
__device__ float exponentialDist(curandState &rng) {
    float a = 0.0f;

    while (true) {
        float u = curand_uniform(&rng);
        const float u0 = u;

        while (true) {
            float uStar = curand_uniform(&rng);
            if (u < uStar) {
                return  a + u0;
            }

            u = curand_uniform(&rng);

            if (u >= uStar) {
                break;
            }
        }

        a += 1.0f;
    }
}
//-----------------------------------------------------------------------------
// Kernel to initialise device RNG seed
template<typename RNGStateType>
__global__ void initRandomSeed(unsigned int sequenceStart, unsigned int numSeed, RNGStateType *d_rowState)
{
    const int i = threadIdx.x + (blockIdx.x * blockDim.x);
    if (i < numSeed) {
        curand_init(SEED, sequenceStart + i, 0, &d_rowState[i]);
    }

}
//-----------------------------------------------------------------------------
// Kernel to initialise initial Poisson time-to-spike
__global__ void initPoissonTimeToSpike(unsigned int numPoisson, const float *d_meanISI, curandState *d_poissonState,
                                       float *d_timeToSpike)
{
    // Get index of neuron in population
    const int i = threadIdx.x + (blockIdx.x * blockDim.x);
    if (i < numPoisson) {
        d_timeToSpike[i] = d_meanISI[i] * exponentialDist(d_poissonState[i]);
    }
}
//-----------------------------------------------------------------------------
// Kernel to simulate population of poisson neurons
__global__ void poisson(unsigned int numPoisson, const float *d_meanISI, curandState *d_poissonState,
                        float *d_timeToSpike, unsigned int *d_numOutSpikes, unsigned int *d_outSpikes)
{
    // Count and buffer to hold spikes output by this block
    __shared__ unsigned int blockSpikeCount;
    extern __shared__ unsigned int blockOutSpikes[];

    // Offset into global spike output buffer
    __shared__ unsigned int blockSpikeOffset;

    // Get index of neuron in population
    const int i = threadIdx.x + (blockIdx.x * blockDim.x);
    const unsigned int batch = blockIdx.y;

    // Use first thread in each block to zero spike counts
    if (threadIdx.x == 0) {
        blockSpikeCount = 0;
    }
    __syncthreads();

    // If there is a neuron for this thread to simulate
    const unsigned int batchOffset = numPoisson * batch;
    if (i < numPoisson) {
        float tts = d_timeToSpike[batchOffset + i];

        if (tts <= 0.0f) {
            tts += (d_meanISI[batchOffset + i] * exponentialDist(d_poissonState[batchOffset + i]));

            // Add spike to output
            unsigned int blockSpikeIndex = atomicAdd(&blockSpikeCount, 1);
            blockOutSpikes[blockSpikeIndex] = i;
        }

        d_timeToSpike[batchOffset + i] = (tts - 1.0f);
    }

    // If block has emitted any spikes, use the first thread to  
    // determine where in global spike output buffer to copy them
    __syncthreads();
    if (threadIdx.x == 0 && blockSpikeCount > 0) {
        blockSpikeOffset = atomicAdd(&d_numOutSpikes[batch], blockSpikeCount);
    }

    // Copy spikes from block output buffer into correct offset in global buffer
    __syncthreads();
    if (threadIdx.x < blockSpikeCount) {
        d_outSpikes[batchOffset + blockSpikeOffset + threadIdx.x] = blockOutSpikes[threadIdx.x];
    }
}
//-----------------------------------------------------------------------------
// Kernel to simulate population of poisson neurons
__global__ void poissonBitfield(unsigned int numPoisson, const float *d_meanISI, curandState *d_poissonState,
                                float *d_timeToSpike, uint32_t *d_outSpikes)
{
    // Get index of neuron in population
    const int i = threadIdx.x + (blockIdx.x * blockDim.x);
    const unsigned int batch = blockIdx.y;


    // If there is a neuron for this thread to simulate
    const unsigned int batchOffset = numPoisson * batch;
    const unsigned int numBatchWords = (blockDim.y + 31) / 32;
    if (i < numPoisson) {
        float tts = d_timeToSpike[batchOffset + i];

        if (tts <= 0.0f) {
            tts += (d_meanISI[batchOffset + i] * exponentialDist(d_poissonState[batchOffset + i]));

            // Set bit in spike bitfield
            atomicOr(&d_outSpikes[(i * numBatchWords) + (batch / 32)], 1 << (batch % 32));
        }

        d_timeToSpike[batchOffset + i] = (tts - 1.0f);
    }
}
//-----------------------------------------------------------------------------
__global__ void sparsePostIndividualWeight(const unsigned int *d_rowLength, const unsigned int *d_indices, 
                                           unsigned int numPre, unsigned int numPost, unsigned int maxRowLength,
                                           const unsigned int *d_numInSpikes, const unsigned int *d_inSpikes,
                                           const float *d_weights, float *d_outCurrents)
{
    extern __shared__ unsigned int s_buffer[];
    unsigned int *s_spike = &s_buffer[0];
    unsigned int *s_rowLength = &s_buffer[blockDim.x];

    const unsigned int id = threadIdx.x + (blockIdx.x * blockDim.x);
    const unsigned int batch = blockIdx.y;

    // Calculate number of blocks (dictated by shared memory) spikes need to be processed in
    const unsigned int numSpikes = d_numInSpikes[batch];
    const unsigned int numSpikeBlocks = (numSpikes + blockDim.x - 1) / blockDim.x;
    const unsigned int preBatchOffset = numPre * batch;
    const unsigned int postBatchOffset = numPost * batch;

    // Loop through spikes blocks
    for (unsigned int b = 0; b < numSpikeBlocks; b++) {
        // Determine how many spikes are in this block
        const unsigned int numSpikesInBlock = (b == (numSpikeBlocks - 1))
            ? ((numSpikes - 1) % blockDim.x) + 1 : blockDim.x;

        __syncthreads();

       // Use first row of threads in block to read spikes and row lengths into shared memory
        if (threadIdx.x < numSpikesInBlock) {
            const unsigned int i = d_inSpikes[preBatchOffset + (b * blockDim.x) + threadIdx.x];
            s_spike[threadIdx.x] = i;
            s_rowLength[threadIdx.x] = d_rowLength[i];
        }

        __syncthreads();

        // If there is a synapse for this thread to process
        if(id < maxRowLength) {
            // Loop through spikes in block
            for(unsigned int i = 0; i < numSpikesInBlock; i++)
            {
                // If there is a synapse for this thread to process
                if(id < s_rowLength[i]) {
                    // Get postsynaptic index
                    const unsigned int synAddress = (s_spike[i] * maxRowLength) + id;
                    const unsigned int j = d_indices[synAddress];

                    // Add input current
                    atomicAdd(&d_outCurrents[postBatchOffset + j], d_weights[synAddress]);
                }
            }
        }

    }
}
//-----------------------------------------------------------------------------
__global__ void sparsePostBitfieldIndividualWeight(const unsigned int *d_rowLength, const unsigned int *d_indices, 
                                                   unsigned int batchSize, unsigned int numPre, unsigned int numPost, unsigned int maxRowLength,
                                                   const uint32_t *d_inSpikes, const float *d_weights, float *d_outCurrents)
{
    const unsigned int id = threadIdx.x + (blockIdx.x * blockDim.x);
    const unsigned int spikeWord = blockIdx.y;
    const unsigned int numBatchWords = (batchSize + 31) / 32;
     
    assert(numBatchWords == 1);

    // **TODO** parallelise across numBatchWords

    // Parallelising across warps, determine if there were spikes in any batch for each postsynaptic neuron
    uint32_t allBatchSpikeWord = (d_inSpikes[((spikeWord * 32) + (threadIdx.x % 32)) * numBatchWords] == 0) ? 0 : (1 << (threadIdx.x % 32));
    allBatchSpikeWord |= __shfl_xor_sync(0xFFFFFFFF, allBatchSpikeWord, 1);
    allBatchSpikeWord |= __shfl_xor_sync(0xFFFFFFFF, allBatchSpikeWord, 2);
    allBatchSpikeWord |= __shfl_xor_sync(0xFFFFFFFF, allBatchSpikeWord, 4);
    allBatchSpikeWord |= __shfl_xor_sync(0xFFFFFFFF, allBatchSpikeWord, 8);
    allBatchSpikeWord |= __shfl_xor_sync(0xFFFFFFFF, allBatchSpikeWord, 16);

    // Calculate neuron id of highest bit of this word
    unsigned int preNeuronID = (spikeWord * 32) + 31;

    // While there are rows with any spikes
    // **NOTE** will not diverge across whole grid
    while(allBatchSpikeWord != 0) {
        // Calculate leading zeros
        const int numNeuronLZ = __clz(allBatchSpikeWord);
                
        // If all bits have now been processed, zero spike word
        // Otherwise shift past the spike we have found
        allBatchSpikeWord = (numNeuronLZ == 31) ? 0 : (allBatchSpikeWord << (numNeuronLZ + 1));
                
        // Subtract number of leading zeros from neuron ID
        preNeuronID -= numNeuronLZ;
        
        // If there is a synapse for this thread to process
        if(id < d_rowLength[preNeuronID]) {
            // Get postsynaptic index
            const unsigned int synAddress = (preNeuronID * maxRowLength) + id;
            const unsigned int j = d_indices[synAddress];
            const float weight = d_weights[synAddress];

            // Loop through all words containing bits for neurons in each batch
            for(unsigned int i = 0; i < numBatchWords; i++) {
                unsigned int batchSpikeWord = d_inSpikes[(preNeuronID * numBatchWords) + i];

                // Calculate neuron id of highest bit of this word
                unsigned int batch = (i * 32) + 31;

                // While there are more batches where this neuron spikes
                while(batchSpikeWord != 0) {
                    // Calculate leading zeros
                    const int numBatchLZ = __clz(batchSpikeWord);

                    // If all bits have now been processed, zero spike word
                    // Otherwise shift past the spike we have found
                    batchSpikeWord = (numBatchLZ == 31) ? 0 : (batchSpikeWord << (numBatchLZ + 1));
                
                    // Subtract number of leading zeros from neuron ID
                    batch -= numBatchLZ;

                    // Add input current
                    const unsigned int postBatchOffset = numPost * batch;
                    atomicAdd(&d_outCurrents[postBatchOffset + j], weight);

                    // New batch id of the highest bit of this word
                    batch--;
                }
            }
        }

        // New neuron id of the highest bit of this word
        preNeuronID--;
    }
}

//-----------------------------------------------------------------------------
// Host functions
//-----------------------------------------------------------------------------
template<typename T>
HostDeviceArray<T> allocateHostDevice(unsigned int count)
{
    T *array = nullptr;
    T *d_array = nullptr;
    CHECK_CUDA_ERRORS(cudaMallocHost(&array, count * sizeof(T)));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_array, count * sizeof(T)));

    return std::make_pair(array, d_array);
}
//-----------------------------------------------------------------------------
template<typename T>
void hostToDeviceCopy(HostDeviceArray<T> &array, unsigned int count, bool deleteHost=false)
{
    CHECK_CUDA_ERRORS(cudaMemcpy(array.second, array.first, sizeof(T) * count, cudaMemcpyHostToDevice));
    if (deleteHost) {
        CHECK_CUDA_ERRORS(cudaFreeHost(array.first));
        array.first = nullptr;
    }
}
//-----------------------------------------------------------------------------
template<typename T>
void deviceToHostCopy(HostDeviceArray<T> &array, unsigned int count)
{
    CHECK_CUDA_ERRORS(cudaMemcpy(array.first, array.second, count * sizeof(T), cudaMemcpyDeviceToHost));
}
//-----------------------------------------------------------------------------
// Evaluates continued fraction for incomplete beta function by modified Lentz'synBlk method
// Adopted from numerical recipes in C p227
double betacf(double a, double b, double x)
{
    const int maxIterations = 200;
    const double epsilon = 3.0E-7;
    const double fpMin = 1.0E-30;

    const double qab = a + b;
    const double qap = a + 1.0;
    const double  qam = a - 1.0;
    double c = 1.0;

    // First step of Lentz�s method.
    double d = 1.0 - qab * x / qap;
    if(fabs(d) < fpMin) {
        d = fpMin;
    }
    d = 1.0 / d;
    double h = d;
    int m;
    for(m = 1; m <= maxIterations; m++) {
        const double m2 = 2.0 * m;
        const double aa1 = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa1 * d;

        // One step (the even one) of the recurrence.
        if(fabs(d) < fpMin) {
            d = fpMin;
        }
        c = 1.0 + aa1 / c;
        if(fabs(c) < fpMin) {
            c = fpMin;
        }
        d = 1.0 / d;
        h *= d * c;
        const double aa2 = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa2 * d;

        // Next step of the recurrence (the odd one).
        if(fabs(d) < fpMin) {
            d = fpMin;
        }
        c = 1.0 + aa2 / c;
        if(fabs(c) < fpMin) {
            c = fpMin;
        }
        d = 1.0 / d;
        const double del = d * c;
        h *= del;

        // Are we done?
        if(fabs(del - 1.0) < epsilon) {
            break;
        }
    }
    if(m > maxIterations) {
        throw std::runtime_error("a or b too big, or MAXIT too small in betacf");
    }
    return h;
}
//----------------------------------------------------------------------------
// Returns the incomplete beta function Ix(a, spkBlk)
// Adopted from numerical recipes in C p227
double betai(double a, double b, double x)
{
    if(x < 0.0 || x > 1.0) {
        throw std::runtime_error("Bad x in routine betai");
    }

    // Factors in front of the continued fraction.
    double bt;
    if(x == 0.0 || x == 1.0) {
        bt = 0.0;
    }
    else {
        bt = exp(lgamma(a + b) - lgamma(a) - lgamma(b) + a * log(x) + b * log(1.0 - x));
    }

    // Use continued fraction directly.
    if(x < ((a + 1.0) / (a + b + 2.0))) {
        return bt * betacf(a, b, x) / a;
    }
    // Otherwise, use continued fraction after making the 
    // symmetry transformation.
    else {
        return 1.0 - (bt * betacf(b, a, 1.0 - x) / b);
    }
}
//----------------------------------------------------------------------------
// Evaluates inverse CDF of binomial distribution
unsigned int binomialInverseCDF(double cdf, unsigned int n, double p)
{
    if(cdf < 0.0 || 1.0 < cdf) {
        throw std::runtime_error("binomialInverseCDF error - CDF < 0 or 1 < CDF");
    }

    // Loop through ks <= n
    for(unsigned int k = 0; k <= n; k++) {
        // Use incomplete beta function to evalauate CDF, if it'synBlk greater than desired CDF value, return k
        if(betai(n - k, 1 + k, 1.0 - p) > cdf) {
            return k;
        }

    }

    throw std::runtime_error("Invalid CDF parameterse");
}
//-----------------------------------------------------------------------------
template <typename Generator>
void buildFixedProbabilityConnector(unsigned int numPre, unsigned int numPost, float probability,
                                    SparseMatrix &projection, Generator &gen)
{
    const double probabilityReciprocal = 1.0 / std::log(1.0f - probability);

    // Calculate suitable quantile for 0.9999 change when drawing numPre times
    const double quantile = pow(0.9999, 1.0 / (double)numPre);

    // Calculate max row length
    projection.maxRowLength = binomialInverseCDF(quantile, numPost, probability);

    // Allocate memory for row length and indices
    projection.rowLength = allocateHostDevice<unsigned int>(numPre);
    projection.indices = allocateHostDevice<unsigned int>(numPre * projection.maxRowLength);

    // Create RNG to draw probabilities
    std::uniform_real_distribution<> dis(0.0, 1.0);

    // Zero row lengths
    std::fill_n(projection.rowLength.first, numPre, 0);

    // Loop through potential synapses
    const int64_t maxConnections = (int64_t)numPre * (int64_t)numPost;
    for(int64_t s = -1;;) {
        // Skip to next connection
        s += (1 + (int64_t)(std::log(dis(gen)) * probabilityReciprocal));

        // If we haven't skipped past end of matrix
        if(s < maxConnections) {
            // Convert synapse number to pre and post index
            const auto prePost = std::div(s, numPost);

            // Get pointer to start of this presynaptic neuron'synBlk connection row
            unsigned int *rowIndices = &projection.indices.first[prePost.quot * projection.maxRowLength];

            // Add synapse
            rowIndices[projection.rowLength.first[prePost.quot]++] = prePost.rem;
            assert(projection.rowLength.first[prePost.quot] <= projection.maxRowLength);
        }
        else {
            break;
        }
    }

    // Upload connectivity
    hostToDeviceCopy(projection.rowLength, numPre, true);
    hostToDeviceCopy(projection.indices, numPre * projection.maxRowLength, true);
}
//-----------------------------------------------------------------------------
int main(int argc, char *argv[])
{
    try
    {
        unsigned int blockSize = 32;
        unsigned int numPre = 10000;
        unsigned int numPost = 10000;
        const float connectionProbability = 0.1f;
        unsigned int batchSize = 32;
        const float dt = 1.0f;
        const float poissonRate = 10.0f;
    
        // Read mode from command line
        Mode mode;
        if(argc < 2) {
            std::cerr << "Expected parameters specifying:" << std::endl;
            std::cerr << "\t Mode (";
            for(int m = 0; m < ModeMax; m++) {
                std::cerr << m << " = " << s_ModeNames[m];
                if(m != (ModeMax - 1)) {
                    std::cerr << ", ";
                }
            }
            std::cerr << ")" << std::endl;
            return EXIT_FAILURE;
        }
        else {
            mode = (Mode)std::stoul(argv[1]);
        }
    
        // If additional parameters are specified, read N
        if(argc > 2) {
            numPre = numPost = std::stoul(argv[2]);
        }
        // If additional parameters are specified, read number of postsynaptic blocks
        if(argc > 3) {
            batchSize = std::stoul(argv[3]);
        }
        if(argc > 4) {
            blockSize = std::stoul(argv[4]);
        }

        const unsigned int preBlocks = ceilDivide(numPre, blockSize);
        const unsigned int preBatchBlocks = ceilDivide(numPre * batchSize, blockSize);
        const unsigned int numBatchWords = ceilDivide(batchSize, 32);
        std::cout << "Mode:" << s_ModeNames[mode] << ", pre:" << numPre << ", num post:" << numPost << ", batch size:" << batchSize << ", block size:" << blockSize << std::endl;
    
        CHECK_CUDA_ERRORS(cudaSetDevice(0));

        //------------------------------------------------------------------------
        // Configure fixed-probability connector
        //------------------------------------------------------------------------
        // Create arrays to hold post-synaptic currents
        auto outCurrents = allocateHostDevice<float>(numPost * batchSize);
        std::fill_n(&outCurrents.first[0], numPost * batchSize, 0.0f);
        hostToDeviceCopy(outCurrents, numPost * batchSize);

        SparseMatrix sparseMatrix;
        HostDeviceArray<float> weights;
        {
            std::mt19937 gen;
            Timer<std::milli> t("Building sparse matrix:");

            buildFixedProbabilityConnector(numPre, numPost, connectionProbability, sparseMatrix, gen);
            std::cout << "Max row length:" << sparseMatrix.maxRowLength << std::endl;
        }

        // If weight isn't constant (in this context it doesn't really matter WHAT is in the memory)
        {
            Timer<std::milli> t("Initializing weights:");

            // Allocate, fill and upload weight array
            const unsigned int numIndices = numPre * sparseMatrix.maxRowLength;
            weights = allocateHostDevice<float>(numIndices);
            std::fill_n(&weights.first[0], numIndices, 1.0f);
            hostToDeviceCopy(weights, numIndices, true);
        }
        
        //------------------------------------------------------------------------
        // Configure poisson population
        //------------------------------------------------------------------------
        HostDeviceArray<unsigned int> poissonNumSpikes;
        HostDeviceArray<unsigned int> poissonSpikes;
        HostDeviceArray<uint32_t> poissonSpikeBits;

        if(mode == ModeSparsePost) {
            // Create arrays to hold poisson spike count
            poissonNumSpikes = allocateHostDevice<unsigned int>(batchSize);

            // Create arrays to hold poisson spikes
            poissonSpikes = allocateHostDevice<unsigned int>(numPre * batchSize);
        }
        else if(mode == ModeSparsePostBitfield) {
            poissonSpikeBits = allocateHostDevice<uint32_t>(numBatchWords * numPre);
        }
        else {
            assert(false);
        }


        // Create arrays to hold poisson interspike intervals
        // **THINK** why not constant?
        auto poissonMeanISI = allocateHostDevice<float>(numPre * batchSize);
        std::fill_n(&poissonMeanISI.first[0], numPre * batchSize, 1000.0 / (poissonRate * dt));
        hostToDeviceCopy(poissonMeanISI, numPre * batchSize, true);

        // Create device random number generator states for poisson generators
        curandState *d_poissonState = nullptr;
        CHECK_CUDA_ERRORS(cudaMalloc(&d_poissonState, numPre * batchSize * sizeof(curandState)));
        {
            Timer<std::milli> t("Seed poisson:");
            // Initialise these seeds using kernel
            // **NOTE** first numPre sequences used by Poisson spike sources
            initRandomSeed <<<preBatchBlocks, blockSize >>>(numPre * batchSize, numPre * batchSize, d_poissonState);
            cudaDeviceSynchronize();
        }

        // Create device array for poisson generator time to spike
        float *d_poissonTimeToSpike = nullptr;
        CHECK_CUDA_ERRORS(cudaMalloc(&d_poissonTimeToSpike, numPre * batchSize * sizeof(float)));

        // Initialise time to spike using kernel
        {
            Timer<std::milli> t("Init poisson TTS:");
            initPoissonTimeToSpike <<<preBatchBlocks, blockSize>>>(numPre * batchSize, poissonMeanISI.second, d_poissonState,
                                                                   d_poissonTimeToSpike);
            cudaDeviceSynchronize();
        }
        const float connectionProbabilityReciprocal = 1.0f / std::log(1.0f - connectionProbability);
    
        // Create timing events
        cudaEvent_t kernelStartEvent;
        cudaEvent_t kernelEndEvent;
        double kernelTime = 0.0;
        CHECK_CUDA_ERRORS(cudaEventCreate(&kernelStartEvent));
        CHECK_CUDA_ERRORS(cudaEventCreate(&kernelEndEvent));

        {
            // Loop through time
            for (unsigned int t = 0; t < 5000; t++) {
                // Simulate poisson population
                {
                    const dim3 threads(blockSize, 1);
                    const dim3 grid(preBlocks, batchSize);

                    if(mode == ModeSparsePost) {
                        // Zero spike count
                        std::fill_n(&poissonNumSpikes.first[0], batchSize, 0);
                        hostToDeviceCopy(poissonNumSpikes, batchSize);

                        const unsigned int sharedBytes = blockSize * sizeof(unsigned int);
                        poisson <<<grid, threads, sharedBytes>>>(numPre, poissonMeanISI.second, d_poissonState,
                                                                 d_poissonTimeToSpike, poissonNumSpikes.second, poissonSpikes.second);
                    }
                    else if(mode == ModeSparsePostBitfield) {
                        // Clear spike bits
                        std::fill_n(&poissonSpikeBits.first[0], numBatchWords * numPre, 0);
                        hostToDeviceCopy(poissonSpikeBits, numBatchWords * numPre);

                        poissonBitfield<<<grid, threads>>>(numPre, poissonMeanISI.second, d_poissonState,
                                                           d_poissonTimeToSpike, poissonSpikeBits.second);
                    }

                }
                CHECK_CUDA_ERRORS(cudaEventRecord(kernelStartEvent));
                const unsigned int numPostSynapseBlocks = ceilDivide(sparseMatrix.maxRowLength, blockSize);
                if(mode == ModeSparsePost) {
                    const dim3 threads(blockSize, 1);
                    const dim3 grid(numPostSynapseBlocks, batchSize);
                    const unsigned int sharedBytes = blockSize * 2 * sizeof(unsigned int);
                    sparsePostIndividualWeight<<<grid, threads, sharedBytes>>>(sparseMatrix.rowLength.second, sparseMatrix.indices.second, 
                                                                               numPre, numPost, sparseMatrix.maxRowLength,
                                                                               poissonNumSpikes.second, poissonSpikes.second,
                                                                               weights.second, outCurrents.second);
                }
                else if(mode == ModeSparsePostBitfield) {
                    const dim3 threads(blockSize, 1);
                    const dim3 grid(numPostSynapseBlocks, ceilDivide(numPre, 32));
                    sparsePostBitfieldIndividualWeight<<<grid, threads>>>(sparseMatrix.rowLength.second, sparseMatrix.indices.second, 
                                                                          batchSize, numPre, numPost, sparseMatrix.maxRowLength,
                                                                          poissonSpikeBits.second, weights.second, outCurrents.second);
                }

                CHECK_CUDA_ERRORS(cudaEventRecord(kernelEndEvent));
                CHECK_CUDA_ERRORS(cudaEventSynchronize(kernelEndEvent));

                float tmp;
                CHECK_CUDA_ERRORS(cudaEventElapsedTime(&tmp, kernelStartEvent, kernelEndEvent));
                kernelTime += tmp;
            }
        }

        std::cout << "Kernel time:" << kernelTime << " ms" << std::endl;

        deviceToHostCopy(outCurrents, numPost * batchSize);
        float meanCurrent = std::accumulate(&outCurrents.first[0], &outCurrents.first[numPost * batchSize], 0.0f) / (numPost * batchSize);
        std::cout << "Mean current:" << meanCurrent << ", estimated mean current:" << numPre * poissonRate * 5.0 * connectionProbability << std::endl;
    }
    catch(std::exception &ex)
    {
        std::cerr << ex.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

