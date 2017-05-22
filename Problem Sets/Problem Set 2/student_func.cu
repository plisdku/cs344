// Homework 2
// Image Blurring
//
// In this homework we are blurring an image. To do this, imagine that we have
// a square array of weight values. For each pixel in the image, imagine that we
// overlay this square array of weights on top of the image such that the center
// of the weight array is aligned with the current pixel. To compute a blurred
// pixel value, we multiply each pair of numbers that line up. In other words, we
// multiply each weight with the pixel underneath it. Finally, we add up all of the
// multiplied numbers and assign that value to our output for the current pixel.
// We repeat this process for all the pixels in the image.

// To help get you started, we have included some useful notes here.

//****************************************************************************

// For a color image that has multiple channels, we suggest separating
// the different color channels so that each color is stored contiguously
// instead of being interleaved. This will simplify your code.

// That is instead of RGBARGBARGBARGBA... we suggest transforming to three
// arrays (as in the previous homework we ignore the alpha channel again):
//  1) RRRRRRRR...
//  2) GGGGGGGG...
//  3) BBBBBBBB...
//
// The original layout is known an Array of Structures (AoS) whereas the
// format we are converting to is known as a Structure of Arrays (SoA).

// As a warm-up, we will ask you to write the kernel that performs this
// separation. You should then write the "meat" of the assignment,
// which is the kernel that performs the actual blur. We provide code that
// re-combines your blurred results for each color channel.

//****************************************************************************

// You must fill in the gaussian_blur kernel to perform the blurring of the
// inputChannel, using the array of weights, and put the result in the outputChannel.

// Here is an example of computing a blur, using a weighted average, for a single
// pixel in a small image.
//
// Array of weights:
//
//  0.0  0.2    0.0
//  0.2  0.2    0.2
//  0.0  0.2    0.0
//
// Image (note that we align the array of weights to the center of the box):
//
//      1    2  5    2  0    3
//           -------
//      3 |2    5    1| 6    0           0.0*2 + 0.2*5 + 0.0*1 +
//          |               |
//      4 |3    6    2| 1    4   ->  0.2*3 + 0.2*6 + 0.2*2 +     ->  3.2
//          |               |
//      0 |4    0    3| 4    2           0.0*4 + 0.2*0 + 0.0*3
//           -------
//      9    6  5    0  3    9
//
//               (1)                                                 (2)                                 (3)
//
// A good starting place is to map each thread to a pixel as you have before.
// Then every thread can perform steps 2 and 3 in the diagram above
// completely independently of one another.

// Note that the array of weights is square, so its height is the same as its width.
// We refer to the array of weights as a filter, and we refer to its width with the
// variable filterWidth.

//****************************************************************************

// Your homework submission will be evaluated based on correctness and speed.
// We test each pixel against a reference solution. If any pixel differs by
// more than some small threshold value, the system will tell you that your
// solution is incorrect, and it will let you try again.

// Once you have gotten that working correctly, then you can think about using
// shared memory and having the threads cooperate to achieve better performance.

//****************************************************************************

// Also note that we've supplied a helpful debugging function called checkCudaErrors.
// You should wrap your allocation and copying statements like we've done in the
// code we're supplying you. Here is an example of the unsafe way to allocate
// memory on the GPU:
//
// cudaMalloc(&d_red, sizeof(unsigned char) * numRows * numCols);
//
// Here is an example of the safe way to do the same thing:
//
// checkCudaErrors(cudaMalloc(&d_red, sizeof(unsigned char) * numRows * numCols));
//
// Writing code the safe way requires slightly more typing, but is very helpful for
// catching mistakes. If you write code the unsafe way and you make a mistake, then
// any subsequent kernels won't compute anything, and it will be hard to figure out
// why. Writing code the safe way will inform you as soon as you make a mistake.

// Finally, remember to free the memory you allocate at the end of the function.

//****************************************************************************

#include "utils.h"

__global__
void gaussian_blur(const unsigned char* const inputChannel,
                                     unsigned char* const outputChannel,
                                     int numRows, int numCols,
                                     const float* const filter, const int filterWidth)
{
    int2 thread2d = make_int2(blockIdx.x*blockDim.x + threadIdx.x,
        blockIdx.y*blockDim.y + threadIdx.y);
    
    if (thread2d.x >= numCols || thread2d.y >= numRows)
    {
        return;
    }
    
    int thread1d = thread2d.x + thread2d.y * numCols;
    
    
    float accum = 0.0;
    
    int filterRadius = filterWidth/2;
    for (int xx = -filterRadius; xx <= filterRadius; xx++)
    {
        int xGlobal = thread2d.x + xx;
        if (xGlobal < 0)
            xGlobal = 0;
        if (xGlobal >= numCols)
            xGlobal = numCols-1;
        
        int xFilter = xx + filterRadius;
        for (int yy = -filterRadius; yy <= filterRadius; yy++)
        {
            int yGlobal = thread2d.y + yy;
            if (yGlobal < 0)
                yGlobal = 0;
            if (yGlobal >= numRows)
                yGlobal = numRows-1;
            
            int yFilter = yy + filterRadius;
            
            int img1d = xGlobal + yGlobal*numCols;
            int filter1d = xFilter + yFilter*filterWidth;
            
//            if (yy == 0 && xx == 0)
//                accum = inputChannel[img1d];
            accum += filter[filter1d] * inputChannel[img1d];
        }
    }
    
    outputChannel[thread1d] = accum;
//    outputChannel[thread1d] = inputChannel[thread1d];
    
    
//    if (thread2d.y == 0)
//    {
//        printf("Copy color x = %d (%d to %d)\n",
//            thread2d.x, inputChannel[thread1d], outputChannel[thread1d]);
//    }
}

__global__
void copyKernel(const unsigned char* const inputChannel,
                                     unsigned char* const outputChannel,
                                     int numRows, int numCols,
                                     const float* const filter, const int filterWidth)
{
    // TODO
    
    
    int2 thread2d = make_int2(blockIdx.x*blockDim.x + threadIdx.x,
        blockIdx.y*blockDim.y + threadIdx.y);
    
    if (thread2d.x > numCols || thread2d.y > numRows)
    {
        return;
    }
    
    int thread1d = thread2d.x + thread2d.y * numCols;
    
    
    
    
//    assert(thread1d >= 0);
//    assert(thread1d < numRows*numCols);
    outputChannel[thread1d] = inputChannel[thread1d];
    
    
//    if (thread2d.y == 0)
//    {
//        printf("Copy color x = %d (%d to %d)\n",
//            thread2d.x, inputChannel[thread1d], outputChannel[thread1d]);
//        
//    }

//    if (thread1d >= numRows*numCols)
//    {
//    }

    // NOTE: Be sure to compute any intermediate results in floating point
    // before storing the final result as unsigned char.

    // NOTE: Be careful not to try to access memory that is outside the bounds of
    // the image. You'll want code that performs the following check before accessing
    // GPU memory:
    //
    // if ( absolute_image_position_x >= numCols ||
    //          absolute_image_position_y >= numRows )
    // {
    //       return;
    // }
    
    // NOTE: If a thread's absolute position 2D position is within the image, but some of
    // its neighbors are outside the image, then you will need to be extra careful. Instead
    // of trying to read such a neighbor value from GPU memory (which won't work because
    // the value is out of bounds), you should explicitly clamp the neighbor values you read
    // to be within the bounds of the image. If this is not clear to you, then please refer
    // to sequential reference solution for the exact clamping semantics you should follow.
}

//This kernel takes in an image represented as a uchar4 and splits
//it into three images consisting of only one color channel each
__global__
void separateChannels(const uchar4* const inputImageRGBA,
                                            int numRows,
                                            int numCols,
                                            unsigned char* const redChannel,
                                            unsigned char* const greenChannel,
                                            unsigned char* const blueChannel)
{
    // TODO
    //
    // NOTE: Be careful not to try to access memory that is outside the bounds of
    // the image. You'll want code that performs the following check before accessing
    // GPU memory:
    //
    // if ( absolute_image_position_x >= numCols ||
    //          absolute_image_position_y >= numRows )
    // {
    //       return;
    // }
    
    int2 thread2d = make_int2(blockIdx.x*blockDim.x + threadIdx.x,
        blockIdx.y*blockDim.y + threadIdx.y);
    
    if (thread2d.x >= numCols || thread2d.y >= numRows)
    {
        return;
    }
    
    int thread1d = thread2d.x + thread2d.y * numCols;
    
    
    redChannel[thread1d] = inputImageRGBA[thread1d].x;
    greenChannel[thread1d] = inputImageRGBA[thread1d].y;
    blueChannel[thread1d] = inputImageRGBA[thread1d].z;
    
//    if (thread2d.y == 0)
//    {
////        printf("Color (%d %d %d) at x = %d\n",
////            inputImageRGBA[thread1d].x, inputImageRGBA[thread1d].y, inputImageRGBA[thread1d].z, thread2d.x);
//        printf("Source color x = %d (%d %d %d)\n",
//            thread2d.x, redChannel[thread1d], greenChannel[thread1d], blueChannel[thread1d]);
//        
//    }
}

//This kernel takes in three color channels and recombines them
//into one image.    The alpha channel is set to 255 to represent
//that this image has no transparency.
__global__
void recombineChannels(const unsigned char* const redChannel,
                                             const unsigned char* const greenChannel,
                                             const unsigned char* const blueChannel,
                                             uchar4* const outputImageRGBA,
                                             int numRows,
                                             int numCols)
{
    const int2 thread2d = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
        blockIdx.y * blockDim.y + threadIdx.y);

    const int thread1d = thread2d.y * numCols + thread2d.x;

//    if (thread2d.x == numCols && thread2d.y == numRows)
//    {
//        printf("thread2d = (%d %d)\n", thread2d.x, thread2d.y);
//        printf("blockIdx = (%d %d) blockDim = (%d %d) threadIdx = (%d %d)\n",
//            blockIdx.x, blockIdx.y, blockDim.x, blockDim.y, threadIdx.x, threadIdx.y);
//    }
//    if (thread2d.x == 0 && thread2d.y == 0)
//    {
//        printf("thread2d = (%d %d)\n", thread2d.x, thread2d.y);
//        printf("blockIdx = (%d %d) blockDim = (%d %d) threadIdx = (%d %d)\n",
//            blockIdx.x, blockIdx.y, blockDim.x, blockDim.y, threadIdx.x, threadIdx.y);
//    }

    
    //make sure we don't try and access memory outside the image
    //by having any threads mapped there return early
    if (thread2d.x >= numCols || thread2d.y >= numRows)
    {
        return;
    }
    
//    if (thread2d.y == 0)
//    {
//        printf("Dest color x = %d (%d %d %d)\n",
//            thread2d.x, redChannel[thread1d], greenChannel[thread1d], blueChannel[thread1d]);
//            
////        printf("Dest color (%d %d %d) at x = %d\n",
////            redChannel[thread1d], greenChannel[thread1d], blueChannel[thread1d], thread2d.x);
//    }


    unsigned char red       = redChannel[thread1d];
    unsigned char green = greenChannel[thread1d];
    unsigned char blue  = blueChannel[thread1d];

    //Alpha should be 255 for no transparency
    uchar4 outputPixel = make_uchar4(red, green, blue, 255);

    outputImageRGBA[thread1d] = outputPixel;
}

unsigned char *d_red, *d_green, *d_blue;
float                   *d_filter;

void allocateMemoryAndCopyToGPU(const size_t numRowsImage, const size_t numColsImage,
                                                                const float* const h_filter, const size_t filterWidth)
{

    //allocate memory for the three different channels
    //original
    checkCudaErrors(cudaMalloc(&d_red,   sizeof(unsigned char) * numRowsImage * numColsImage));
    checkCudaErrors(cudaMalloc(&d_green, sizeof(unsigned char) * numRowsImage * numColsImage));
    checkCudaErrors(cudaMalloc(&d_blue,  sizeof(unsigned char) * numRowsImage * numColsImage));
    
    checkCudaErrors(cudaMemset(d_red, 11, sizeof(unsigned char)*numRowsImage * numColsImage));
    checkCudaErrors(cudaMemset(d_green, 22, sizeof(unsigned char)*numRowsImage * numColsImage));
    checkCudaErrors(cudaMemset(d_blue, 33, sizeof(unsigned char)*numRowsImage * numColsImage));

    //TODO:
    //Allocate memory for the filter on the GPU
    //Use the pointer d_filter that we have already declared for you
    //You need to allocate memory for the filter with cudaMalloc
    //be sure to use checkCudaErrors like the above examples to
    //be able to tell if anything goes wrong
    //IMPORTANT: Notice that we pass a pointer to a pointer to cudaMalloc
    
    checkCudaErrors(cudaMalloc(&d_filter, sizeof(float)*filterWidth*filterWidth));

    //TODO:
    //Copy the filter on the host (h_filter) to the memory you just allocated
    //on the GPU.    cudaMemcpy(dst, src, numBytes, cudaMemcpyHostToDevice);
    //Remember to use checkCudaErrors!
    checkCudaErrors(cudaMemcpy(d_filter, h_filter, sizeof(float)*filterWidth*filterWidth, cudaMemcpyHostToDevice));
}

// preconditions:
// all the input arrays are created and initialized somehow (e.g. to zero)
void your_gaussian_blur(const uchar4 * const h_inputImageRGBA, uchar4 * const d_inputImageRGBA,
                                                uchar4* const d_outputImageRGBA, const size_t numRows, const size_t numCols,
                                                unsigned char *d_redBlurred, 
                                                unsigned char *d_greenBlurred, 
                                                unsigned char *d_blueBlurred,
                                                const int filterWidth)
{
    int numPixels = numRows * numCols;
    
//    printf("Image size is %d rows x %d cols\n", numRows, numCols);
//    printf("Num pixels %d\n", numPixels);
//    printf("Pointers: r %d g %d b %d\n", d_red, d_green, d_blue);
//    printf("Pointers blurred: r %d g %d b %d\n", d_redBlurred, d_greenBlurred, d_blueBlurred);
    
    //TODO: Set reasonable block size (i.e., number of threads per block)
    const dim3 blockSize(32,32,1);

    //TODO:
    //Compute correct grid size (i.e., number of blocks per kernel launch)
    //from the image size and and block size.
    
    // I'll overshoot but not undershoot, here.
    // Let's swap from row/col to y/x (IMAGE to CARTESIAN).  Seems like what
    // they ask for.
    const dim3 gridSize((numCols+blockSize.x-1)/blockSize.x,
        (numRows+blockSize.y-1)/blockSize.y,
        1);
    
    // I suppose I can make double use of the RGB channels but I'd have to change
    // the function signature for my convolution kernel.  How about instead
    // I create some arrays on the GPU.
    
//    unsigned char *d_inputR, *d_inputG, *d_inputB;
//    checkCudaErrors(cudaMalloc(&d_inputR, sizeof(char)*numPixels));
//    checkCudaErrors(cudaMalloc(&d_inputG, sizeof(char)*numPixels));
//    checkCudaErrors(cudaMalloc(&d_inputB, sizeof(char)*numPixels));

    //TODO: Launch a kernel for separating the RGBA image into different color channels
    separateChannels<<<gridSize, blockSize>>>(d_inputImageRGBA, numRows, numCols,
        d_red, d_green, d_blue);

    // Call cudaDeviceSynchronize(), then call checkCudaErrors() immediately after
    // launching your kernel to make sure that you didn't make any mistakes.
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    //TODO: Call your convolution kernel here 3 times, once for each color channel.
    
    gaussian_blur<<<gridSize, blockSize>>>(d_red, d_redBlurred, numRows, numCols, d_filter, filterWidth);
    gaussian_blur<<<gridSize, blockSize>>>(d_green, d_greenBlurred, numRows, numCols, d_filter, filterWidth);
    gaussian_blur<<<gridSize, blockSize>>>(d_blue, d_blueBlurred, numRows, numCols, d_filter, filterWidth);

    // Again, call cudaDeviceSynchronize(), then call checkCudaErrors() immediately after
    // launching your kernel to make sure that you didn't make any mistakes.
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    // Now we recombine your results. We take care of launching this kernel for you.
    //
    // NOTE: This kernel launch depends on the gridSize and blockSize variables,
    // which you must set yourself.
    recombineChannels<<<gridSize, blockSize>>>(
        d_redBlurred,
        d_greenBlurred,
        d_blueBlurred,
        d_outputImageRGBA,
        numRows,
        numCols);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

}


//Free all the memory that we allocated
//TODO: make sure you free any arrays that you allocated
void cleanup() {
    checkCudaErrors(cudaFree(d_red));
    checkCudaErrors(cudaFree(d_green));
    checkCudaErrors(cudaFree(d_blue));
}
