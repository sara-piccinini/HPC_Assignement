#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <chrono>
#include <cuda_runtime.h>
#include <string> 
#include <fstream>
#ifndef TILE_DIM
#define TILE_DIM 16  // Block Thread Dimension
#endif
#define HALO_SIZE 1  // Halo size, how much columns/rows of pixels to load around the tile 
                     // 1 because the filter is 3x3


/*----------------------------------------------------------------------------------------------------------------------------------------

                                             Function to get the image in order to visualize it 
                                                                
------------------------------------------------------------------------------------------------------------------------------------------*/

cv::Mat createG_x_y_Matrix(int channelId, float* gxy, int C, int R){
    
    cv::Mat gxy_cpu, tempMat;
    cv::Mat gxy_normalized, gxy_8U;
    std::vector<float> temp(3 * R * C);

    //copy all the channels from device to host
    cudaMemcpy(temp.data(), gxy, 3 * R * C * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Choose the right channel from the temp vector
    tempMat = cv::Mat(R, C, CV_32F, temp.data() + channelId  * R * C);
    gxy_cpu = tempMat.clone();

    // Normalize the matrix to the range [0, 255] and convert to CV_8U for visualization
    cv::normalize(gxy_cpu, gxy_normalized, 0, 255, cv::NORM_MINMAX);
    gxy_normalized.convertTo(gxy_8U, CV_8U);
    
    return gxy_8U;
}

/* ----------------------------------------------------------------------------------------------------------------------------------------

                                                                CUDA Kernel for gxy calculation

------------------------------------------------------------------------------------------------------------------------------------------*/

__global__ void g_x_y_calculation(float *channel, float *gxy, int R, int CO){

    // shared tile memory declaration 
    __shared__ float tile[TILE_DIM + 2 * HALO_SIZE][TILE_DIM + 2 * HALO_SIZE]; //2 * HALO_SIZE, 1 for right/up and the other for left/down

    // index calculation
    int tx = threadIdx.x;
    int ty = threadIdx.y; 
    int z = blockIdx.z;

    // Global output and input coordinates for the pixel taken into account
    //x_out is the place where the thread will write the result
    int x_out = blockIdx.x * TILE_DIM + tx;
    int y_out = blockIdx.y * TILE_DIM + ty;

    //x_in is the place where the thread will read the data from
    //Both initialized to the center pixel
    int x_in = x_out;
    int y_in = y_out;

    // Load the tile with the data from the channel
    if (x_in < CO && y_in < R) {
        tile[ty + HALO_SIZE][tx + HALO_SIZE] = channel[z * (R * CO) + y_in * CO + x_in];
    } else {
        tile[ty + HALO_SIZE][tx + HALO_SIZE] = 0.0f;
    }

    if (tx < HALO_SIZE) { //Only these specifics threads (close to the start/end) will load the left and right halo
        // Load the left halo
        x_in = x_out - HALO_SIZE;
        if (x_in >= 0 && y_in < R) {
            tile[ty + HALO_SIZE][tx] = channel[z * (R * CO) + y_in * CO + x_in];
        } else {
            tile[ty + HALO_SIZE][tx] = 0.0f;
        }
        // Load right Halo
        x_in = x_out + TILE_DIM;
        if (x_in < CO && y_in < R) {
            tile[ty + HALO_SIZE][tx + TILE_DIM + HALO_SIZE] = channel[z * (R * CO) + y_in * CO + x_in];
        } else {
            tile[ty + HALO_SIZE][tx + TILE_DIM + HALO_SIZE] = 0.0f;
        }
    }


    if (ty < HALO_SIZE) { //Only these specifics threads (close to the start/end) will load the upper and lower halo
        // Load upper halo
        y_in = y_out - HALO_SIZE;
        if (y_in >= 0 && x_out < CO) {
            tile[ty][tx + HALO_SIZE] = channel[z * (R * CO) + y_in * CO + x_out];
        } else {
            tile[ty][tx + HALO_SIZE] = 0.0f;
        }
        // Load lower Halo
        y_in = y_out + TILE_DIM;
        if (y_in < R && x_out < CO) {
            tile[ty + TILE_DIM + HALO_SIZE][tx + HALO_SIZE] = channel[z * (R * CO) + y_in * CO + x_out];
        } else {
            tile[ty + TILE_DIM + HALO_SIZE][tx + HALO_SIZE] = 0.0f;
        }
    }

    // Load corners
    if (tx < HALO_SIZE && ty < HALO_SIZE) {
        // up left
        x_in = x_out - HALO_SIZE;
        y_in = y_out - HALO_SIZE;
        if (x_in >= 0 && y_in >= 0) 
            tile[ty][tx] = channel[z * (R * CO) + y_in * CO + x_in]; 
        else 
            tile[ty][tx] = 0.0f;

        // up right
        x_in = x_out + TILE_DIM;
        y_in = y_out - HALO_SIZE;
        if (x_in < CO && y_in >= 0) 
            tile[ty][tx + TILE_DIM + HALO_SIZE] = channel[z * (R * CO) + y_in * CO + x_in]; 
        else 
            tile[ty][tx + TILE_DIM + HALO_SIZE] = 0.0f;

        // low left
        x_in = x_out - HALO_SIZE;
        y_in = y_out + TILE_DIM;
        if (x_in >= 0 && y_in < R) 
            tile[ty + TILE_DIM + HALO_SIZE][tx] = channel[z * (R * CO) + y_in * CO + x_in]; 
        else 
            tile[ty + TILE_DIM + HALO_SIZE][tx] = 0.0f;

        // low right
        x_in = x_out + TILE_DIM;
        y_in = y_out + TILE_DIM;
        if (x_in < CO && y_in < R) 
            tile[ty + TILE_DIM + HALO_SIZE][tx + TILE_DIM + HALO_SIZE] = channel[z * (R * CO) + y_in * CO + x_in]; 
        else 
            tile[ty + TILE_DIM + HALO_SIZE][tx + TILE_DIM + HALO_SIZE] = 0.0f;
    }

    //Thread sync to be sure all threads have written to the tile
    __syncthreads();

    // Check if the output coordinates are within bounds
    if (x_out >= CO || y_out >= R) return;
    
    // Actual fitler application
    if (x_out < CO && y_out < R) {
        // At border it is 0 by default
        if (x_out == 0 || x_out >= CO - 1 || y_out == 0 || y_out >= R - 1) {
            gxy[z * (R * CO) + y_out * CO + x_out] = 0.0f;
        } else {
            int W[3][3] = {{1, 2, 1}, {3, 4, 3}, {1, 2, 1}};
            float sum = 0.0f;

            #pragma unroll // Reduce overhead of loop control
            for (int i = 0; i < 3; i++) {
                #pragma unroll
                for (int j = 0; j < 3; j++) {
                    sum += (1.0f / 16.0f) * W[i][j] * tile[ty + i][tx + j];
                }
            }
            gxy[z * (R * CO) + y_out * CO + x_out] = sum;
        }
    }
}


/* ----------------------------------------------------------------------------------------------------------------------------------------------
    
                                                                Function to process the image

    ------------------------------------------------------------------------------------------------------------------------------------------------*/



cv::Mat GetResult(std::string imagePath, std::ofstream &logFile) {
    /* ----------------------------------------------------------------------------------------------------------------------------------------------
    
                                                                        OpenCV Setup

    ------------------------------------------------------------------------------------------------------------------------------------------------*/
    
    //Read the Image
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);

    // Decomposition of the image into its RGB channels
    if(image.empty()) {
        std::cerr << "Error: Could not open or find the image!" << std::endl;
        exit(EXIT_FAILURE);
    }  
    //image size
    logFile << std::to_string(image.rows) << ";" << std::to_string(image.cols) << ";";

    std::vector<cv::Mat> channels;
    cv::split(image, channels);


    /* ----------------------------------------------------------------------------------------------------------------------------------------------
    
                                                                        CUDA Setup

    ------------------------------------------------------------------------------------------------------------------------------------------------*/
    // start the timer
    auto start = std::chrono::high_resolution_clock::now();
    // get the number of threads from the environment variable
    //std::cout << "Thread used: " << TILE_DIM * TILE_DIM << std::endl;
    logFile << TILE_DIM * TILE_DIM << ";";
    // instatiate cv matrix from which you will get the data to be stored in CUDA memory
    std::vector<cv::Mat> ch32(3);
    std::vector<float> channel_host(3 * channels[0].rows * channels[0].cols);    

    // Convert the channels to CV_32F for CUDA processing and Flat the channels into a single vector
    for (int i = 0; i < 3; i++) {        
        // create a matrix to hold the channel data converted to CV_32F
        channels[i].convertTo(ch32[i], CV_32F);
        std::memcpy(
            channel_host.data() + i * channels[0].rows * channels[0].cols,
            ch32[i].ptr<float>(),
            channels[0].rows * channels[0].cols * sizeof(float)
        );
    }
    // Initialize the gxy vector with zeros
    std::vector<float> gxy_channels(3 * channels[0].rows * channels[0].cols, 0.0f);
    
    // Allocate memory for the channels (data as input) and gxy (data processed) on the device
    float *channel, *gxy;

    // Define the number of threads per block and the number of blocks
    dim3 threadsPerBlock(TILE_DIM, TILE_DIM);
    dim3 numBlocks(
        (channels[0].cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (channels[0].rows + threadsPerBlock.y - 1) / threadsPerBlock.y,
        3 // 3 channels (R, G, B)
    );
    
    // Allocate memory on the device for the channels and gxy
    cudaMalloc(&channel, 3 * channels[0].rows * channels[0].cols * sizeof(float));
    cudaMalloc(&gxy, 3 * channels[0].rows * channels[0].cols * sizeof(float));

    // copy the data from the image
    cudaMemcpy(channel, channel_host.data(), 3 * channels[0].rows * channels[0].cols * sizeof(float), cudaMemcpyHostToDevice);
    // copy the all 0s matrix that host the processed data
    cudaMemcpy(gxy, gxy_channels.data(), 3 * channels[0].rows * channels[0].cols * sizeof(float), cudaMemcpyHostToDevice);
    //time measurement
    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);

    // Launch the kernel to calculate gxy for each channel
    cudaEventRecord(start_event);
    g_x_y_calculation<<<numBlocks, threadsPerBlock>>>(channel, gxy, channels[0].rows, channels[0].cols);
    cudaEventRecord(stop_event);
    cudaEventSynchronize(stop_event);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start_event, stop_event);
    logFile << std::to_string(milliseconds) << ";";
    //std::cout << "Kernel execution time: " << milliseconds << " ms" << std::endl;

    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);



    /*-----------------------------------------------------------------------------------------------------------------------------------------------
                                                                        
                                                                    Display and Save Results

    ------------------------------------------------------------------------------------------------------------------------------------------------*/

    /* Local debugging
    cv::imshow("Red Channel", channels[2]);
    cv::imshow("Green Channel", channels[1]);
    cv::imshow("Blue Channel", channels[0]);
    cv::waitKey(0);
    */


    //                                                     Create the gxy matrices for each channel
    
    //Red Channel
    //std::cout << "Red channel " << std::endl;
    cv::Mat Redgxy = createG_x_y_Matrix(2, gxy, channels[2].cols, channels[2].rows);

    //Green Channel
    //std::cout << "Green channel " << std::endl;
    cv::Mat Greengxy = createG_x_y_Matrix(1, gxy, channels[1].cols, channels[1].rows);
    
    //Blue Channel
    //std::cout << "Blue channel " << std::endl;
    cv::Mat Bluegxy = createG_x_y_Matrix(0, gxy, channels[0].cols, channels[0].rows);
      
    // Cuda free the memory
    cudaFree(channel);
    cudaFree(gxy);

    //end the timer
    auto end = std::chrono::high_resolution_clock::now();
    // Calculate the elapsed time in milliseconds
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    //std::cout << "Elapsed time: " << elapsed << " ms" << std::endl;
    logFile << std::to_string(elapsed) << "\n";
    //Recombine the image
    cv::Mat gxyResult;
    cv::merge(std::vector<cv::Mat>{Bluegxy, Greengxy, Redgxy}, gxyResult);
    
    return gxyResult;
}

/* ----------------------------------------------------------------------------------------------------------------------------------------------
    
                                                                    Main Function

    ------------------------------------------------------------------------------------------------------------------------------------------------*/

int main(int argc, char** argv) {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    std::ofstream logFile("./logData/logFile", std::ios::app);
    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
/*
        ----------------- Local debugging -----------------

        std::cout << "Device " << i << ": " << prop.name << std::endl;
        std::cout << "  Multiprocessors: " << prop.multiProcessorCount << std::endl;
        std::cout << "  Max thread per block: " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "  Max thread per multiprocessor: " << prop.maxThreadsPerMultiProcessor << std::endl;
        std::cout << "  Max dim for block: "
                  << prop.maxThreadsDim[0] << " x "
                  << prop.maxThreadsDim[1] << " x "
                  << prop.maxThreadsDim[2] << std::endl;
        std::cout << "  Max dim for grid: "
                  << prop.maxGridSize[0] << " x "
                  << prop.maxGridSize[1] << " x "
                  << prop.maxGridSize[2] << std::endl;
        std::cout << "-------------------------------\n";
*/
    }


    
        logFile << argv[1] <<";";
        //std::cout << "Processing image: " << argv[1] << std::endl;
        cv::Mat result = GetResult(argv[1], logFile);
        //std::cout << "Saving result to: " << argv[2] << std::endl;        
        if (argc > 3 && argv[3] == std::string("-s")) {
            cv::imwrite(argv[2], result);
            std::cout << "Result saved successfully image"<< argv[1] << std::endl;
        }

        logFile <<"\n";
    
    return 0;
}
