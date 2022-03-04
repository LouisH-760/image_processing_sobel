#include <stdio.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#define WINDOWFLAGS WINDOW_NORMAL|WINDOW_KEEPRATIO|WINDOW_GUI_EXPANDED
#define WNAME "Sobel"
#define OUTNAME "result.png"
#define DISPLAY_SCALE 0.2

#define BLOCK_SIZE 32
#define THREAD_WORK 100


using namespace cv;

int showAndSave(Mat sobel)
{
    Mat resized;
    resize(sobel, resized, Size(), DISPLAY_SCALE, DISPLAY_SCALE, INTER_AREA);
    namedWindow(WNAME, WINDOWFLAGS);
    imshow(WNAME, resized);
    imwrite(OUTNAME, sobel);
    return waitKey(0);
}

Mat loadImage(int argc, char** argv) {
    Mat image;
    if (argc != 2) {
        printf("usage: DisplayImage.out <Image_Path>\n");
        return image;
    }
    image = imread( argv[1], IMREAD_GRAYSCALE);
    if (image.empty()) {
        printf("No image data \n");
        return image;
    }
    return image;
}

void matrixToArray(Mat matrix, int* arr, int size) {
    int pos;
    int row = matrix.rows;
    int col = matrix.cols;
    for(int i = 0; i < matrix.rows; i++) {
        for(int j = 0; j < matrix.cols; j++) {
            pos = i * col + j;
            *(arr + pos) = (int) matrix.at<uchar>(i, j);
        }
    }
}

Mat arrayToMatrix(int* arr, int size, int cols, int rows) {
    int pos;
    Mat out(rows, cols, CV_8UC1);
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            pos = i * cols + j;
            out.at<uchar>(i, j) = (unsigned char) *(arr + pos);
        }
    }
    return out;
}

__global__ void sobelNaive(int *img, int *output, int size) {
    int index = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    for(int i = 0; i < THREAD_WORK; i++) {
        if(index + i < size) {
            int val = *(img + index + i);
            *(output + index + i) = val;
        }
    }
}

int main(int argc, char** argv ) {
    Mat orig = loadImage(argc, argv);
    const int dims = orig.cols * orig.rows;
    const int blocks = (int) ceil(dims);
    // with a big image, dims > size_t. Can't use arrays :(
    int *img = (int *) calloc(dims, sizeof(int));
    int *remoteImg, *remoteOutput, *output;
    cudaMalloc(&remoteImg, sizeof(int) * dims);
    cudaMalloc(&remoteOutput, sizeof(int) *  dims);
    // since we're on GPU, we don't want the Mat type but an int array, if possible
    matrixToArray(orig, img, dims);

    cudaMemcpy(remoteImg, img,  dims, cudaMemcpyHostToDevice);
    free(img);

    sobelNaive<<<blocks, BLOCK_SIZE>>>(remoteImg, remoteOutput,  dims);

    cudaDeviceSynchronize();
    cudaFree(remoteImg);
    output = (int *) malloc( dims * (sizeof(int)));
    cudaMemcpy(output, remoteOutput, sizeof(int) *  dims, cudaMemcpyDeviceToHost);

    showAndSave(arrayToMatrix(output, dims, orig.cols, orig.rows));
    
    free(output);
    cudaFree(remoteOutput);
    return 0;
}