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
    for(int i = 0; i < size; i++) {
        *(arr + i) = (int) matrix.at<uchar>(i%matrix.cols, (int) round(i/matrix.cols));
    }
}

Mat arrayToMatrix(int* arr, int size, int w, int h) {
    Mat out(w, h, CV_8UC1);
    for(int i = 0; i < size; i++) {
        out.at<uchar>(i%w, (int) round(i/w)) = (unsigned char) *(arr + i);
    }
    return out;
}

__global__ void sobelNaive(int *img, int *output, int size) {
    int index = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if(index < size) {
        int val = *(img + index);
        *(output + index) = val;
    }
}

int main(int argc, char** argv ) {
    printf("begin");
    Mat orig = loadImage(argc, argv);
    printf("loadimg");
    const int dims = orig.cols * orig.rows;
    const int blocks = (int) ceil(dims / BLOCK_SIZE);
    // with a big image, dims > size_t. Can't use arrays :(
    int *img = (int *) malloc(dims * sizeof(int));
    int *remoteImg, *remoteOutput, *output;
    printf("vars");
    cudaMalloc(&remoteImg, sizeof(int) * orig.total());
    cudaMalloc(&remoteOutput, sizeof(int) *  orig.total());
    printf("allocs");
    // since we're on GPU, we don't want the Mat type but an int array, if possible
    matrixToArray(orig, img,  orig.total());
    printf("init");

    cudaMemcpy(remoteImg, img,  orig.total(), cudaMemcpyHostToDevice);
    free(img);

    sobelNaive<<<blocks, BLOCK_SIZE>>>(remoteImg, remoteOutput,  orig.total());

    cudaDeviceSynchronize();
    cudaFree(remoteImg);
    output = (int *) malloc( dims * (sizeof(int)));
    cudaMemcpy(output, remoteOutput, sizeof(int) *  orig.total(), cudaMemcpyDeviceToHost);

    showAndSave(arrayToMatrix(output, orig.total(), orig.rows, orig.cols));
    
    free(output);
    cudaFree(remoteOutput);
    return 0;
}