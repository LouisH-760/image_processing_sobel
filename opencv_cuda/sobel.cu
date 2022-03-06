#include <stdio.h>
#include <sys/time.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#define WINDOWFLAGS WINDOW_NORMAL|WINDOW_KEEPRATIO|WINDOW_GUI_EXPANDED
#define WNAME "Sobel"
#define OUTNAME "result.png"
#define DISPLAY_SCALE 0.3

#define BLOCKS 32000
#define THREADS 1024


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

void matrixToArray(Mat matrix, uchar* arr, unsigned short int cols, unsigned short int rows) {
    for(auto row = 0; row < rows; row++) {
        for(auto col = 0; col < cols; col++) {
            arr[row*cols+col] = matrix.at<uchar>(row, col);
        }
    }
}

Mat arrayToMatrix(uchar* arr, unsigned short int cols, unsigned short int rows) {
    Mat out(rows, cols, CV_8UC1);
    for(auto row = 0; row < rows; row++) {
        for(auto col = 0; col < cols; col++) {
            out.at<uchar>(row, col) = arr[row*cols+col];
        }
    }
    return out;
}

__global__ void sobelNaive(uchar *img, uchar *output, unsigned short int cols, unsigned short int rows) {
    int index = blockIdx.x * THREADS + threadIdx.x;
    int currCol = index % cols;
    int currRow = (int) truncf(index / cols);
    bool usable = (currCol % (cols - 1)) != 0 && (currRow % (rows - 1)) != 0;
    int xsob[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };

    int ysob[3][3] = {
        {-1, -2, -1},
        {0, 0, 0},
        {1, 2, 1}
    };
    if(usable) {
        int x = 0;
        int y = 0;
        for(auto i = -1; i < 2; i++) {
            for(auto j = -1; j < 2; j++) {
                x += xsob[i + 1][j + 1] * img[(i + currRow) * cols + (j + currCol)];
                y += ysob[i + 1][j + 1] * img[(i + currRow) * cols + (j + currCol)];
            }
        }
        output[index] = (int) roundf(sqrtf(x*x + y*y));
    }
}

int main(int argc, char** argv ) {
    struct timeval start, end;
    Mat orig = loadImage(argc, argv);
    // don't need more than an unsigned short int, assuming realistic images
    // max image size: 65535 * 65535
    const unsigned short int cols = orig.cols;
    const unsigned short int rows = orig.rows;
    const unsigned int size = cols*rows*sizeof(char);
    const unsigned int blocks = (unsigned int) ceil((cols * rows) / THREADS);
    uchar *rImage, *rOutput;

    auto image = (uchar *) malloc(size);

    matrixToArray(orig, image, cols, rows);

    orig.release();

    cudaMalloc(&rImage, size);
    cudaMalloc(&rOutput, size);

    cudaMemcpy(rImage, image, size, cudaMemcpyHostToDevice);

    gettimeofday(&start, NULL);
    sobelNaive<<<blocks, THREADS>>>(rImage, rOutput, cols, rows);
    cudaDeviceSynchronize();
    gettimeofday(&end, NULL);
    double time = (end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) / 1000;
    printf("Sobel function exec time: %f milliseconds\n", time);
    cudaFree(rImage);

    cudaMemcpy(image, rOutput, size, cudaMemcpyDeviceToHost);

    cudaFree(rOutput);

    showAndSave(arrayToMatrix(image, cols, rows));

    free(image);
    return 0;
}