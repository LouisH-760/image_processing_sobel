#include <stdio.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#define WINDOWFLAGS WINDOW_NORMAL|WINDOW_KEEPRATIO|WINDOW_GUI_EXPANDED
#define WNAME "Sobel"
#define OUTNAME "result.png"
#define DISPLAY_SCALE 0.2

#define BLOCK_SIZE 1
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

__global__ void sobelNaive(int *img, int *output, int size, int offset) {
    int index = (blockIdx.x * BLOCK_SIZE + threadIdx.x) + offset;
    if(index < size) {
        int val = *(img + index);
        *(output + index) = val;
    }
}

int main(int argc, char** argv ) {
    Mat orig = loadImage(argc, argv);
    // don't need more than an unsigned short int for now because bracket syntax is used for arrays
    // max image size: 65535 * 65535
    const unsigned short int cols = orig.cols;
    const unsigned short int rows = orig.rows;
    uchar (*image) = (uchar (*)) calloc(cols*rows, sizeof(uchar));
    matrixToArray(orig, image, cols, rows);
    showAndSave(arrayToMatrix(image, cols, rows));
    free(image);
    return 0;
}