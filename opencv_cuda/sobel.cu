#include <stdio.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#define WINDOWFLAGS WINDOW_NORMAL|WINDOW_KEEPRATIO|WINDOW_GUI_EXPANDED
#define WNAME "Sobel"
#define OUTNAME "result.png"
#define DISPLAY_SCALE 0.2


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
    for(int i = 0; i < matrix.rows; i++) {
        for(int j = 0; j < matrix.cols; j++) {
            if(i*j < size - 1) {
                *(arr + i*j) = (int) matrix.at<uchar>(i, j);
            }
        }
    }
}

int main(int argc, char** argv ) {
    Mat orig = loadImage(argc, argv);
    const int dims = orig.rows * orig.cols;
    // with a big image, dims > size_t. Can't use arrays :(
    int *img = (int *) malloc(dims * (sizeof(int)));
    // since we're on GPU, we don't want the Mat type but an int array, if possible
    matrixToArray(orig, img, dims);
    free(img);
    return 0;
}