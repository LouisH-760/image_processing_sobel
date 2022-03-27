#include <stdio.h>
#include <omp.h>
#include <sys/time.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <omp.h>

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

int main(int argc, char** argv ) {
    struct timespec start, end;

    Mat orig = loadImage(argc, argv);
    Mat out(orig.rows, orig.cols, CV_8UC1);
    int ver, hor;

    clock_gettime(CLOCK_REALTIME, &start);
    //
    // , (int) orig.rows*orig.cols/4
    // schedule(dynamic, (orig.rows-1)/4)
    #pragma omp parallel for private(ver, hor) schedule(guided)
    for(auto row = 1; row < orig.rows - 1; row++) {
        for(auto col = 1; col < orig.cols - 1; col++) {
            ver = -1 * (int) orig.at<uchar>(row-1, col-1)
                +  1 * (int) orig.at<uchar>(row+1, col-1)
                + -2 * (int) orig.at<uchar>(row-1, col)
                +  2 * (int) orig.at<uchar>(row+1, col)
                + -1 * (int) orig.at<uchar>(row-1, col+1)
                +  1 * (int) orig.at<uchar>(row+1, col+1);
            hor = -1 * (int) orig.at<uchar>(row-1, col-1)
                + -2 * (int) orig.at<uchar>(row, col-1)
                + -1 * (int) orig.at<uchar>(row+1, col-1)
                +  1 * (int) orig.at<uchar>(row-1, col+1)
                +  2 * (int) orig.at<uchar>(row, col+1)
                +  1 * (int) orig.at<uchar>(row+1, col+1);
            out.at<uchar>(row, col) = (uchar) round(sqrt(ver * ver + hor * hor));
        }
    }
    //
    clock_gettime(CLOCK_REALTIME, &end);
    double time = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_nsec - start.tv_nsec) / 1000;
    double ms = time / 1000;
    printf("Sobel function exec time: %f microseconds (%f milliseconds)\n", time, ms);
    //
    showAndSave(out);
    return 0;
}