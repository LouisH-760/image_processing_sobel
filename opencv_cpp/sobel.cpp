#include <stdio.h>
#include <sys/time.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#define WINDOWFLAGS WINDOW_NORMAL|WINDOW_KEEPRATIO|WINDOW_GUI_EXPANDED
#define WNAME "Sobel"
#define OUTNAME "result.png"
#define DISPLAY_SCALE 0.4

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

Mat sobel(Mat orig, double* time) {
    struct timespec start, end;
    clock_gettime(CLOCK_REALTIME, &start);
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
    Mat out(orig.rows, orig.cols, CV_8UC1);
    unsigned short int result, mx, my;
    int ver, hor;
    for(auto row = 1; row < orig.rows - 1; row++) {
        for(auto col = 1; col < orig.cols - 1; col++) {
            ver = 0;
            hor = 0;
            for(auto x = 0; x < 3; x++) {
                for(auto y = 0; y < 3; y++) {
                    mx = col - 1 + x;
                    my = row - 1 + y;
                    ver += xsob[x][y] * (int) orig.at<uchar>(my, mx);
                    hor += ysob[x][y] * (int) orig.at<uchar>(my, mx);
                }
            }
            result = (unsigned short int) round(sqrt(ver * ver + hor * hor));
            out.at<uchar>(row, col) = (uchar) result;
        }
    }
    clock_gettime(CLOCK_REALTIME, &end);
    *(time) = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_nsec - start.tv_nsec) / 1000;
    return out;
}

int main(int argc, char** argv ) {
    Mat orig = loadImage(argc, argv);
    double time;
    Mat out = sobel(orig, &time);
    double ms = time / 1000;
    printf("Sobel function exec time: %f microseconds (%f milliseconds)\n", time, ms);
    showAndSave(out);
    return 0;
}