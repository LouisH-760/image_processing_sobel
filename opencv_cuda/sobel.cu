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

Mat loadimage(int argc, char** argv) {
    Mat image;
    if ( argc != 2 )
    {
        printf("usage: DisplayImage.out <Image_Path>\n");
        return image;
    }
    image = imread( argv[1], IMREAD_GRAYSCALE);
    if ( !image.data )
    {
        printf("No image data \n");
        return image;
    }
    return image;
}

int main(int argc, char** argv ) {
    showAndSave(loadimage(argc, argv));
    return 0;
}