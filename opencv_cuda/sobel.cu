#include <stdio.h>
#include <sys/time.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#define WINDOWFLAGS WINDOW_NORMAL | WINDOW_KEEPRATIO | WINDOW_GUI_EXPANDED
#define WNAME "Sobel"
#define OUTNAME "result.png"
#define DISPLAY_SCALE 0.3

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

Mat loadImage(int argc, char **argv)
{
    Mat image;
    if (argc != 2)
    {
        printf("usage: DisplayImage.out <Image_Path>\n");
        return image;
    }
    image = imread(argv[1], IMREAD_GRAYSCALE);
    if (image.empty())
    {
        printf("No image data \n");
        return image;
    }
    return image;
}

void matrixToArray(Mat matrix, uchar *arr, unsigned short int cols, unsigned short int rows)
{
    for (auto row = 0; row < rows; row++)
    {
        for (auto col = 0; col < cols; col++)
        {
            arr[row * cols + col] = matrix.at<uchar>(row, col);
        }
    }
}

Mat arrayToMatrix(uchar *arr, unsigned short int cols, unsigned short int rows)
{
    Mat out(rows, cols, CV_8UC1);
    for (auto row = 0; row < rows; row++)
    {
        for (auto col = 0; col < cols; col++)
        {
            out.at<uchar>(row, col) = arr[row * cols + col];
        }
    }
    return out;
}

__global__ void sobelNaive(uchar *img, uchar *output, unsigned short int cols, unsigned short int rows)
{
    int index = blockIdx.x * THREADS + threadIdx.x;
    int currCol = index % cols;
    int currRow = (int)truncf(index / cols);
    bool usable = (currCol % (cols - 1)) != 0 && (currRow % (rows - 1)) != 0;
    int xsob[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}};

    int ysob[3][3] = {
        {-1, -2, -1},
        {0, 0, 0},
        {1, 2, 1}};
    if (usable)
    {
        int x = 0;
        int y = 0;
        for (auto i = -1; i < 2; i++)
        {
            for (auto j = -1; j < 2; j++)
            {
                x += xsob[i + 1][j + 1] * img[(i + currRow) * cols + (j + currCol)];
                y += ysob[i + 1][j + 1] * img[(i + currRow) * cols + (j + currCol)];
            }
        }
        output[index] = (int)roundf(sqrtf(x * x + y * y));
    }
}

__global__ void sobelLessNaive(uchar *img, uchar *output, unsigned short int cols, unsigned short int rows)
{
    int index = blockIdx.x * THREADS + threadIdx.x;
    int currCol = index % cols;
    int currRow = (int)truncf(index / cols);
    bool usable = (currCol % (cols - 1)) != 0 && (currRow % (rows - 1)) != 0;
    if (usable)
    {
        int x = 0;
        int y = 0;
        x += -1 * img[(-1 + currRow) * cols + (-1 + currCol)] +
        1 * img[(-1 + currRow) * cols + (1 + currCol)] +
        -2 * img[(currRow) * cols + (-1 + currCol)] +
        2 * img[(currRow) * cols + (1 + currCol)] +
        -1 * img[(1 + currRow) * cols + (-1 + currCol)] +
        1 * img[(1 + currRow) * cols + (1 + currCol)];

        y += -1 * img[(-1 + currRow) * cols + (-1 + currCol)] +
        -2 * img[(-1 + currRow) * cols + (currCol)] +
        -1 * img[(-1 + currRow) * cols + (1 + currCol)] +
        1 * img[(1 + currRow) * cols + (-1 + currCol)] +
        2 * img[(1 + currRow) * cols + (currCol)] +
        1 * img[(1 + currRow) * cols + (1 + currCol)];
        output[index] = (int)roundf(sqrtf(x * x + y * y));
    }
}

__global__ void sobelNaiveCache(uchar *img, uchar *output, unsigned short int cols, unsigned short int rows)
{
    int index = blockIdx.x * THREADS + threadIdx.x;
    int currCol = index % cols;
    int currRow = (int)truncf(index / cols);
    bool usable = (currCol % (cols - 1)) != 0 && (currRow % (rows - 1)) != 0;
    if (usable)
    {

        uchar commonpixels[] = {
            img[(-1 + currRow) * cols + (-1 + currCol)], // 0
            img[(-1 + currRow) * cols + (currCol)], // 1
            img[(-1 + currRow) * cols + (1 + currCol)], // 2
            img[(currRow) * cols + (-1 + currCol)], // 3
            img[(currRow) * cols + (1 + currCol)], // 4
            img[(1 + currRow) * cols + (-1 + currCol)], // 5
            img[(1 + currRow) * cols + (currCol)], // 6
            img[(1 + currRow) * cols + (1 + currCol)] // 7

        };
        int x = 0;
        int y = 0;

        x += -1 * commonpixels[0] +
        1 * commonpixels[2] +
        -2 * commonpixels[3] +
        2 * commonpixels[4] +
        -1 * commonpixels[5] +
        1 * commonpixels[7];

        y += -1 * commonpixels[0] +
        -2 * commonpixels[1] +
        -1 * commonpixels[2] +
        1 * commonpixels[5] +
        2 * commonpixels[6] +
        1 * commonpixels[7];
        output[index] = (int)roundf(sqrtf(x * x + y * y));
    }
}

__global__ void sobelMemOpt2(uchar *img, uchar *output, unsigned short int cols, unsigned short int rows)
{
    int index = (blockIdx.x * THREADS + threadIdx.x) * 2;
    int currCol = index % cols;
    int currRow = (int)truncf(index / cols);
    bool usable = (currCol % (cols - 2)) != 0 && (currRow % (rows - 1)) != 0;
    if (usable)
    {
        int xa, xb, ya, yb;

        uchar fa[] = {
            img[(-1 + currRow) * cols + (-1 + currCol)],
            img[(-1 + currRow) * cols + (currCol)],
            img[(-1 + currRow) * cols + (1 + currCol)],
            img[(-1 + currRow) * cols + (2 + currCol)]
        };
        xa = -1 * fa[0] + fa[2];
        xb = -1 * fa[1] + fa[3];

        ya = -1 * fa[0] + -2 * fa[1] + -1 * fa[2];
        yb = -1 * fa[1] + -2 * fa[2] + -1 * fa[3];

        uchar fb[] = {
            img[(currRow) * cols + (-1 + currCol)],
            img[(currRow) * cols + (currCol)],
            img[(currRow) * cols + (1 + currCol)],
            img[(currRow) * cols + (2 + currCol)]
        };

        xa += -2 * fb[0] + 2 * fb[2];
        xb += -2 * fb[1] + 2 * fb[3];

        uchar fc[] = {
            img[(currRow + 1) * cols + (-1 + currCol)],
            img[(currRow + 1) * cols + (currCol)],
            img[(currRow + 1) * cols + (1 + currCol)],
            img[(currRow + 1) * cols + (2 + currCol)]
        };

        xa += -1 * fc[0] + fc[2];
        xb += -1 * fc[1] + fc[3];

        ya += 1 * fc[0] + 2 * fc[1] + 1 * fc[2];
        yb += 1 * fc[1] + 2 * fc[2] + 1 * fc[3];

        output[index] = (int)roundf(sqrtf(xa * xa + ya * ya));
        output[index + 1] = (int)roundf(sqrtf(xb * xb + yb * yb));
    }
}

__global__ void sobelMemOpt4(uchar *img, uchar *output, unsigned short int cols, unsigned short int rows)
{
    int index = (blockIdx.x * THREADS + threadIdx.x) * 4;
    int currCol = index % cols;
    int currRow = (int)truncf(index / cols);
    bool usable = (currCol % (cols - 4)) != 0 && (currRow % (rows - 1)) != 0;
    if (usable)
    {
        int xa, xb, xc, xd, ya, yb, yc, yd;

        uchar fa[] = {
            img[(-1 + currRow) * cols + (-1 + currCol)],
            img[(-1 + currRow) * cols + (currCol)],
            img[(-1 + currRow) * cols + (1 + currCol)],
            img[(-1 + currRow) * cols + (2 + currCol)],
            img[(-1 + currRow) * cols + (3 + currCol)],
            img[(-1 + currRow) * cols + (4 + currCol)]
        };
        xa = -1 * fa[0] + fa[2];
        xb = -1 * fa[1] + fa[3];
        xc = -1 * fa[2] + fa[4];
        xd = -1 * fa[3] + fa[5];

        ya = -1 * fa[0] + -2 * fa[1] + -1 * fa[2];
        yb = -1 * fa[1] + -2 * fa[2] + -1 * fa[3];
        yc = -1 * fa[2] + -2 * fa[3] + -1 * fa[4];
        yd = -1 * fa[3] + -2 * fa[4] + -1 * fa[5];

        uchar fb[] = {
            img[(currRow) * cols + (-1 + currCol)],
            img[(currRow) * cols + (currCol)],
            img[(currRow) * cols + (1 + currCol)],
            img[(currRow) * cols + (2 + currCol)],
            img[(currRow) * cols + (3 + currCol)],
            img[(currRow) * cols + (4 + currCol)]
        };

        xa += -2 * fb[0] + 2 * fb[2];
        xb += -2 * fb[1] + 2 * fb[3];
        xc += -2 * fb[2] + 2 * fb[4];
        xd += -2 * fb[3] + 2 * fb[5];

        uchar fc[] = {
            img[(currRow + 1) * cols + (-1 + currCol)],
            img[(currRow + 1) * cols + (currCol)],
            img[(currRow + 1) * cols + (1 + currCol)],
            img[(currRow + 1) * cols + (2 + currCol)],
            img[(currRow + 1) * cols + (3 + currCol)],
            img[(currRow + 1) * cols + (4 + currCol)]
        };

        xa += -1 * fc[0] + fc[2];
        xb += -1 * fc[1] + fc[3];
        xc += -1 * fc[2] + fc[4];
        xd += -1 * fc[3] + fc[5];

        ya += 1 * fc[0] + 2 * fc[1] + 1 * fc[2];
        yb += 1 * fc[1] + 2 * fc[2] + 1 * fc[3];
        yc += 1 * fc[2] + 2 * fc[3] + 1 * fc[4];
        yd += 1 * fc[3] + 2 * fc[4] + 1 * fc[5];

        output[index] = (int)roundf(sqrtf(xa * xa + ya * ya));
        output[index + 1] = (int)roundf(sqrtf(xb * xb + yb * yb));
        output[index + 2] = (int)roundf(sqrtf(xc * xc + yc * yc));
        output[index + 3] = (int)roundf(sqrtf(xd * xd + yd * yd));
    }
}

int main(int argc, char **argv)
{
    struct timespec start, end;
    Mat orig = loadImage(argc, argv);
    // don't need more than an unsigned short int, assuming realistic images
    // max image size: 65535 * 65535
    const unsigned short int cols = orig.cols;
    const unsigned short int rows = orig.rows;
    const unsigned int size = cols * rows * sizeof(char);
    const unsigned int blocks = (unsigned int)ceil(((cols) * rows) / THREADS);
    const unsigned int blocks2 = (unsigned int)ceil(((cols / 2) * rows) / THREADS);
    const unsigned int blocks4 = (unsigned int)ceil(((cols / 4) * rows) / THREADS);
    uchar *rImage, *rOutput;

    auto image = (uchar *)malloc(size);

    matrixToArray(orig, image, cols, rows);

    orig.release();

    cudaMalloc(&rImage, size);
    cudaMalloc(&rOutput, size);

    cudaMemcpy(rImage, image, size, cudaMemcpyHostToDevice);

    clock_gettime(CLOCK_REALTIME, &start);
    sobelMemOpt4<<<blocks4, THREADS>>>(rImage, rOutput, cols, rows);
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_REALTIME, &end);
    double time = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_nsec - start.tv_nsec) / 1000;
    double ms = time / 1000;
    printf("Sobel function exec time: %f microseconds (%f milliseconds)\n", time, ms);
    cudaFree(rImage);

    cudaMemcpy(image, rOutput, size, cudaMemcpyDeviceToHost);

    cudaFree(rOutput);

    showAndSave(arrayToMatrix(image, cols, rows));

    free(image);
    return 0;
}