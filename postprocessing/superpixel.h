#ifndef SUPERPIXEL_H_
#define SUPERPIXEL_H_

//#include <cstdio>
#include "superpixel/segment-image.h"

void segment_cpp(unsigned char *in_arr, int* out_arr, int width, int height,
                 float sigma, float k, int min_size) {
    //float sigma = 0.5;
    //float k = 500;
    //int min_size = 20;

    image<rgb> *img = new image<rgb>(width, height, false);
    delete[] img->data;

    // override with our data
    img->data = (rgb*) in_arr;
    for (int i = 0; i < img->height(); i++)
        img->access[i] = img->data + (i * img->width());

    int num_components;
    image<int> *segm = segment_image_ints(img, sigma, k, min_size, &num_components);
    //printf("Number of componenets is %d\n", num_components);

    // copy data to our array
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
            out_arr[x + y * width] = imRef(segm, x, y);

    delete segm;
    img->data = new rgb[1];
    delete img;
}

#endif // SUPERPIXEL_H_
