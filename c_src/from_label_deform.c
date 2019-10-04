#include <math.h>

void from_label_deform(float* label_x, float* label_y, 
                        float* src_img_b, float* src_img_g, float* src_img_r, 
                        float* dst_img_b, float* dst_img_g, float* dst_img_r, 
                        int* shape) {
    int rows = shape[0], cols = shape[1];

    int i, j;
    float src_x, src_y;
    int ceil_x, ceil_y, floor_x, floor_y;


    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            src_x = i - label_x[i * cols + j];
            src_y = j - label_y[i * cols + j];

            if ((src_x < 0 || src_x >= rows - 1) || (src_y < 0 || src_y >= cols - 1))
                continue;
            else {
                ceil_x = (int)ceilf(src_x);
                ceil_y = (int)ceilf(src_y);
                floor_x = (int)floorf(src_x);
                floor_y = (int)floorf(src_y);

                dst_img_b[i * cols + j] = (int)floorf(
                    src_img_b[floor_x * cols + floor_y] * (ceil_x - src_x) * (ceil_y - src_y) +
                    src_img_b[floor_x * cols + ceil_y] * (ceil_x - src_x) * (src_y - floor_y) +
                    src_img_b[ceil_x * cols + floor_y] * (src_x - floor_x) * (ceil_y - src_y) +
                    src_img_b[ceil_x * cols + ceil_y] * (src_x - floor_x) * (src_y - floor_y)
                );

                dst_img_g[i * cols + j] = (int)floorf(
                    src_img_g[floor_x * cols + floor_y] * (ceil_x - src_x) * (ceil_y - src_y) +
                    src_img_g[floor_x * cols + ceil_y] * (ceil_x - src_x) * (src_y - floor_y) +
                    src_img_g[ceil_x * cols + floor_y] * (src_x - floor_x) * (ceil_y - src_y) +
                    src_img_g[ceil_x * cols + ceil_y] * (src_x - floor_x) * (src_y - floor_y)
                );

                dst_img_r[i * cols + j] = (int)floorf(
                    src_img_r[floor_x * cols + floor_y] * (ceil_x - src_x) * (ceil_y - src_y) +
                    src_img_r[floor_x * cols + ceil_y] * (ceil_x - src_x) * (src_y - floor_y) +
                    src_img_r[ceil_x * cols + floor_y] * (src_x - floor_x) * (ceil_y - src_y) +
                    src_img_r[ceil_x * cols + ceil_y] * (src_x - floor_x) * (src_y - floor_y)
                );
            }
        }
    }
}


