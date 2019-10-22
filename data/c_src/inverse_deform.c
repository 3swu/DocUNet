#include <math.h>
#define EPSINON 0.0001

void inverse_deform(double* label_x, double* label_y, int* src_shape,
                    double* src_img_b, double* src_img_g, double* src_img_r,
                    double* dst_img_b, double* dst_img_g, double* dst_img_r,
                    int* dst_shape, int* offset, int* mask) {

    int src_rows = src_shape[0], src_cols = src_shape[1];
    int dst_rows = dst_shape[0], dst_cols = dst_shape[1];

    int offset_x = offset[0], offset_y = offset[1];

    double dst_x, dst_y;
    int floor_x, ceil_x, floor_y, ceil_y;

    for(int i = 0; i < src_rows; i++) {
        for(int j = 0; j < src_cols; j++) {
            if(label_x[i * src_cols + j] >= -EPSINON && label_x[i * src_cols + j] <= EPSINON)
                continue;
            else {
                dst_x = offset_x + i - label_x[i * src_cols + j];
                dst_y = offset_y + j - label_y[i * src_cols + j];

                if((dst_x < 0 || dst_x >= dst_rows) || (dst_y < 0 || dst_y >= dst_cols))
                    continue;
                else {
                    floor_x = (int)floor(dst_x);
                    ceil_x = (int)ceil(dst_x);
                    floor_y = (int)floor(dst_y);
                    ceil_y = (int)ceil(dst_y);

                    dst_img_b[floor_x * dst_cols + floor_y] = src_img_b[i * src_cols + j];

                    dst_img_g[floor_x * dst_cols + floor_y] = src_img_g[i * src_cols + j];

                    dst_img_r[floor_x * dst_cols + floor_y] = src_img_r[i * src_cols + j];

                    mask[floor_x * dst_cols + floor_y] = 1;
                }
            } 
        }
    }
}



void get_edge(double* img, int* shape, int* edge) {
    int rows = shape[0], cols = shape[1];

    int min_x = rows, max_x = 0, min_y = cols, max_y = 0;

    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            if(!(img[i + cols + j] >= -EPSINON && img[i * cols + j] <= EPSINON)) {
                min_x = i < min_x ? i : min_x;
                max_x = i > max_x ? i : max_x;
                min_y = j < min_y ? j : min_y;
                max_y = j > max_y ? j : max_y;
            }

        }
    }
    edge[0] = min_x;
    edge[1] = max_x;
    edge[2] = min_y;
    edge[3] = max_y;
}

void interpolate(double* img_b, double* img_g, double* img_r, int* mask, int* shape) {
    int rows = shape[0], cols = shape[1];

    double sum_b = 0, sum_g = 0, sum_r = 0;
    int count = 0;
    
    for(int i = 1; i < rows - 1; i++) {
        for(int j = 1; j < cols - 1; j++) {
            if(mask[i * cols + j] >= -EPSINON && mask[i * cols + j] <= EPSINON) {
                if(mask[(i - 1) * cols + j] == 1) {
                    sum_b += img_b[(i - 1) * cols + j];
                    sum_g += img_g[(i - 1) * cols + j];
                    sum_r += img_r[(i - 1) * cols + j];
                    count ++;
                }
                if(mask[(i - 1) * cols + (j - 1)] == 1) {
                    sum_b += img_b[(i - 1) * cols + j - 1];
                    sum_g += img_g[(i - 1) * cols + j - 1];
                    sum_r += img_r[(i - 1) * cols + j - 1];
                    count ++;

                }
                if(mask[(i + 1) * cols + j] == 1) {
                    sum_b += img_b[(i + 1) * cols + j];
                    sum_g += img_g[(i + 1) * cols + j];
                    sum_r += img_r[(i + 1) * cols + j];
                    count ++;
                }
                if(mask[(i + 1) * cols + j + 1] == 1) {
                    sum_b += img_b[(i + 1) * cols + j + 1];
                    sum_g += img_g[(i + 1) * cols + j + 1];
                    sum_r += img_r[(i + 1) * cols + j + 1];
                    count ++;
                }

                img_b[i * cols + j] = sum_b / count;
                img_g[i * cols + j] = sum_g / count;
                img_r[i * cols + j] = sum_r / count;

                sum_b = 0;
                sum_g = 0;
                sum_r = 0;
                count = 0;

            }
        }
    }
}