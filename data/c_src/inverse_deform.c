#include <math.h>
#define EPSINON 0.0001

void inverse_deform(double* label_x, double* label_y, int* src_shape,
                    double* src_img_b, double* src_img_g, double* src_img_r,
                    double* dst_img_b, double* dst_img_g, double* dst_img_r,
                    int* dst_shape, int* offset) {

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

void interpolate(double* img_b, double* img_g, double* img_r, double* mask, int* shape) {
    int rows = shape[0], cols = shape[1];

    double sum = 0;
    int count = 0;
    
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            if(mask[i * cols + j] >= -EPSINON && mask[i * cols + j] <= EPSINON) {

            }
        }
    }
}