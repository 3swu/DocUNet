#include <math.h>
#define EPSINON 0.0001



void from_label_deform(double* label_x, double* label_y, int* label_dst_shape,
                       double* src_img_b, double* src_img_g, double* src_img_r, 
                       double* dst_img_b, double* dst_img_g, double* dst_img_r,
                       int* src_img_shape, int* image_edge) {
    
    int dst_rows = label_dst_shape[0], dst_cols = label_dst_shape[1];
    int src_rows = src_img_shape[0], src_cols = src_img_shape[1];

    double src_x, src_y;
    int floor_x, ceil_x, floor_y, ceil_y;
    int offset_x = (int)(src_rows * 0.5);
    int offset_y = (int)(src_cols * 0.5);

    int min_row = dst_rows, max_row = 0;
    int min_col = dst_cols, max_col = 0;

    for(int i = 0; i < dst_rows; i++) {
        for(int j = 0; j < dst_cols; j++) {
            if(label_x[i * dst_cols + j] > -EPSINON && label_x[i * dst_cols + j] < EPSINON)
                continue;
            else {
                src_x = i - label_x[i * dst_cols + j] - offset_x;
                src_y = j - label_y[i * dst_cols + j] - offset_y;

                if((src_x < 0 || src_x >= src_rows - 1) || (src_y < 0 || src_y >= src_cols - 1))
                    continue;
                else {
                    
                    min_row = i < min_row ? i : min_row;
                    max_row = i > max_row ? i : max_row;
                    min_col = j < min_col ? j : min_col;
                    max_col = j > max_col ? j : max_col;

                    ceil_x = (int)ceil(src_x);
                    ceil_y = (int)ceil(src_y);
                    floor_x = (int)floor(src_x);
                    floor_y = (int)floor(src_y);

                    dst_img_b[i * dst_cols + j] = (int)floor(
                        src_img_b[floor_x * src_cols + floor_y] * (ceil_x - src_x) * (ceil_y - src_y) +
                        src_img_b[floor_x * src_cols + ceil_y] * (ceil_x - src_x) * (src_y - floor_y) +
                        src_img_b[ceil_x * src_cols + floor_y] * (src_x - floor_x) * (ceil_y - src_y) +
                        src_img_b[ceil_x * src_cols + ceil_y] * (src_x - floor_x) * (src_y - floor_y)
                    );

                    dst_img_g[i * dst_cols + j] = (int)floor(
                        src_img_g[floor_x * src_cols + floor_y] * (ceil_x - src_x) * (ceil_y - src_y) +
                        src_img_g[floor_x * src_cols + ceil_y] * (ceil_x - src_x) * (src_y - floor_y) +
                        src_img_g[ceil_x * src_cols + floor_y] * (src_x - floor_x) * (ceil_y - src_y) +
                        src_img_g[ceil_x * src_cols + ceil_y] * (src_x - floor_x) * (src_y - floor_y)
                    );

                    dst_img_r[i * dst_cols + j] = (int)floor(
                        src_img_r[floor_x * src_cols + floor_y] * (ceil_x - src_x) * (ceil_y - src_y) +
                        src_img_r[floor_x * src_cols + ceil_y] * (ceil_x - src_x) * (src_y - floor_y) +
                        src_img_r[ceil_x * src_cols + floor_y] * (src_x - floor_x) * (ceil_y - src_y) +
                        src_img_r[ceil_x * src_cols + ceil_y] * (src_x - floor_x) * (src_y - floor_y)
                    );

                }
            }
        }
    }
    image_edge[0] = min_row;
    image_edge[1] = max_row;
    image_edge[2] = min_col;
    image_edge[3] = max_col;
}