#include <math.h>
#define EPSINON 0.0001

double bilinear_interpolation(double src_x, double src_y, 
                              int floor_x, int ceil_x, int floor_y, int ceil_y, 
                              double val_fx_fy, double val_cx_fy, double val_fx_cy, double val_cx_cy) {
    
    return  val_fx_fy * (src_x - floor_x) * (src_y - floor_y) +
            val_cx_fy * (ceil_x - src_x)  * (src_y - floor_y) +
            val_fx_cy * (src_x - floor_x) * (ceil_y - src_y)  +
            val_cx_cy * (ceil_x - src_x)  * (ceil_y - src_y);
    
}

void resize_img_label(double* old_label_x, double* old_label_y,
                      double* new_label_x, double* new_label_y,
                      double* old_img_b, double* old_img_g, double* old_img_r,
                      double* new_img_b, double* new_img_g, double* new_img_r,
                      int* old_shape, int* new_shape) {
    
    int old_rows = old_shape[0], old_cols = old_shape[1];
    int new_rows = new_shape[0], new_cols = new_shape[1];

    double row_prop = (old_rows * 1.0) / new_rows;
    double col_prop = (old_cols * 1.0) / new_cols;

    double src_x, src_y;
    int floor_x, ceil_x, floor_y, ceil_y;
    double offset_x, offset_y;

    for(int i = 0; i < new_rows; i++) {
        for(int j = 0; j < new_cols; j++) {
            src_x = i * row_prop;
            src_y = j * col_prop;

            if ((src_x - (int)src_x) > -EPSINON && (src_x - (int)src_x) < EPSINON) {
                floor_x = (int)floor(src_x);
                ceil_x = floor_x + 1;
            }
            else {
                floor_x = (int)floor(src_x);
                ceil_x = (int)ceil(src_x);
            }
            if ((src_y - (int)src_y) > -EPSINON && (src_y - (int)src_y) < EPSINON) {
                floor_y = (int)floor(src_y);
                ceil_y = floor_y + 1;
            }
            else {
                floor_y = (int)floor(src_y);
                ceil_y = (int)ceil(src_y);
            }

            offset_x = bilinear_interpolation(src_x, src_y, 
                                              floor_x, ceil_x, floor_y, ceil_y, 
                                              old_label_x[floor_x * old_cols + floor_y], old_label_x[ceil_x * old_cols + floor_y], 
                                              old_label_x[floor_x * old_cols + ceil_y], old_label_x[ceil_x * old_cols + ceil_y]);
            // offset_x = old_label_x[floor_x * old_cols + floor_y];
            new_label_x[i * new_cols + j] = offset_x;

            offset_y = bilinear_interpolation(src_x, src_y, 
                                              floor_x, ceil_x, floor_y, ceil_y, 
                                              old_label_y[floor_x * old_cols + floor_y], old_label_y[ceil_x * old_cols + floor_y], 
                                              old_label_y[floor_x * old_cols + ceil_y], old_label_y[ceil_x * old_cols + ceil_y]);
            // offset_y = old_label_y[floor_x * old_cols + floor_y];
            new_label_y[i * new_cols + j] = offset_y;

            new_img_b[i * new_cols + j] = bilinear_interpolation(src_x, src_y, 
                                                                   floor_x, ceil_x, floor_y, ceil_y, 
                                                                   old_img_b[floor_x * old_cols + floor_y], old_img_b[ceil_x * old_cols + floor_y], 
                                                                   old_img_b[floor_x * old_cols + ceil_y], old_img_b[ceil_x * old_cols + ceil_y]);
           
            new_img_g[i * new_cols + j] = bilinear_interpolation(src_x, src_y, 
                                                                   floor_x, ceil_x, floor_y, ceil_y, 
                                                                   old_img_g[floor_x * old_cols + floor_y], old_img_g[ceil_x * old_cols + floor_y], 
                                                                   old_img_g[floor_x * old_cols + ceil_y], old_img_g[ceil_x * old_cols + ceil_y]);                                                                   

            new_img_r[i * new_cols + j] = bilinear_interpolation(src_x, src_y, 
                                                                   floor_x, ceil_x, floor_y, ceil_y, 
                                                                   old_img_r[floor_x * old_cols + floor_y], old_img_r[ceil_x * old_cols + floor_y], 
                                                                   old_img_r[floor_x * old_cols + ceil_y], old_img_r[ceil_x * old_cols + ceil_y]);
        }
    }
}