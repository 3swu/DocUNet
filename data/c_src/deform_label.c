#include <math.h>
#define EPSINON 0.0001

double distance(double k, int vertex_x, int vertex_y, int point_x, int point_y) {
    double c = k * vertex_x - vertex_y;
    double b = -1 * k;
    int a = 1;

    float result = fabs(a * point_x + b * point_y + c) / sqrt(a * a + b * b);
    
    return result;
}

double get_w(double alpha, double distance_xy, int rows, int type) {
    // type: 0 - fold
    //       1 - curve
    // when type is 0, rows is 0

    return type == 0 ? alpha / (distance_xy + alpha) : (1 - pow(distance_xy / (rows / 2), alpha));
}



void deform(double* old_label_x, double* old_label_y, double* new_label_x, double* new_label_y, int* shape, int* vertex, double* v, int type) {
    int rows = shape[0], cols = shape[1];
    double k = tan(v[0]), avg = (rows + cols) / 2;
    
    //parameter alpha can be modified
    double alpha = type == 0 ? (avg / 3) : 1.5;

    int i, j;

    // get distance array(same shape of the label and the image)
    double distance_array_2d[rows][cols];
    
    for(i = 0; i < rows; i++)
        for(j = 0; j < cols; j++) 
            distance_array_2d[i][j] = distance(k, vertex[0], vertex[1], i, j);

    double w, offset_x, offset_y;
    double old_x, old_y;
    int floor_x, ceil_x, floor_y, ceil_y;
    int temp_x, temp_y;

    for(i = 0; i < rows; i++) {
        for(j = 0; j < cols; j++) {
            w = get_w(alpha, distance_array_2d[i][j], rows, type);
            offset_x = v[1] * cos(v[0]) * w;
            offset_y = v[1] * sin(v[0]) * w;

            old_x = i - offset_x;
            old_y = j - offset_y;

            temp_x = (int)old_x;
            temp_y = (int)old_y;

            if ((old_x < 0 || old_x >= rows - 1) || (old_y < 0 || old_y >= cols - 1))
                continue;
            
            else if(old_label_x[temp_x * cols + temp_y] >= -EPSINON && old_label_x[temp_x * cols + temp_y] <= EPSINON) 
                continue;
        
            else {
                floor_x = (int)floor(old_x);
                floor_y = (int)floor(old_y);

                new_label_x[i * cols + j] = offset_x + old_label_x[floor_x * cols + floor_y];
                new_label_y[i * cols + j] = offset_y + old_label_y[floor_x * cols + floor_y];
            }
        }
    }
}

