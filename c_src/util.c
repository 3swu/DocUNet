#include <math.h>
#include <stdio.h>

float distance(float k, int vertex_x, int vertex_y, int point_x, int point_y) {
    float c = k * vertex_x - vertex_y;
    float b = -1 * k;
    int a = 1;

    float result = fabs(a * point_x + b * point_y + c) / sqrt(a * a + b * b);
    
    return result;
}

float get_w(float alpha, float distance_xy, int rows, int type) {
    // type: 0 - fold
    //       1 - curve
    // when type is 0, rows is 0

    return type == 0 ? alpha / (distance_xy + alpha) : (1 - pow(distance_xy / (rows / 2), alpha));
}



void deform(float* label_x, float* label_y, int* shape, int* vertex, float* v, int type) {
    int rows = shape[0], cols = shape[1];
    float k = tanf(v[0]), avg = (rows + cols) / 2;
    
    //parameter alpha can be modified
    float alpha = type == 0 ? (avg / 3) : 2.0;

    // get distance array(same shape of the label and the image)
    float distance_array_2d[rows][cols];
    int i, j;
    for(i = 0; i < rows; i++)
        for(j = 0; j < cols; j++) 
            distance_array_2d[i][j] = distance(k, vertex[0], vertex[1], i, j);

    float w, offset_x, offset_y;
    for(i = 0; i < rows; i++) {
        for(j = 0; j < cols; j++) {
            w = get_w(alpha, distance_array_2d[i][j], rows, type);
            offset_x = v[1] * cosf(v[0]) * w;
            offset_y = v[1] * sinf(v[0]) * w;

            label_x[i * cols + j] += offset_x;
            label_y[i * cols + j] += offset_y;

        }
    }

}

