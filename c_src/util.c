#include <math.h>
#include <stdio.h>

float distance(float k, int vertex_x, int vertex_y, int point_x, int point_y) {
    float c = k * vertex_x - vertex_y;
    float b = -1 * k;
    int a = 1;

    float result = fabs(a * point_x + b * point_y + c) / sqrt(a * a + b * b);
    
    return result;
}

float w(float alpha, float distance_xy, int rows, int type) {
    // type: 0 - fold
    //       1 - curve
    // when type is 0, rows is 0

    return type == 0 ? alpha / (distance_xy + alpha) : (1 - pow(distance_xy / (rows / 2), alpha));
}



int main() {
    // int v_x = 261;
    // int v_y = 392;
    // float k = 1.0842066;
    // printf("%f\n", distance(k, v_x, v_y, 500, 500));

    printf("%lf", 1 - pow(0.788512, 2.0));

}