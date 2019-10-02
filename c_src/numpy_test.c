#include <stdio.h>

// int test(int matrix[], int rows, int cols) {
//     int sum = 0;
//     for(int i = 0; i < rows; i++) {
//         for(int j = 0; j < cols; j++) {
//             printf("%d ", matrix[i * rows + j]);
//             sum += matrix[i * rows + j];
//         }
//         printf("\n");
//     }

//     return sum;
// }

// float test(float matrix[], long strides[], long shapes[]) {
//     float sum = 0;
//     long rows = shapes[0], cols = shapes[1];
//     long s0 = strides[0] / sizeof(float), s1 = strides[1] / sizeof(float);

//     printf("%ld, %ld, %ld, %ld\n", rows, cols, s0, s1);

//     for (int i = 0; i < rows; i++) {
//         for (int j = 0; j < cols; j++) {
//             printf("%f ", matrix[i * s0 + j * s1]);
//             // matrix[i * s0 + j * s1] += 1;
//             sum += matrix[i * s0 + j * s1];
//         }
//         printf("\n");
//     }

//     return sum;
// }

float test(float* matrix_x, float* matrix_y, int rows, int cols) {
    float sum = 0;
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            printf("%f | %f, ", matrix_x[i * rows + j], matrix_y[i * rows + j]);
            matrix_x[i * rows + j] += 1;
            matrix_y[i * rows + j] += 1;
            // printf("%f ", matrix_x[i * rows + j]);
        }
        printf("\n");
    }
    return 0;
}