#include <stdio.h>


void transpose(float* input, int rows, int cols, float* output) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            output[j * rows + i] = input[i * cols + j];
        }
    }
}

void dot_product(const float* A, const float* B, float* C, int m, int n, int p) {
    //     In the provided example, the dimensions m, n, and p of the matrices are as follows:

    // Matrix A: m x n = 2 x 3 (2 rows, 3 columns)
    // Matrix B: n x p = 3 x 2 (3 rows, 2 columns)
    // Matrix C: m x p = 2 x 2 (2 rows, 2 columns)
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            float sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * p + j];
            }
            C[i * p + j] = sum;
        }
    }
}

void print_matrix(const float* matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%g ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}

void print_matrixI(const int64_t* matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%g ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}

void kp_rotation(const float* axis, float theta, float* result) {
    float x = axis[0];
    float y = axis[1];
    float z = axis[2];
    float c = cos(theta);
    float s = sin(theta);

    result[0] = c + (x * x) * (1 - c);
    result[1] = x * y * (1 - c) - z * s;
    result[2] = y * s + x * z * (1 - c);
    
    result[3] = z * s + x * y * (1 - c);
    result[4] = c + (y * y) * (1 - c);
    result[5] = -x * s + y * z * (1 - c);
    
    result[6] = -y * s + x * z * (1 - c);
    result[7] = x * s + y * z * (1 - c);
    result[8] = c + (z * z) * (1 - c);

}