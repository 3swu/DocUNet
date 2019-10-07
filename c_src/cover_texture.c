#define EPSINON 0.0001

void cover_texture(double* src_img_b, double* src_img_g, double* src_img_r,
                   double* texture_b, double* texture_g, double* texture_r,
                   double* label, int* shape) {

    int rows = shape[0], cols = shape[1];
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            if(label[i * cols + j] > -EPSINON && label[i * cols + j] < EPSINON) {
                src_img_b[i * cols + j] = texture_b[i * cols + j];
                src_img_g[i * cols + j] = texture_g[i * cols + j];
                src_img_r[i * cols + j] = texture_r[i * cols + j];
            }
        }
    }
}