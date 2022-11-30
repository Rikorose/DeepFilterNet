#include <stdio.h>
#include "deep_filter.h"

#define FRAME_SIZE 480

int main(int argc, char **argv) {
  int first = 1;
  short x[FRAME_SIZE];
  FILE *f1, *fout;
  DFState *st;
  st = df_create("../models/DeepFilterNet3_onnx.tar.gz", 100.);
  if (argc!=3) {
    fprintf(stderr, "usage: %s <noisy speech> <output denoised>\n", argv[0]);
    return 1;
  }
  f1 = fopen(argv[1], "rb");
  fout = fopen(argv[2], "wb");
  while (1) {
    fread(x, sizeof(short), FRAME_SIZE, f1);
    if (feof(f1)) break;
    df_process_frame_i16(st, x, x);
    if (!first) fwrite(x, sizeof(short), FRAME_SIZE, fout);
    first = 0;
  }
  df_free(st);
  fclose(f1);
  fclose(fout);
  return 0;
}
