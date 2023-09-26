// write the header file for the testkit.c
//

#ifndef _TESTKIT_H_
#define _TESTKIT_H_

#ifdef __cplusplus
extern "C" {
#endif

int64_t count_len(int64_t *arr);
void printArray(int64_t arr[], int64_t n);
void printArrayshort(int8_t arr[], int64_t n);
void printArrayD(double arr[], int64_t n);

#ifdef __cplusplus
}
#endif

#endif




