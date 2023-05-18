#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>

int64_t count_len(int64_t *arr)
{
    int64_t count = 0;
    while (*arr != '\0')
    {
        count++;
        arr++;
    }
    printf("Length of array: %d\n", count);

    return count;
}

void printArray(int64_t arr[], int64_t n)
{
    for (int64_t i = 0; i < n; i++)
    {
        printf("%d ", arr[i]);
        if (i % 3 == 2)
        {
            printf("\n");
        }
    }
    printf("\n");
}

void printArrayshort(int8_t arr[], int64_t n)
{
    for (int64_t i = 0; i < n; i++)
    {
        printf("%d ", arr[i]);
        if (i % 3 == 2)
        {
            printf("\n");
        }
    }
    printf("\n");
}

void printArrayD(double arr[], int64_t n)
{
    for (int64_t i = 0; i < n; i++)
    {
        printf("%.15lf ", arr[i]);
        if (i % 3 == 2)
        {
            printf("\n");
        }
    }
    printf("\n");
}