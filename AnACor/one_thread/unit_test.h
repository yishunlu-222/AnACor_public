#ifndef UNIT_TEST_H
#define UNIT_TEST_H
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <malloc.h>
#include "ray_tracing.h"

void test_ray_classes( Path2_c path_1,int* coord,int * h_ray_classes,int diagonal){
        printf("CPU ==> coord is [%ld,%ld,%ld] \n",coord[0],coord[1],coord[2]);
        int *path_ray_classes = path_1.ray_classes;
        int len_path_2 = path_1.len_path_2;
        int len_classes = path_1.len_classes;
        printf("CPU ==> len_path_2 is %ld \n",len_path_2);
        printf("CPU ==> len_classes is %ld \n",len_classes);
        printf("CPU ==> path_ray_classes\n");
        int counter=0;
        for (int i=0; i<len_path_2; i++){
            printf(" %ld",path_ray_classes[i]);
            if(path_ray_classes[i]!=0){
                counter++;  
            }
        }
        printf("\n");
        printf("nonzero number is %d \n",counter);
        printf("\n");
        for (int i=diagonal; i<diagonal+len_path_2; i++){
            printf(" %d",h_ray_classes[i]);
        }
        printf("\n");
        printf("difference \n");
        for (int i=0; i<len_path_2; i++){
            int h = i+diagonal;
            printf(" %ld",path_ray_classes[i] - h_ray_classes[h]);
        }
}

#endif  // RAY_H