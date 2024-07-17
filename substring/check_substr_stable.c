/* author : Abhinav yukth S*/
#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include <immintrin.h>
#include <stdbool.h>

#if !defined(__GNUC__)
    #error The program was specifically tuned for GNU-C compiler
#endif

typedef char uint8;
typedef int uint32;
typedef long long uint64;

const uint32 BYTE = 8; //8 bits
const uint32 REG_SIZE = 8; //bytes 
const uint32 WSIZE = sizeof(uint8); //word size 1byte
const uint32 W_PER_REG = REG_SIZE/WSIZE; //8 words
const uint32 W_BIT_SIZE = WSIZE*BYTE; //1x8

#define TO_STRING_HELPER(X)   #X
#define TO_STRING(X)          TO_STRING_HELPER(X)
#define UNROLL_LOOP _Pragma(TO_STRING(unroll (REG_SIZE)))

void printa(uint8* arr, int N);
uint8 CHSS_GT_32(uint8 *string, uint8 *substr, uint32 subsize);// __attribute__((optimize("prefetch-loop-arrays")))
uint8 CHSS_LT_32(uint8 *string, uint8 *substr, uint32 subsize);
int _SUBSIZE_GT_REGSIZE_(uint8 *string, uint32 str_size, uint8 *substr, uint32 sub_size);
int _SUBSIZE_LT_REGSIZE_(uint8 *string, uint32 str_size, uint8 *substr, uint32 sub_size);

#define CHECK_SUBSTR(string,substr,subsize)subsize<32?CHSS_LT_32(string,substr,subsize):CHSS_GT_32(string,substr,subsize)
#define IS_SUBSTRING(string,str_size,substr,sub_size) sub_size<REG_SIZE?_SUBSIZE_LT_REGSIZE_(string,\
                        str_size,substr,sub_size):_SUBSIZE_GT_REGSIZE_(string,str_size,substr,sub_size)
#define W_MAX_INT 73
#define W_MIN_INT 70

/*  SUB ARRAY CHECK ROUTINE, SIZE GT 32 BYTES*/
__attribute__((target("avx2"))) uint8 __attribute__((optimize("O0"))) CHSS_GT_32(uint8 *string, uint8 *substr, uint32 subsize){
    __m256i mmVEC256_1, mmVEC256_2, mmVEC256res;
    unsigned bitmask;
    for(int i=0;i<subsize;i+=32){
        mmVEC256_1 = _mm256_loadu_si256((__m256i*)(string + i));
        mmVEC256_2 = _mm256_loadu_si256((__m256i*)(substr + i));
        mmVEC256res = _mm256_cmpeq_epi8(mmVEC256_1, mmVEC256_2);
        bitmask = _mm256_movemask_epi8(mmVEC256res);
        if (bitmask != 0xffffffffU) return 0;}
    uint32 rsize = subsize%32;
    uint32 addridx = subsize-rsize;
    return CHSS_LT_32(&string[addridx],&substr[addridx],rsize);
}

/* TO BE VECTORIZED -> SSE/AVX1 | SUB ARRAY CHECK ROUTINE, SIZE LT 32 BYTES  */
uint8 __attribute__((optimize("O3"))) CHSS_LT_32(uint8 *string, uint8 *substr, uint32 substrlen){
    uint8 match = 1;
    for(int k=0;k<substrlen;k++)
        match = match&(string[k]==substr[k]);  
    return match;
}

/*      SIZE OF SUB STRING MORE THAN REGISTER SIZE  */
int __attribute__((optimize("O0"))) _SUBSIZE_GT_REGSIZE_(uint8 *string, uint32 str_size, uint8 *substr, uint32 sub_size){
    register uint64 a = 0;
    register uint64 b = 0;
    register uint64 c;
    int idx = -1;
    UNROLL_LOOP
    for(uint8 i=0;i<W_PER_REG;i++) a = (a<<W_BIT_SIZE)+string[i];
    UNROLL_LOOP
    for(uint8 i=0;i<W_PER_REG;i++) b = (b<<W_BIT_SIZE)+substr[i];
    
    uint32 i;
    uint8 match = 0;
    for(i=W_PER_REG;i<(str_size-sub_size+REG_SIZE);i++){
        c = a^b;
        if(c==0){ match = 1;
            if(0<(sub_size-REG_SIZE)) match = CHECK_SUBSTR(&string[i],&substr[REG_SIZE],sub_size-REG_SIZE);//check_substr(&string[i],&substr[REG_SIZE],sub_size-REG_SIZE);
            if(match) return (int)i-REG_SIZE;}
        a = a<<W_BIT_SIZE;
        a |= string[i];
    }
    return idx;
}

/*  SUB STRING LOWER THAN REGISTER SIZE  */
int __attribute__((optimize("O0"))) _SUBSIZE_LT_REGSIZE_(uint8 *string, uint32 str_size, uint8 *substr, uint32 sub_size){
    int idx = -1;
    int i;
    uint32 LSIZE = (REG_SIZE-sub_size)*W_BIT_SIZE;
    register uint64 a = 0;
    register uint64 b = 0;
    register uint64 c;
    for(i=0;i<sub_size;i++) a = (a<<W_BIT_SIZE) + string[i];
    for(i=0;i<sub_size;i++) b = (b<<W_BIT_SIZE) + substr[i];

    for(i=sub_size;i<str_size+1;i++){
        c = a^b;
        if(c==0){return i-sub_size;}
        a = a<<(LSIZE+W_BIT_SIZE);
        a = (a>>LSIZE);
        a |= string[i];
    }
    return idx;
}

/* FUNCTION FOR CDLL CALLS */
uint8 issubstr(uint8 *string, uint32 str_size, uint8 *substr, uint32 sub_size){
    return IS_SUBSTRING(string,str_size,substr,sub_size);
}

/*  --------------TESTING-----------   */
int main(){
    srand((int)time(0));
    size_t str_size = 6000000; //large
    size_t sub_size = 40; //64
    size_t inter = 5600000;

    uint8 *string = malloc((str_size+1)*sizeof(uint8));
    for(int i=0;i<str_size;i++)string[i]=(uint8)(W_MIN_INT+(double)(rand()*(W_MAX_INT-W_MIN_INT)/RAND_MAX));

    uint8 *sub_str = malloc((sub_size+1)*sizeof(uint8));
    for(int i=0;i<sub_size;i++)
        {sub_str[i]=(uint8)(W_MIN_INT+(double)(rand()*(W_MAX_INT-W_MIN_INT)/RAND_MAX));
        string[i]=(uint8)(W_MIN_INT+(double)(rand()*(W_MAX_INT-W_MIN_INT)/RAND_MAX));
        }
    printf("SubS: ");
    printa(sub_str,sub_size);

    for(int i=0;i<sub_size;i++)string[i+inter] = sub_str[i];
    int idx = IS_SUBSTRING(string,str_size,sub_str,sub_size);
    if(idx>0)
        printa(&string[idx],sub_size);
    printf("\n index=%d\n",idx);
    return 0;
}

void printa(uint8* arr, int N){
    for(int i=0;i<N;i++)
        printf(" %d ,",arr[i]);
    printf("\n");
}