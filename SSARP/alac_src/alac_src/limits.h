#ifndef _LIMITS_
#define _LIMITS_

#define KB 1024
#define MB 1024*1024
#define GB 1024*1024*1024

#define MAX_CLASSES 6
#define MAX_RULE_SIZE 10
#define MAX_RULES 500*KB
#define MAX_ITEMS 500*KB
#define MAX_TIME_PER_TEST 1.00

extern int MIN_COUNT, MIN_SIZE, MAX_SIZE;
extern float MIN_CONF, MIN_LEVEL, MIN_SUPP, FACTOR;

#endif
