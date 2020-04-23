#ifndef _TRAINING_
#define _TRAINING_

#include <stdio.h>
#include <string.h>
#include <set>
#include <map>

#include "itemset.h"
#include "evidence.h"
#include "limits.h"
#include "timer.h"

using namespace std;

extern int CLASS_MAP[MAX_CLASSES], META_LEARNING, TARGET_ID[MAX_CLASSES], COUNT_TARGET[MAX_CLASSES], TRAIN_EMPTY;
extern long N_TRANSACTIONS;
extern char *DELIM;
extern map<string, int> CLASS_NAME;
extern map<int, string> SYMBOL_TABLE;
extern map<string, int> ITEM_MAP;

int read_training_set(char* training);

#endif
