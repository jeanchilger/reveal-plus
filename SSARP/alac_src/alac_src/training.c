/**
THIS MODULE IMPLEMENTS ALL OPERATIONS OF READING/PARSING THE TRAINING DATA.
*/

#include "training.h"

/**
READS THE TRAINING DATA. ALL ITEMS ARE STORED IN A GLOBAL ARRAY, WHICH IS
THEN ACCESSED DURING RULE INDUCTION.
*/
int read_training_set(char* training) {
	__START_TIMER__
	printf("Reading training data ");
	fflush(stdout);
	int target=-1;
	char** target_name=(char**)malloc(sizeof(char*)*MAX_CLASSES);
	for(int i=0;i<MAX_CLASSES;i++) target_name[i]=(char*)malloc(sizeof(char)*100);
	set<string> proc_items;
	FILE* file=fopen(training,"r");
	if(file==NULL) {
		fprintf(stderr,"Training set %s not found.\n\n", training);
		exit(-1);
	}
	free(training);
	int n_lines=0;
	char done[10];
	while(1) {
		char line[200*KB];
		fgets(line, 200*KB, file);
		if(feof(file)) break;
		n_lines++;
	}
	// marks whether training set is empty or not
	if (n_lines==0) {
        TRAIN_EMPTY=1;
    } else {
        TRAIN_EMPTY=0;
    }
	rewind(file);
	for(int i=0;i<MAX_CLASSES;i++) {
		sprintf(target_name[i], "CLASS=%d", i);
		CLASS_NAME[target_name[i]]=i;
		COUNT_TARGET[i]=0;
		TARGET_ID[i]=N_ITEMSETS;
		ITEMSETS[N_ITEMSETS].count=0;
		ITEMSETS[N_ITEMSETS].size=1;
		ITEMSETS[N_ITEMSETS].evidence=CLASS;
		ITEMSETS[N_ITEMSETS].layout=(int*)malloc(sizeof(int));
		ITEMSETS[N_ITEMSETS].list=(int*)malloc(sizeof(int));
		ITEMSETS[N_ITEMSETS].layout[0]=N_ITEMSETS;
		ITEMSETS[N_ITEMSETS].list[0]=-1;
		SYMBOL_TABLE[N_ITEMSETS]=strdup(target_name[i]);
		ITEM_MAP[target_name[i]]=N_ITEMSETS;
		++N_ITEMSETS;
	}
	while(1) {
		int attr_type=CLASS;
		char line[200*KB];
		fgets(line, 200*KB, file);
		if(feof(file)) break;
		N_TRANSACTIONS++;
		proc_items.clear();
		target=-1;
		char* item=strtok(line, " \t\n");
		item=strtok(NULL, " \t\n");
		while(item!=NULL) {
			if(CLASS_NAME.find(item)!=CLASS_NAME.end()) {
				target=(int)CLASS_NAME[item];
				COUNT_TARGET[target]++;
			}
			if(proc_items.find(item)==proc_items.end()) {
				proc_items.insert(item);
				if(ITEM_MAP.find(item)!=ITEM_MAP.end()) {
					int index=(int)(ITEM_MAP[item]);
					ITEMSETS[index].list=(int*) realloc(ITEMSETS[index].list, sizeof(int)*(ITEMSETS[index].count+1));
					ITEMSETS[index].list[ITEMSETS[index].count]=N_TRANSACTIONS;
					ITEMSETS[index].count++;
				}
				else {
					if(item[0]==A1) attr_type=1;
					else if(item[0]==A2) attr_type=2;
					else if(item[0]==A3) attr_type=4;
					else attr_type=CLASS;
					ITEMSETS[N_ITEMSETS].count=1;
					ITEMSETS[N_ITEMSETS].size=1;
					ITEMSETS[N_ITEMSETS].evidence=attr_type;
					ITEMSETS[N_ITEMSETS].layout=(int*)malloc(sizeof(int));
					ITEMSETS[N_ITEMSETS].list=(int*)malloc(sizeof(int));
					ITEMSETS[N_ITEMSETS].layout[0]=N_ITEMSETS;
					ITEMSETS[N_ITEMSETS].list[0]=N_TRANSACTIONS;
					SYMBOL_TABLE[N_ITEMSETS]=strdup(item);
					ITEM_MAP[item]=N_ITEMSETS;
					++N_ITEMSETS;
				}
			}
			item=strtok(NULL, " \t\n");
		}
		//if(N_TRANSACTIONS%(n_lines/(int)100)==0) {
		//	for(unsigned int i=0;i<strlen(done);i++) printf("\b");
		//	sprintf(done,"%d%%",(int)(100*N_TRANSACTIONS/n_lines));
		//	printf("%s", done);
		//	fflush(stdout);
		//}
	}
	for(unsigned int i=0;i<strlen(done);i++) printf("\b");
	printf("[done]\n");
	for(int i=0;i<MAX_CLASSES;i++) free(target_name[i]);
	free(target_name);
	fclose(file);
	__FINISH_TIMER__
	return(0);
}
