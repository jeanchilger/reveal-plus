#include "kissdb.h"
#include<stdio.h>
#include<string.h>
#include<stdlib.h>

int main(int argc,char **argv){
    if(argc == 5){
	KISSDB db;
        int key_size = atoi(argv[3]);
        int val_size = atoi(argv[4]);

        char line[val_size];
        char key[key_size];
        memset(line, 0, sizeof(line));

	if (KISSDB_open(&db,argv[2],KISSDB_OPEN_MODE_RWREPLACE,1<<20,sizeof(key),sizeof(line))) {
		printf("KISSDB_open failed\n");
		return 1;
	}
        FILE *raw_file = fopen(argv[1], "r");
        while(fgets(line, sizeof(line), raw_file)){
            memset(key, 0, sizeof(key));
            int i;
            for(i = 0;line[i] && line[i] != ' ';i++)
                key[i] = line[i];
            key[i] = 0;
            if (KISSDB_put(&db,key,line)) {
                printf("KISSDB_put failed \n");
                return 1;
            }
            memset(line, 0, sizeof(line));
        }
        fclose(raw_file);
        KISSDB_close(&db);
    }else if(argc == 4){
	KISSDB db;
        int key_size = atoi(argv[2]);
        int val_size = atoi(argv[3]);

        char val[val_size];
        char key[key_size];
        memset(key, 0, sizeof(key));

	if (KISSDB_open(&db,argv[1],KISSDB_OPEN_MODE_RDONLY,1<<20,sizeof(key),sizeof(val))) {
		printf("KISSDB_open failed\n");
		return 1;
	}
        while(scanf("%s", key) != EOF){
            int q;
            if ((q = KISSDB_get(&db,key,val))) {
                    return 1;
            }
            printf("%s", val);
            memset(key, 0, sizeof(key));
        }
        KISSDB_close(&db);
    }
    return 0;
}
