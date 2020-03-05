all:
	gcc -Wall -O2 -o indexer kissdb/kissdb.c kissdb/indexer.c

clean:
	rm -f indexer
