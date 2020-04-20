Debug: lazy.o rule.o itemset.o cache.o test.o score.o training.o core.o unlabeled.o evaluation.o
	g++ -Wall -g -o bin/Debug/alac lazy.o rule.o itemset.o cache.o test.o score.o training.o core.o unlabeled.o evaluation.o -lm

Release: lazy.o rule.o itemset.o cache.o test.o score.o training.o core.o unlabeled.o evaluation.o
	g++ -Wall -g -o bin/Release/alac lazy.o rule.o itemset.o cache.o test.o score.o training.o core.o unlabeled.o evaluation.o -lm

lazy.o: lazy.c rule.c itemset.c criteria.h limits.h
	g++ -Wall -g -c lazy.c

rule.o: rule.c rule.h criteria.h limits.h
	g++ -Wall -g -c rule.c

itemset.o: itemset.c itemset.h limits.h
	g++ -Wall -g -c itemset.c

cache.o: cache.c cache.h limits.h
	g++ -Wall -g -c cache.c

test.o: test.c test.h limits.h
	g++ -Wall -g -c test.c

score.o: score.c score.h limits.h
	g++ -Wall -g -c score.c

training.o: training.c training.h limits.h
	g++ -Wall -g -c training.c

core.o: core.c core.h limits.h
	g++ -Wall -g -c core.c

evaluation.o: evaluation.c evaluation.h limits.h
	g++ -Wall -g -c evaluation.c

unlabeled.o: unlabeled.c unlabeled.h
	g++ -Wall -g -c unlabeled.c

clean:
	rm *.o lazy.exe
