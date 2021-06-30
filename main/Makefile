main:main.cpp
	rm -f main
	g++ -O3 -mtune=native -march=native -mfpmath=both -o main -I ../qulacs/include -I ../include -L ../qulacs/lib main.cpp ../src/DataCreator.cpp -fopenmp -lcppsim_static -lcsim_static

backup:main_backup.cpp
	rm -f main_backup
	g++ -O3 -mtune=native -march=native -mfpmath=both -o main_backup -I ../qulacs_single/include -L ../qulacs_single/lib main_backup.cpp -fopenmp -lcppsim_static -lcsim_static

clean:
	rm -f main
	rm -f main_backup

clean_all:
	rm -f main
	rm -f main_backup
	rm -f ../result/*.txt
	rm -f ../result/*.csv