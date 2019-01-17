#include"helper_functions.h"

clock_t begin;

void validate(int *a, int *b, int length) {
	for (int i = 0; i < length; ++i) {
		if (a[i] != b[i]) {
			printf("Different value detected at position: %d,"
				" expected %d but get %d\n", i, a[i], b[i]);
			return;
		}
	}
	printf("Tests PASSED successfully! There is no differences\n");
}

void initialize_data_random(int **data, int data_size) {

	static time_t t;
	srand((unsigned)time(&t));

	*data = (int *)malloc(sizeof(int) * data_size);
	for (int i = 0; i < data_size; i++) {
		(*data)[i] = rand() % RANGE;
	}
}

void initialize_data_zero(int **data, int data_size) {
	*data = (int *)malloc(sizeof(int) * data_size);
	memset(*data, 0, data_size * sizeof(int));
}

void set_clock(){
	begin = clock();
}

double get_time(){
	clock_t end = clock();
	return ((double)(end - begin) / CLOCKS_PER_SEC)*100;
}