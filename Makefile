SRC_DIR := src
SRC_FILES := $(wildcard $(SRC_DIR)/*.cpp)

CUDA_SRC_FILES := $(wildcard $(SRC_DIR)/cuda/*.cu)


display: run
	python3 display.py

run: nbody
	./nbody

nbody: $(SRC_FILES)
	nvcc  $(SRC_FILES) $(CUDA_SRC_FILES) -Xcompiler -fopenmp -Isrc -o nbody

clean:
	rm -f *.o nbody
