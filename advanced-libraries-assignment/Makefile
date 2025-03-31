NVCC = nvcc
TARGET = advanced_lib_assignment
SRC = advanced_lib_assignment.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) -o $(TARGET) $(SRC) -lcurand

profile: $(TARGET)
	./advanced_lib_assignment 128
	./advanced_lib_assignment 512
	./advanced_lib_assignment 4096
	./advanced_lib_assignment 32768
	./advanced_lib_assignment 262144
	./advanced_lib_assignment 2097152
	./advanced_lib_assignment 16777216
	./advanced_lib_assignment 134217728

clean:
	rm -f $(TARGET)
