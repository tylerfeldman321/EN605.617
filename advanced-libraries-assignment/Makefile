NVCC = nvcc
TARGET = advanced_lib_assignment
SRC = advanced_lib_assignment.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) -o $(TARGET) $(SRC) -lcurand

clean:
	rm -f $(TARGET)
