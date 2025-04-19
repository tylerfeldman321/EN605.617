all: opencl-audio-panning.cpp
	g++ opencl-audio-panning.cpp -I/usr/include/ -L/usr/lib/x86_64-linux-gnu/ -lOpenCL -lsndfile -o audio-panning.exe

clean: audio-panning.exe
	rm audio-panning.exe