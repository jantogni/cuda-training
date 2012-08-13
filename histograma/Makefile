all: compile

compile:
	@nvcc -arch=sm_11 example.cu -o output

clean:
	@rm -f output

x:
	@./output < img.dat 
