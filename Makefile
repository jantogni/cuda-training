all: compile

compile:
	nvcc example.cu -o output

clean:
	rm -f output
