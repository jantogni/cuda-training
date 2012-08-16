all: compile

compile:
	@nvcc bitreverse.cu -o output

debug_compile:
	@nvcc -g -G bitreverse.cu -o output

clean:
	@rm -f output debug_out

x:
	@./output

x_debugg:
	@./debug_out
