#cuda-gdb
#ncu

rm tmpt/*
#rm bin/*; rm trace_impl; rm trace_tete;
clear
printf "[\033[93m***\033[0m] \033[103mCompilation ...\033[0m \n"

#	Compiler
#python3 compiler_tout.py

# g c
# G cuda
A="-Idef -g -G -diag-suppress 2464 -O3 -lm -Xcompiler -fopenmp -Xcompiler -O3"
#
nvcc -c impl/etc/etc.cu ${A} &
nvcc -c impl/etc/marchee.cu ${A} &
nvcc -c impl/etc/perlin.cu ${A} &
#
nvcc -c impl/mdl/mdl.cu ${A} &
nvcc -c impl/mdl/mdl_pred.cu ${A} &
nvcc -c impl/mdl/mdl_plume.cu ${A} &
#
nvcc -c impl/mdl/impl_cpu/impl_cpu_simple.cu ${A} &
nvcc -c impl/mdl/impl_cpu/impl_cpu_thread_cpp.cu ${A} &
nvcc -c impl/mdl/impl_cpu/mdl_f.cu ${A} &
#
nvcc -c impl/mdl/impl_cuda/cuda_methode0_basique.cu ${A} &
nvcc -c impl/mdl/impl_cuda/cuda_methode1_instructions.cu ${A} &
nvcc -c impl/mdl/impl_cuda/cuda_methode2_inst_mini_kerd.cu ${A} &
#
nvcc -c impl/main.cu ${A} &
nvcc -c impl/main_init.cu ${A} &
#
wait
nvcc *.o -o prog ${A}

if [ $? -eq 1 ]
then
	printf "\n[\033[91m***\033[0m] \033[101mErreure. Pas d'execution.\033[0m\n"
	rm *.o
	exit
fi
rm *.o

#	Executer
printf "[\033[92m***\033[0m] \033[102m========= Execution du programme =========\033[0m\n"

#valgrind --leak-check=yes --track-origins=yes ./prog
time ./prog
if [ $? -ne 0 ]
then
	printf "[\033[91m***\033[0m] \033[101mErreur durant l'execution.\033[0m\n"
	#valgrind --leak-check=yes --track-origins=yes ./prog
	#sudo systemd-run --scope -p MemoryMax=100M gdb ./prog
	exit
else
	printf "[\033[92m***\033[0m] \033[102mAucune erreure durant l'execution.\033[0m\n"
fi