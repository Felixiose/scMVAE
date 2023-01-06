# This script serves as a template for creating all commands for training of the models on the Euler cluster.
# Feel free to modify it to the needs of your local machine/cluster.

for ((i=0;i<=5;i++));do
    cat experiments.txt|grep "uc_epi"|awk -v seed=$i '{ print "bsub -W 24:00 -R \"rusage[mem=5120]\" -n 10",\
	"python -m Semester_Project.examples.run",\
	"--id="$1,\
	"--dataset="$2,\
	"--model="$3,\
	"--fixed_curvature="$4,\
	"--universal="$5,\
	"--seed="seed	
	 }'  >> training_uc_epi_commands.txt
done