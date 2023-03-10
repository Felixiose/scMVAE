# This script serves as a template for creating all kNN commands on the Euler cluster.
# Feel free to modify it to the needs of your local machine/cluster.

for exp in `ls -d /cluster/scratch/cdoumont/scMVAE/chkpt/*adipose*`; do
	count_file=`ls -1 "$exp"/*.chkpt 2>/dev/null | wc -l`
	count_file_tsv=`find $exp -maxdepth 1 -name "*.tsv"|wc -l`
	if test $count_file_tsv -ne 0; then
	    :
	elif test $count_file -eq 1; then
	    exp_str=$(basename $exp|sed 's:vae-::g'|sed 's:-fc:-:g'|sed 's:-uni:-:g'|sed 's:-see:-:g')
	    chkpt_file=$(ls -1 "$exp"/*.chkpt)
		echo $exp_str|awk -F "-" -v var1="$chkpt_file" '{ print "bsub -W 24:00 -R \"rusage[mem=5120]\" -n 10",\
		"python -m Semester_Project.scMVAE.kNN.kNN_samples",\
		"--id="$1,\
		"--dataset="$2,\
		"--model="$3,\
		"--fixed_curvature="$4,\
		"--universal="$5,\
		"--seed="$6,\
		"--chkpt="var1 \
		}'
	else
		echo -e \# folder $exp has $count_file chkpt-files. Must be one. skipping
	fi
	done >> knn_adipose_commands.txt