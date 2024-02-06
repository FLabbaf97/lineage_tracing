#!/bin/bash -l

#SBATCH --gres=gpu:1
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 18
#SBATCH --mem=71G
#SBATCH --time=72:00:00
##SBATCH --time=1:00:00
##SBATCH -p debug

#SBATCH --chdir /home/labbaf/lineage_tracing/

# input_pdb="RFdiffusion_input/2cci_onlyA.pdb"
# output_prefix="RFdiffusion_tmp_output/2cci-more"
# # By making contigs_AS different from contigs_RFDIFF, the user can implement a more conservative strategy:
# # RFdiffusion keeps more of the protein fixed, for example, by defining more of the protein in contigs_RFDIFF
# # The rest of the pipeline only focuses on contigs_AS, however.
# contigs_AS="[4-12/A8-21/3-11/A29-52/10-40/A78-89/21-51/A126-135/2-10/A142-148/6-18/A161-169/19-49/A204-208/10-100]"
# contigs_RFDIFF="[A0-159/1-2/A161-296]"
# num_designs=10
# num_seq_per_target=40

# first_design_to_loop_over=0
# last_design_to_loop_over=$num_designs



echo STARTING At $(date)

conda activate bread
conda info


srun python train.py 

# module load gcc/8.4.0-cuda
# module load cuda/11.7.0
# module load cudnn/8.0.5.39-11.1-linux-x64


echo FINISHED at $(date)
