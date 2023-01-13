#!/bin/sh

module load global/cluster
rm cluster.sh
touch cluster.sh
core=30
num=$[${core}-1]
# writing the parallelisation script to distribute into codes
echo "#!/bin/sh" >> cluster.sh
echo "source /dls/science/groups/i23/yishun/dials_yishun/dials" >> cluster.sh
echo "num=${num}" >> cluster.sh
echo 'for i in $(seq 0 1 ${num});' >> cluster.sh
echo "do" >> cluster.sh
echo '    echo ${i}' >> cluster.sh
echo '    nohup dials.python -u test.py > /home/eaf28336/absorption_correction/cluster/nohup_${i}.out &' >> cluster.sh
echo "done" >> cluster.sh
echo "dials.python -u test.py" >> cluster.sh
qsub -S /bin/sh -l h_rt=03:00:00  -pe smp ${core} cluster.sh


#qsub -S /bin/sh -l h_rt=00:05:00  -pe smp 8 cluster.sh {bash << EOF
##!/bin/sh
#source /dls/science/groups/i23/yishun/dials_yishun/dials
#for i in $(seq 0 1 7);
#do  
#    echo ${i}
#    nohup dials.python -u test.py > /home/eaf28336/absorption_correction/cluster/nohup_${i}.out &
#
#done
#dials.python -u test.py
#EOF}

#
##$ -S /bin/sh
##$ -l h_rt=00:05:00  
##$ -pe smp 8
#
#
#source /dls/science/groups/i23/yishun/dials_yishun/dials
#for i in $(seq 0 1 7);
#do  
#    echo ${i}
#    nohup dials.python -u test.py > /home/eaf28336/absorption_correction/cluster/nohup_${i}.out &
#
#done
#dials.python -u test.py
