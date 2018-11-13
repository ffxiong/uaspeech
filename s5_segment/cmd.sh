# "queue.pl" uses qsub.  The options to it are
# options to qsub.  If you have GridEngine installed,
# change this to a queue you have access to.
# Otherwise, use "run.pl", which will run jobs locally
# (make sure your --num-jobs options are no more than
# the number of cpus on your machine.

# run it locally...
export train_cmd=run.pl
export decode_cmd=run.pl
export cuda_cmd=run.pl
export mkgraph_cmd=run.pl


memval=4G
#h_rt="08:00:00"
#mem="-l mem=$mem_val,h_rt=$h_rt"

export train_cmd="queue.pl --mem $memval"
export decode_cmd="queue.pl --mem $memval"
export mkgraph_cmd="queue.pl --mem $memval"
export cuda_cmd="queue.pl --mem $memval --gpu 1"
export cmd="queue.pl --mem $memval"





