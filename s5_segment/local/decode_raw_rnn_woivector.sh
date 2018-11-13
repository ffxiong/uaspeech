#!/bin/bash

# Feifei Xiong (f.xiong@sheffield.ac.uk) @ 2018, SPandH, The University of Sheffield

# This script does decoding with an autoencoder net, which is trained
# by "train_rnn_raw.sh"
# 
# Begin configuration section.
stage=1
nj=8 # number of decoding jobs.  If --transform-dir set, must match that number!
cmd=run.pl
iter=final
num_threads=1 # if >1, will use gmm-latgen-faster-parallel
extra_left_context=0
extra_right_context=0
use_gpu=yes
extra_left_context_initial=-1
extra_right_context_final=-1
online_ivector_dir=
frames_per_chunk=50
# End configuration section.

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [ $# -ne 4 ]; then
  echo "Usage: $0 [options] <data-dir> <mdl-dir> <decode-dir> <dest-data>"
  echo "e.g.:   $0 data/train exp/lstm exp/lstm/test data/test \\"
  echo "main options (for others, see top of script file)"
  echo "  --nj <nj>                                # number of parallel jobs"
  echo "  --cmd <cmd>                              # Command to run in parallel with"
  echo "  --iter <iter>                            # Iteration of model to decode; default is final."
  echo "  --num-threads <n>                        # number of threads to use, default 1."
  echo "  --parallel-opts <opts>                   # e.g. '--num-threads 4' if you supply --num-threads 4"
  exit 1;
fi

data=$1
srcdir=$2
dir=$3
destdir=$4

mdl=$srcdir/$iter.raw
if [ ! -f $mdl ]; then
   echo "$0: no such file $mdl" && exit 1;
fi

sdata=$data/split$nj;

cmvn_opts=`cat $srcdir/cmvn_opts` || exit 1;
thread_string=
[ $num_threads -gt 1 ] && thread_string="-parallel --num-threads=$num_threads"


mkdir -p $destdir/log
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;
echo $nj > $destdir/num_jobs

## Set up feature
feat_type=`cat $srcdir/feat_type` || feat_type="raw";
echo "$0: feature type is $feat_type"

if [ -f $srcdir/splice_opts ]; then
   splice_opts=`cat $srcdir/splice_opts 2>/dev/null` || exit 1;
fi

case $feat_type in
  raw) feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- |";;
  lda) feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $srcdir/final.mat ark:- ark:- |"
    ;;
  *) echo "$0: invalid feature type $feat_type" && exit 1;
esac

if [ $stage -le 1 ]; then
  $cmd JOB=1:$nj $destdir/log/decode.JOB.log \
    nnet3-compute \
	--use-gpu=$use_gpu \
	--frames-per-chunk=$frames_per_chunk \
	--extra-left-context=$extra_left_context \
	--extra-right-context=$extra_right_context \
	--extra-left-context-initial=$extra_left_context_initial \
     	--extra-right-context-final=$extra_right_context_final \
	$mdl \
	"$feats" \
    	ark:- \| \
    copy-feats --compress=true ark:- ark,scp:$dir/decode.JOB.ark,$dir/decode.JOB.scp || exit 1;
#	ark:- \| \
#    copy-feats-to-htk --sample-period=100000 --output-dir=$htkdir --output-ext=htk ark:- || exit 1;
fi

# copy the source data information to destination data dir
utils/copy_data_dir.sh $data $destdir
[ -f $destdir/feats.scp ] && rm $destdir/feats.scp 2>/dev/null
[ -f $destdir/cmvn.scp ] && rm $destdir/cmvn.scp 2>/dev/null

# concatenate the .scp files together.
for n in $(seq $nj); do
  cat $dir/decode.$n.scp || exit 1;
done > $destdir/feats.scp

nf=`cat $destdir/feats.scp | wc -l`
nu=`cat $destdir/utt2spk | wc -l`
if [ $nf -ne $nu ]; then
  echo "It seems not all of the feature files were successfully processed ($nf != $nu);"
  echo "consider using utils/fix_data_dir.sh $destdir"
fi

echo "Decoding done."

