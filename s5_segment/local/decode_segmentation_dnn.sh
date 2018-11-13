#!/bin/bash

# Copyright 2014  Guoguo Chen, 2015 GoVivace Inc. (Nagendra Goel)
#           2017  Vimal Manohar
# Apache 2.0

# Some basic error checking, similar to steps/nnet/decode.sh is added.

set -e
set -o pipefail

# Begin configuration section.
nnet=               # non-default location of DNN (optional)
feature_transform=  # non-default location of feature_transform (optional)
model=              # non-default location of transition model (optional)
class_frame_counts= # non-default location of PDF counts (optional)
srcdir=             # non-default location of DNN-dir (decouples model dir from decode dir)

stage=0 # stage=1 skips lattice generation
nj=4
cmd=run.pl

acwt=0.10 # note: only really affects pruning (scoring is on lattices).
beam=13.0
lattice_beam=6.0
min_active=200
max_active=7000 # limit of active tokens
max_mem=50000000 # approx. limit to memory consumption during minimization in bytes
nnet_forward_opts="--no-softmax=true --prior-scale=1.0"

scoring_opts=
allow_partial=true
# note: there are no more min-lmwt and max-lmwt options, instead use
# e.g. --scoring-opts "--min-lmwt 1 --max-lmwt 20"
skip_scoring=false

num_threads=1 # if >1, will use latgen-faster-parallel
parallel_opts=   # Ignored now.
use_gpu="no" # yes|no|optionaly
# End configuration section.

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
   echo "$0: This is a special decoding script for segmentation where we"
   echo "use one decoding graph per segment. We assume a file HCLG.fsts.scp exists"
   echo "which is the scp file of the graphs for each segment."
   echo "This will normally be obtained by steps/cleanup/make_biased_lm_graphs.sh."
   echo "This script does not estimate fMLLR transforms; you have to use"
   echo "the --transform-dir option if you want to use fMLLR."
   echo ""
   echo "Usage: $0 [options] <graph-dir> <data-dir> <decode-dir>"
   echo " e.g.: $0 exp/tri2b/graph_train_si284_split \\"
   echo "             data/train_si284_split exp/tri2b/decode_train_si284_split"
   echo ""
   echo "where <decode-dir> is assumed to be a sub-directory of the directory"
   echo "where the model is."
   echo ""
   echo "main options (for others, see top of script file)"
   echo "  --config <config-file>                           # config containing options"
   echo "  --nj <nj>                                        # number of parallel jobs"
   echo "  --iter <iter>                                    # Iteration of model to test."
   echo "  --model <model>                                  # which model to use (e.g. to"
   echo "                                                   # specify the final.alimdl)"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   echo "  --transform-dir <trans-dir>                      # dir to find fMLLR transforms "
   echo "  --acwt <float>                                   # acoustic scale used for lattice generation "
   echo "  --scoring-opts <string>                          # options to local/score.sh"
   echo "  --num-threads <n>                                # number of threads to use, default 1."
   exit 1;
fi


graphdir=$1
data=$2
dir=$3

mkdir -p $dir/log

if [ -z "$srcdir" ]; then
 if [ -e $dir/final.mdl ]; then
   srcdir=$dir
 elif [ -e $dir/../final.mdl ]; then
   srcdir=$(dirname $dir)
 else
   echo "$0: expected either $dir/final.mdl or $dir/../final.mdl to exist"
   exit 1
 fi
fi
sdata=$data/split$nj;

[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;
echo $nj > $dir/num_jobs

utils/lang/check_phones_compatible.sh $graph_dir/phones.txt $srcdir/phones.txt

# Split HCLG.fsts.scp by input utterance
n1=$(cat $graphdir/HCLG.fsts.scp | wc -l)
n2=$(cat $data/feats.scp | wc -l)
if [ $n1 != $n2 ]; then
  echo "$0: expected $n2 graphs in $graphdir/HCLG.fsts.scp, got $n1"
fi


mkdir -p $dir/split_fsts
sort -k1,1 $graphdir/HCLG.fsts.scp > $dir/HCLG.fsts.sorted.scp
utils/filter_scps.pl --no-warn -f 1 JOB=1:$nj \
  $sdata/JOB/feats.scp $dir/HCLG.fsts.sorted.scp $dir/split_fsts/HCLG.fsts.JOB.scp
HCLG=scp:$dir/split_fsts/HCLG.fsts.JOB.scp


# Select default locations to model files (if not already set externally)
[ -z "$nnet" ] && nnet=$srcdir/final.nnet
[ -z "$model" ] && model=$srcdir/final.mdl
[ -z "$feature_transform" -a -e $srcdir/final.feature_transform ] && feature_transform=$srcdir/final.feature_transform
#
[ -z "$class_frame_counts" -a -f $srcdir/prior_counts ] && class_frame_counts=$srcdir/prior_counts # priority,
[ -z "$class_frame_counts" ] && class_frame_counts=$srcdir/ali_train_pdf.counts

# Check that files exist,
for f in $sdata/1/feats.scp $nnet $model $feature_transform $class_frame_counts; do
  [ ! -f $f ] && echo "$0: missing file $f" && exit 1;
done

# Possibly use multi-threaded decoder
thread_string=
[ $num_threads -gt 1 ] && thread_string="-parallel --num-threads=$num_threads"


# PREPARE FEATURE EXTRACTION PIPELINE
# import config,
cmvn_opts=
delta_opts=
D=$srcdir
[ -e $D/norm_vars ] && cmvn_opts="--norm-means=true --norm-vars=$(cat $D/norm_vars)" # Bwd-compatibility,
[ -e $D/cmvn_opts ] && cmvn_opts=$(cat $D/cmvn_opts)
[ -e $D/delta_order ] && delta_opts="--delta-order=$(cat $D/delta_order)" # Bwd-compatibility,
[ -e $D/delta_opts ] && delta_opts=$(cat $D/delta_opts)
#
# Create the feature stream,
feats="ark,s,cs:copy-feats scp:$sdata/JOB/feats.scp ark:- |"
# apply-cmvn (optional),
[ ! -z "$cmvn_opts" -a ! -f $sdata/1/cmvn.scp ] && echo "$0: Missing $sdata/1/cmvn.scp" && exit 1
[ ! -z "$cmvn_opts" ] && feats="$feats apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp ark:- ark:- |"
# add-deltas (optional),
[ ! -z "$delta_opts" ] && feats="$feats add-deltas $delta_opts ark:- ark:- |"
# add-pytel transform (optional),
[ -e $D/pytel_transform.py ] && feats="$feats /bin/env python $D/pytel_transform.py |"


# Run the decoding in the queue,
if [ $stage -le 0 ]; then
  if [ -f "$graphdir/num_pdfs" ]; then
    [ "`cat $graphdir/num_pdfs`" -eq `am-info --print-args=false $model | grep pdfs | awk '{print $NF}'` ] || \
      { echo "Mismatch in number of pdfs with $model"; exit 1; }
  fi
  $cmd --num-threads $((num_threads+1)) JOB=1:$nj $dir/log/decode.JOB.log \
    nnet-forward $nnet_forward_opts --feature-transform=$feature_transform --class-frame-counts=$class_frame_counts --use-gpu=$use_gpu "$nnet" "$feats" ark:- \| \
    latgen-faster-mapped$thread_string --min-active=$min_active --max-active=$max_active --max-mem=$max_mem --beam=$beam \
    --lattice-beam=$lattice_beam --acoustic-scale=$acwt --allow-partial=$allow_partial --word-symbol-table=$graphdir/words.txt \
    $model "$HCLG" ark:- "ark:|gzip -c > $dir/lat.JOB.gz" || exit 1;
fi


if ! $skip_scoring ; then
  [ ! -x local/score.sh ] && \
    echo "$0: Not scoring because local/score.sh does not exist or not executable." && exit 1;
  local/score.sh --cmd "$cmd" $scoring_opts $data $graphdir $dir ||
    { echo "$0: Scoring failed. (ignore by '--skip-scoring true')"; exit 1; }
fi

exit 0;

