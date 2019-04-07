#!/bin/bash

# Feifei Xiong (f.xiong@sheffield.ac.uk) @ 2019, SPandH, The University of Sheffield
# contact (Feifei Xiong, Jon Barker, Heidi Christensen)

# specially for UASPEECH corpus
# please ensure that you downloaded the latest version of uaspeech dataset

# NOTE:
# please run the scripts "run_si.sh" firstly !!!

# Begin configuration section.
nj=1  
decode_nj=1
thread_nj=1
stage=0
spkset="F02" 	# speaker-dependent
basemdl="dys/si"
primary_lr_factor=0.25
tdnnf_l=192
edits_layer="set-learning-rate-factor name=* learning-rate-factor=${primary_lr_factor}; set-learning-rate-factor name=*linear learning-rate-factor=1.0; set-learning-rate-factor name=prefinal-l* learning-rate-factor=1.0"
outdir="linear_all"
# End configuration section


. ./utils/parse_options.sh
. ./cmd.sh
. ./path.sh

set -e # exit on error

# path to the global dataset
uaspeech_src=/data/UASPEECH/audio
home_dir=/data/kaldi/egs/uaspeech/s5
flist_dir=$home_dir/local/flist 	# check local dir for some specific scripts for uaspeech

# path to data, feature and exp
data_dir=$home_dir/data
exp_dir=$home_dir/exp

# parameters
lang=$data_dir/lang
lang_chain=$data_dir/lang_chain
scoring_opts="--word-ins-penalty 0.0 --max-lmwt 1"
cmvn_opts="--norm-means=false --norm-vars=false"  	# set both false if online mode

#primary_lr_factor=0.25 # The learning-rate factor for transferred layers from source
                       # model. e.g. if 0, it fixed the paramters transferred from source.
                       # The learning-rate factor for new added layers is 1.0.

# specific parameters
trset="train_dys_sp"	# take speed perturbation version
etset="test_dys"
namset="sd"
#spkset="M04 F03 M12 M01 M07 F02 M16 M05 M11 F04 M09 M14 M10 M08 F05" 	# each dysarthric speaker
#spkset="CM04 CF03 CM12 CM01 CF02 CM05 CF04 CM09 CM10 CM08 CF05 CM06 CM13" 	# each control speaker

trainset=$namset/$spkset
trdata=$data_dir/$trset/$trainset
train_data_dir=$data_dir/${trset}_hires/$trainset
etdata=$data_dir/${etset}_hires/$trainset

# existing model
src_gmm_dir=$exp_dir/train_${basemdl}/$spkset/tri3
src_mdl=$exp_dir/train_${basemdl}/$spkset/chain_cnn_tdnnf/final.mdl
src_tree_dir=$exp_dir/train_${basemdl}/$spkset/chain_cnn_tdnnf/tree_sp


if [ $stage -le 0 ]; then
  required_files="$trdata/feats.scp $train_data_dir/feats.scp $etdata/feats.scp $src_mdl $src_tree_dir/tree"
  for f in $required_files; do
    if [ ! -f $f ]; then
      echo "$0: no such file $f" && exit 1;
    fi
  done
fi


nj_sp=$((3*nj))
# write to tree/num_jobs to ensure train.py runs properly!
nj_tree=`cat $src_tree_dir/num_jobs` || exit 1;
if [ $nj_sp -ne $nj_tree ]; then
  mv $src_tree_dir/num_jobs $src_tree_dir/num_jobs_orig
  echo $nj_sp > $src_tree_dir/num_jobs
  echo "num jobs changes, check tree_sp/num_jobs ..."
fi



lat_dir=$exp_dir/train_${basemdl}/$spkset/transfer/gmm_lats
mkdir -p $lat_dir
# using GMM for alignment with new adapted data
if [ $stage -le 4 ]; then
  steps/align_fmllr_lats.sh --nj $nj_sp --cmd "$train_cmd" \
    --generate-ali-from-lats true \
    $trdata $lang $src_gmm_dir $lat_dir || exit 1;
  rm $lat_dir/fsts.*.gz 2>/dev/null || true # save space
fi



train_stage=-10
xent_regularize=0.1
feat_dim=40
tdnn_initjob=1
tdnn_finaljob=1
chunk_width=140,100,160 		# default: 140,100,160
minibatch_size=128,64			# 128,64
chunk_left_context=0
chunk_right_context=0
train_ivector_dir=
remove_egs=true
common_egs_dir=
reporting_email=
frame_per_iter=3000000		# 3000000
lm_states=2000			# default 1000, always set to 2000 !
dropout_schedule="" 		#'0,0@0.20,0.5@0.50,0'
use_gpu=true			# never 'false' otherwise (18h vs. 20min)

tdnnf_dim=1024  	# 1024 seems to provide better performance!
tdnnf_bn=128		# default 128; 
tdnnf_bn2=$((tdnnf_bn*2))
#tdnnf_l=192		# default 192; 
cnn_opts="l2-regularize=0.01"
tdnn_opts="l2-regularize=0.01 dropout-proportion=0.0 dropout-per-dim-continuous=true"
tdnnf_first_opts="l2-regularize=0.01 dropout-proportion=0.0 bypass-scale=0.0"
tdnnf_opts="l2-regularize=0.01 dropout-proportion=0.0 bypass-scale=0.66"
linear_opts="l2-regularize=0.01 orthonormal-constraint=-1.0"
prefinal_opts="l2-regularize=0.01"
output_opts="l2-regularize=0.005"
dropout_schedule="" #'0,0@0.20,0.5@0.50,0'


# set-learning-rate-factor name=*linear learning-rate-factor=1.0;
# set-learning-rate-factor name=prefinal-l* learning-rate-factor=1.0; 
# set-learning-rate-factor name=output* learning-rate-factor=1.0
# set-learning-rate-factor name=tdnnf15* learning-rate-factor=1.0; 
# set-learning-rate-factor name=output-xent* learning-rate-factor=5.0  # seems not help!

num_epochs=1		# original 4, it seems that performance degrades with larger epochs
initial_effective_lrate=0.0005	# 0.001 default!
final_effective_lrate=0.0001	# 5 times less

nnet_dir=$exp_dir/train_${basemdl}/$spkset/transfer/chain_${primary_lr_factor}_ep${num_epochs}_lr0.5_${outdir}
mkdir -p $nnet_dir $nnet_dir/log


if [ $stage -le 5 ]; then
  # Set the learning-rate-factor for all transferred layers but other layers to primary_lr_factor.
  $train_cmd $nnet_dir/log/generate_input_mdl.log \
    nnet3-am-copy --raw=true --edits="$edits_layer" \
      $src_mdl $nnet_dir/input.raw || exit 1;
fi


chain_opts=
if [ $stage -le 6 ]; then
  steps/nnet3/chain/train.py --stage=$train_stage ${chain_opts[@]} \
    --cmd="$decode_cmd" \
    --trainer.input-model=$nnet_dir/input.raw \
    --feat.online-ivector-dir=$train_ivector_dir \
    --feat.cmvn-opts="$cmvn_opts" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient=0.1 \
    --chain.l2-regularize=0.0000 \
    --chain.apply-deriv-weights=false \
    --chain.lm-opts="--num-extra-lm-states=$lm_states" \
    --trainer.dropout-schedule "$dropout_schedule" \
    --trainer.srand=0 \
    --trainer.max-param-change=2.0 \
    --trainer.num-epochs=$num_epochs \
    --trainer.frames-per-iter=$frame_per_iter \
    --trainer.optimization.num-jobs-initial=$tdnn_initjob \
    --trainer.optimization.num-jobs-final=$tdnn_finaljob \
    --trainer.optimization.initial-effective-lrate=$initial_effective_lrate \
    --trainer.optimization.final-effective-lrate=$final_effective_lrate \
    --trainer.num-chunk-per-minibatch=$minibatch_size \
    --trainer.optimization.momentum=0.0 \
    --egs.chunk-width=$chunk_width \
    --egs.chunk-left-context=$chunk_left_context \
    --egs.chunk-right-context=$chunk_right_context \
    --egs.chunk-left-context-initial=0 \
    --egs.chunk-right-context-final=0 \
    --egs.dir="$common_egs_dir" \
    --egs.opts="--frames-overlap-per-eg 0" \
    --cleanup.remove-egs=$remove_egs \
    --use-gpu=$use_gpu \
    --reporting.email="$reporting_email" \
    --feat-dir=$train_data_dir \
    --tree-dir=$src_tree_dir \
    --lat-dir=$lat_dir \
    --dir=$nnet_dir  || exit 1;
fi


if [ $stage -le 7 ]; then
  # Note: it might appear that this $lang directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  test_ivector_dir=
  frames_per_chunk=$(echo $chunk_width | cut -d, -f1)
  if [ ! -f ${nnet_dir}/decode_${etset}/scoring_kaldi/best_wer ]; then
    steps/nnet3/decode.sh \
	--acwt 1.0 --post-decode-acwt 10.0 \
	--frames-per-chunk $frames_per_chunk \
    	--nj $decode_nj --cmd "$decode_cmd"  --num-threads $thread_nj \
        --online-ivector-dir "$test_ivector_dir" \
	--scoring-opts "$scoring_opts" --stage 0 \
        $src_tree_dir/graph $etdata ${nnet_dir}/decode_${etset} || exit 1;
  fi
fi





