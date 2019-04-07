#!/bin/bash

# Feifei Xiong (f.xiong@sheffield.ac.uk) @ 2019, SPandH, The University of Sheffield
# contact (Feifei Xiong, Jon Barker, Heidi Christensen)

# specially for UASPEECH corpus
# please ensure that you downloaded the latest version of uaspeech dataset

# NOTE:
# please run the scripts firstly from /s5_segment/ (https://github.com/ffxiong/uaspeech/s5_segment)

# Begin configuration section.
nj=14  # probably max 13 for ctl and max 15 for dys due to limited speakers
decode_nj=1  # 20
thread_nj=1  # 4
stage=0
trainset="F02" 	# ctl: training with control speech data, or "dys": training with speech from dysarthric speakers
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
etset="test_dys"
lang=$data_dir/lang
boost_sil=1.25
scoring_opts="--word-ins-penalty 0.0"
cmvn_opts="--norm-means=false --norm-vars=false"  	# set both false if online mode
numLeavesTri1=1000
numGaussTri1=10000
numLeavesMLLT=1000
numGaussMLLT=10000
numLeavesSAT=1000
numGaussSAT=15000
numLeavesChain=700 # less than numLeavesSAT


# ================================================================================
# prepare data and lang
trdata=$data_dir/train_dys
trdata_sp=$data_dir/train_dys_sp
trdata_sp_hires=$data_dir/train_dys_sp_hires

trdatasi=$trdata/si/$trainset
trdatasi_sp=$trdata_sp/si/$trainset
trdatasi_sp_hires=$trdata_sp_hires/si/$trainset
mkdir -p $trdatasi $trdatasi_sp $trdatasi_sp_hires

if [ $stage -le 1 ]; then
  # exclude the test speaker from training set
  utils/copy_data_dir.sh ${trdata} ${trdatasi}
  grep -v -e "^${trainset}_B" ${trdata}/text > ${trdatasi}/text
  utils/fix_data_dir.sh ${trdatasi}
  [[ -d ${trdatasi}/.backup ]] && rm -rf ${trdatasi}/.backup

  utils/copy_data_dir.sh ${trdata_sp} ${trdatasi_sp}
  grep -v -e "^sp0.9-${trainset}_B" -e "^sp1.1-${trainset}_B" -e "^${trainset}_B" ${trdata_sp}/text > ${trdatasi_sp}/text
  utils/fix_data_dir.sh ${trdatasi_sp}
  [[ -d ${trdatasi_sp}/.backup ]] && rm -rf ${trdatasi_sp}/.backup

  utils/copy_data_dir.sh ${trdata_sp_hires} ${trdatasi_sp_hires}
  grep -v -e "^sp0.9-${trainset}_B" -e "^sp1.1-${trainset}_B" -e "^${trainset}_B" ${trdata_sp_hires}/text > ${trdatasi_sp_hires}/text
  utils/fix_data_dir.sh ${trdatasi_sp_hires}
  [[ -d ${trdatasi_sp}/.backup ]] && rm -rf ${trdatasi_sp_hires}/.backup
fi

trset=train_dys/si/$trainset
etsetsi=$etset/sd/$trainset
etsetsi_hires=${etset}_hires/sd/$trainset


# ================================================================================
# GMM-HMM training
if [ $stage -le 4 ]; then
 if [ ! -f $exp_dir/$trset/tri2/final.mdl ]; then
  # Starting basic training on MFCC features for control data based on GMM/HMM
  steps/train_mono.sh --nj $nj --cmd "$train_cmd" --cmvn-opts "$cmvn_opts" --boost-silence $boost_sil \
			$data_dir/$trset $lang $exp_dir/$trset/mono
  steps/align_si.sh --nj $nj --cmd "$train_cmd" --boost-silence $boost_sil \
			$data_dir/$trset $lang $exp_dir/$trset/mono $exp_dir/$trset/mono_ali
  steps/train_deltas.sh --cmd "$train_cmd" --cmvn-opts "$cmvn_opts" --boost-silence $boost_sil \
			$numLeavesTri1 $numGaussTri1 $data_dir/$trset $lang $exp_dir/$trset/mono_ali $exp_dir/$trset/tri1
  steps/align_si.sh --nj $nj --cmd "$train_cmd" --boost-silence $boost_sil \
			$data_dir/$trset $lang $exp_dir/$trset/tri1 $exp_dir/$trset/tri1_ali
  steps/train_lda_mllt.sh --cmd "$train_cmd" --cmvn-opts "$cmvn_opts" --boost-silence $boost_sil \
			$numLeavesMLLT $numGaussMLLT $data_dir/$trset $lang $exp_dir/$trset/tri1_ali $exp_dir/$trset/tri2
 fi

 if [ ! -f $exp_dir/$trset/tri3/final.mdl ]; then 
  # SAT training
  steps/align_si.sh --nj $nj --cmd "$train_cmd" --boost-silence $boost_sil \
			$data_dir/$trset $lang $exp_dir/$trset/tri2 $exp_dir/$trset/tri2_ali
  steps/train_sat.sh --cmd "$train_cmd" --boost-silence $boost_sil \
			$numLeavesSAT $numGaussSAT $data_dir/$trset $lang $exp_dir/$trset/tri2_ali $exp_dir/$trset/tri3
 fi
fi

if [ $stage -le 5 ]; then
 # decode + SAT
 utils/mkgraph.sh $lang $exp_dir/$trset/tri3 $exp_dir/$trset/tri3/graph

   if [ ! -f $exp_dir/$trset/tri3/decode_${etset}/scoring_kaldi/best_wer ]; then
    steps/decode_fmllr.sh --nj $decode_nj --cmd "$decode_cmd"  --num-threads $thread_nj --scoring-opts "$scoring_opts" --stage 0 \
			  $exp_dir/$trset/tri3/graph $data_dir/$etsetsi $exp_dir/$trset/tri3/decode_${etset}
   fi

fi
# print results:
# grep WER $exp_dir/$trset/tri*/decode_*/scoring_kaldi/best_wer


################################################################################
# cnn + tdnnf + lfmmi
gmm_dir=$exp_dir/$trset/tri3
ali_dir=${gmm_dir}_ali_sp
scoring_opts="--word-ins-penalty 0.0 --max-lmwt 1"


if [ $stage -le 10 ]; then
  echo "$0: aligning with the perturbed low-resolution data"
  steps/align_fmllr.sh --nj ${nj} --cmd "$train_cmd" \
    ${trdatasi_sp} $lang $gmm_dir $ali_dir || exit 1
fi


# lattice-free
lat_dir=${ali_dir}_lats
if [ $stage -le 11 ]; then
  # Get the alignments as lattices (gives the chain training more freedom).
  # use the same num-jobs as the alignments
  steps/align_fmllr_lats.sh --nj ${nj} --cmd "$train_cmd" ${trdatasi_sp} $lang $gmm_dir $lat_dir
  rm $lat_dir/fsts.*.gz # save space
fi


chaindir=$exp_dir/$trset/chain_cnn_tdnnf
lang_chain=${lang}_chain
tree_dir=$chaindir/tree_sp

# tree 
if [ $stage -le 12 ]; then
  # Build a tree using our new topology.  We know we have alignments for the
  # speed-perturbed data (local/nnet3/run_ivector_common.sh made them), so use
  # those.  The num-leaves is always somewhat less than the num-leaves from the GMM baseline.
  steps/nnet3/chain/build_tree.sh \
    --frame-subsampling-factor 3 \
    --context-opts "--context-width=2 --central-position=1" \
    --cmd "$train_cmd" $numLeavesChain ${trdatasi_sp} \
    $lang_chain $ali_dir $tree_dir
  utils/mkgraph.sh --self-loop-scale 1.0 $lang $tree_dir $tree_dir/graph || exit 1;
fi

xent_regularize=0.1
feat_dim=40
num_epochs=4				# original 4, it seems that performance degrades with larger epochs
tdnn_initjob=2
tdnn_finaljob=4
chunk_width=140,100,160 		# default: 140,100,160
minibatch_size=128,64			# 128,64
chunk_left_context=0
chunk_right_context=0
train_ivector_dir=
remove_egs=true
common_egs_dir=
reporting_email=
frame_per_iter=3000000		# 3000000
initial_effective_lrate=0.0005	# 0.0005
final_effective_lrate=0.00005
lm_states=2000			# default 1000, usually set to 2000 !

tdnnf_dim=1024  	# 1024 seems to provide better performance!
tdnnf_bn=128		# default 128; 
tdnnf_bn2=$((tdnnf_bn*2))
tdnnf_l=192		# default 192; 


cnn_opts="l2-regularize=0.01"
tdnn_opts="l2-regularize=0.01 dropout-proportion=0.0 dropout-per-dim-continuous=true"
tdnnf_first_opts="l2-regularize=0.01 dropout-proportion=0.0 bypass-scale=0.0"
tdnnf_opts="l2-regularize=0.01 dropout-proportion=0.0 bypass-scale=0.66"
linear_opts="l2-regularize=0.01 orthonormal-constraint=-1.0"
prefinal_opts="l2-regularize=0.01"
output_opts="l2-regularize=0.005"
dropout_schedule="" #'0,0@0.20,0.5@0.50,0'

if [ $stage -le 20 ]; then
  num_targets=$(tree-info $tree_dir/tree |grep num-pdfs|awk '{print $2}')
  learning_rate_factor=$(echo "print 0.5/$xent_regularize" | python)

  mkdir -p $chaindir/configs
  cat <<EOF > $chaindir/configs/network.xconfig
  input dim=$feat_dim name=input

  # this takes the MFCCs and generates filterbank coefficients.  The MFCCs
  # are more compressible so we prefer to dump the MFCCs to disk rather
  # than filterbanks.
  idct-layer name=idct input=input dim=$feat_dim cepstral-lifter=22 affine-transform-file=$chaindir/configs/idct.mat

  batchnorm-component name=idct-batchnorm input=idct

  conv-relu-batchnorm-layer name=cnn1 $cnn_opts height-in=40 height-out=40 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=48 learning-rate-factor=0.333 max-change=0.25
  conv-relu-batchnorm-layer name=cnn2 $cnn_opts height-in=40 height-out=40 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=48
  conv-relu-batchnorm-layer name=cnn3 $cnn_opts height-in=40 height-out=20 height-subsample-out=2 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=64
  conv-relu-batchnorm-layer name=cnn4 $cnn_opts height-in=20 height-out=20 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=64
  conv-relu-batchnorm-layer name=cnn5 $cnn_opts height-in=20 height-out=10 height-subsample-out=2 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=64
  conv-relu-batchnorm-layer name=cnn6 $cnn_opts height-in=10 height-out=5 height-subsample-out=2 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=128

  # the first TDNN-F layer has no bypass (since dims don't match), and a larger bottleneck so the information bottleneck doesn't become a problem.
  tdnnf-layer name=tdnnf7 $tdnnf_first_opts dim=$tdnnf_dim bottleneck-dim=$tdnnf_bn2 time-stride=0
  tdnnf-layer name=tdnnf8 $tdnnf_opts dim=$tdnnf_dim bottleneck-dim=$tdnnf_bn time-stride=3
  tdnnf-layer name=tdnnf9 $tdnnf_opts dim=$tdnnf_dim bottleneck-dim=$tdnnf_bn time-stride=3
  tdnnf-layer name=tdnnf10 $tdnnf_opts dim=$tdnnf_dim bottleneck-dim=$tdnnf_bn time-stride=3
  tdnnf-layer name=tdnnf11 $tdnnf_opts dim=$tdnnf_dim bottleneck-dim=$tdnnf_bn time-stride=3
  tdnnf-layer name=tdnnf12 $tdnnf_opts dim=$tdnnf_dim bottleneck-dim=$tdnnf_bn time-stride=3
  tdnnf-layer name=tdnnf13 $tdnnf_opts dim=$tdnnf_dim bottleneck-dim=$tdnnf_bn time-stride=3
  tdnnf-layer name=tdnnf14 $tdnnf_opts dim=$tdnnf_dim bottleneck-dim=$tdnnf_bn time-stride=3
  tdnnf-layer name=tdnnf15 $tdnnf_opts dim=$tdnnf_dim bottleneck-dim=$tdnnf_bn time-stride=3
  linear-component name=prefinal-l dim=$tdnnf_l $linear_opts

  prefinal-layer name=prefinal-chain input=prefinal-l $prefinal_opts big-dim=$tdnnf_dim small-dim=$tdnnf_l
  output-layer name=output include-log-softmax=false dim=$num_targets $output_opts

  prefinal-layer name=prefinal-xent input=prefinal-l $prefinal_opts big-dim=$tdnnf_dim small-dim=$tdnnf_l
  output-layer name=output-xent dim=$num_targets learning-rate-factor=$learning_rate_factor $output_opts
EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $chaindir/configs/network.xconfig --config-dir $chaindir/configs/
fi


if [ $stage -le 21 ]; then
  steps/nnet3/chain/train.py --stage=-10 \
    --cmd="$decode_cmd" \
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
    --use-gpu=true \
    --reporting.email="$reporting_email" \
    --feat-dir=${trdatasi_sp_hires} \
    --tree-dir=$tree_dir \
    --lat-dir=$lat_dir \
    --dir=$chaindir  || exit 1;
fi


test_ivector_dir=
frames_per_chunk=$(echo $chunk_width | cut -d, -f1)
if [ $stage -le 22 ]; then

      if [ ! -f ${chaindir}/decode_${etset}/scoring_kaldi/best_wer ]; then
       steps/nnet3/decode.sh \
          --acwt 1.0 --post-decode-acwt 10.0 \
          --extra-left-context $chunk_left_context \
          --extra-right-context $chunk_right_context \
          --extra-left-context-initial 0 \
          --extra-right-context-final 0 \
          --frames-per-chunk $frames_per_chunk \
          --nj $decode_nj --cmd "$decode_cmd"  --num-threads $thread_nj \
          --online-ivector-dir "$test_ivector_dir" \
	  --scoring-opts "$scoring_opts" --stage 0 \
          $tree_dir/graph $data_dir/${etsetsi_hires} ${chaindir}/decode_${etset} || exit 1;
      fi

fi

echo " ${trainset} done..."


