#!/bin/bash

# Feifei Xiong (f.xiong@sheffield.ac.uk) @ 2018, SPandH, The University of Sheffield
# contact (Feifei Xiong, Jon Barker, Heidi Christensen)

# specially for UASPEECH corpus
# please ensure that you downloaded the latest version of uaspeech dataset

# Begin configuration section.
nj=1  # probably max 13 for ctl and max 15 for dys due to limited speakers
decode_nj=1  # 20
thread_nj=1  # 4
stage=0
trainset="dys" 	# ctl: training with control speech data, or "dys": training with speech from dysarthric speakers
# End configuration section


. ./utils/parse_options.sh
. ./cmd.sh
. ./path.sh

set -e # exit on error


# path to the global dataset
uaspeech_src=/data/UASPEECH/audio # change to your corpus folder
home_dir=/data/kaldi/egs/uaspeech/s5_segment
flist_dir=$home_dir/local/flist 	# check local dir for some specific scripts for uaspeech

# path to data, feature and exp
data_dir=$home_dir/data  # if "ln -s /fastdata/$USER/kaldi/egs/uaspeech/s5/data6 data6", then do it at the beginning, or unlink 
feat_dir=$home_dir/mfcc
exp_dir=$home_dir/exp

# parameters
settyp=$trainset
trset="train_${settyp}"
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
nndepth=7
rbm_lrate=0.1
rbm_iter=6  	# smaller datasets should have more iterations!
hid_dim=2048	# according to the total pdfs (gmm-info tri3/final.mdl)
learn_rate=0.002
acwt=0.1 	# only affects pruning (scoring is on lattices)
bn_dim=320

if [[ "$settyp" = "ctl"* ]]; then
 etset="test_ctl test_dys" 	# test speech from both control and dysarthric speakers, you can switch off one if you'd like
 numLeavesTri1=500
 numGaussTri1=5000
 numLeavesMLLT=500
 numGaussMLLT=5000
 numLeavesSAT=500
 numGaussSAT=5000
 numLeavesChain=400 # less than numLeavesSAT
 rbm_lrate=0.2
 rbm_iter=10  	# smaller datasets should have more iterations!
 learn_rate=0.004
 bn_dim=224
fi


# ================================================================================
# prepare data and lang
if [ $stage -le 1 ]; then
  # generate the data files for training and test, in data/train*,test*
  local/prepare_uaspeech_data_segments.sh --stage 0 --settyp "$settyp" --nj $nj --cleanup false $flist_dir $uaspeech_src $data_dir || exit 1;
fi

if [ $stage -le 2 ]; then
  # generate uni grammer for UASPEECH in data/lang
  if [ ! -f $data_dir/lang/G.fst ]; then
   local/prepare_uaspeech_lang.sh $flist_dir $data_dir || exit 1;
  fi
fi


# ================================================================================
# feature calculation
if [ $stage -le 3 ]; then
  # mfcc
  for x in $trset $etset; do
    if [ ! -f $data_dir/$x/cmvn.scp ]; then
     steps/make_mfcc.sh --nj $nj --cmd "$train_cmd" $data_dir/$x $exp_dir/make_mfcc/$x $feat_dir/$x
     steps/compute_cmvn_stats.sh $data_dir/$x $exp_dir/make_mfcc/$x $feat_dir/$x
     utils/fix_data_dir.sh $data_dir/$x
    fi
  done
fi


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
 # decode 
 utils/mkgraph.sh $lang $exp_dir/$trset/tri2 $exp_dir/$trset/tri2/graph
 for dset in $etset; do
   if [ ! -f $exp_dir/$trset/tri2/decode_${dset}/scoring_kaldi/best_wer ]; then
    steps/decode.sh --nj $decode_nj --cmd "$decode_cmd"  --num-threads $thread_nj --scoring-opts "$scoring_opts" --stage 0 \
		    $exp_dir/$trset/tri2/graph $data_dir/${dset} $exp_dir/$trset/tri2/decode_${dset}
   fi
 done
 # decode + SAT
 utils/mkgraph.sh $lang $exp_dir/$trset/tri3 $exp_dir/$trset/tri3/graph
 for dset in $etset; do
   if [ ! -f $exp_dir/$trset/tri3/decode_${dset}/scoring_kaldi/best_wer ]; then
    steps/decode_fmllr.sh --nj $decode_nj --cmd "$decode_cmd"  --num-threads $thread_nj --scoring-opts "$scoring_opts" --stage 0 \
			  $exp_dir/$trset/tri3/graph $data_dir/${dset} $exp_dir/$trset/tri3/decode_${dset}
   fi
 done
fi
# print results:
# grep WER $exp_dir/$trset/tri*/decode_*/scoring_kaldi/best_wer



# ================================================================================
# DNN training with mfcc-lda-mllt-fmllr features
gmmdir=$exp_dir/$trset/tri3
dnndir=$exp_dir/$trset/dnn
data_fmllr=$dnndir/fmllr-tri3

if [ $stage -le 6 ]; then
 # dump fmllr features for regular DNN training with Kaldi nnet, do not need cmvn!
 for dset in $etset; do
   steps/nnet/make_fmllr_feats.sh --nj $nj --cmd "$train_cmd" --transform-dir $gmmdir/decode_${dset} \
     $data_fmllr/$dset $data_dir/$dset $gmmdir $data_fmllr/$dset/log $data_fmllr/$dset/data || exit 1
 done
 
 # for training data
 if [ ! -f $${gmmdir}_ali/ali.1.gz ]; then
  steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" --boost-silence $boost_sil \
			$data_dir/$trset $lang ${gmmdir} ${gmmdir}_ali || exit 1;
 fi

 steps/nnet/make_fmllr_feats.sh --nj $nj --cmd "$train_cmd" --transform-dir ${gmmdir}_ali \
     $data_fmllr/$trset $data_dir/$trset $gmmdir $data_fmllr/$trset/log $data_fmllr/$trset/data || exit 1
 # split the data : 90% train 10% cross-validation (held-out)
 utils/subset_data_dir_tr_cv.sh $data_fmllr/${trset} $data_fmllr/${trset}/tr90 $data_fmllr/${trset}/cv10 || exit 1 
fi


if [ $stage -le 7 ]; then
 # Pre-train DBN, i.e. a stack of RBMs
 dir=$dnndir/pretrain
 if [ ! -f $dir/${nndepth}.dbn ]; then
  (tail --pid=$$ -F $dir/log/pretrain_dbn.log 2>/dev/null)& # forward log
  $cuda_cmd $dir/log/pretrain_dbn.log \
     steps/nnet/pretrain_dbn.sh --nn-depth $nndepth --hid-dim $hid_dim --rbm-lrate $rbm_lrate --rbm-iter $rbm_iter \
				$data_fmllr/${trset} $dir || exit 1;
 fi

 # Train the DNN optimizing per-frame cross-entropy.
 dir=$dnndir
 ali=${gmmdir}_ali
 feature_transform=$dir/pretrain/final.feature_transform
 dbn=$dir/pretrain/${nndepth}.dbn
 (tail --pid=$$ -F $dir/log/train_nnet.log 2>/dev/null)& # forward log
 # Train
 $cuda_cmd $dir/log/train_nnet.log \
   steps/nnet/train.sh --feature-transform $feature_transform --dbn $dbn --hid-layers 0 --learn-rate $learn_rate \
    $data_fmllr/${trset}/tr90 $data_fmllr/${trset}/cv10 $lang $ali $ali $dir || exit 1;
fi

if [ $stage -le 8 ]; then
 # Decode (reuse HCLG graph)
 for dset in $etset; do
   if [ ! -f $dnndir/decode_${dset}/scoring_kaldi/best_wer ]; then
    steps/nnet/decode.sh --nj $decode_nj --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt $acwt \
	--scoring-opts "$scoring_opts" --stage 0 \
      	$gmmdir/graph $data_fmllr/$dset $dnndir/decode_${dset} || exit 1;
   fi
 done
fi
# print results:
# grep WER $dnndir/decode_*/scoring_kaldi/best_wer



# ================================================================================
# TDNN setting without I-vector

# nnet3 config
gmmdir=$exp_dir/$trset/tri3
tdnndir=$exp_dir/$trset/tdnn
tdnn_dim=512
if [ $stage -le 10 ]; then
  echo "$0: creating neural net configs using the xconfig parser";

  num_targets=$(tree-info $gmmdir/tree |grep num-pdfs|awk '{print $2}')

  mkdir -p $tdnndir/configs
  cat <<EOF > $tdnndir/configs/network.xconfig
  input dim=40 name=input

  # the first splicing is moved before the lda layer, so no splicing here
  relu-renorm-layer name=tdnn1 dim=$tdnn_dim input=Append(-2,-1,0,1,2)
  relu-renorm-layer name=tdnn2 dim=$tdnn_dim input=Append(-1,0,1)
  relu-renorm-layer name=tdnn3 dim=$tdnn_dim
  relu-renorm-layer name=tdnn4 dim=$tdnn_dim input=Append(-1,0,1)
  relu-renorm-layer name=tdnn5 dim=$tdnn_dim
  relu-renorm-layer name=tdnn6 dim=$tdnn_dim input=Append(-3,0,3)
  relu-renorm-layer name=tdnn7 dim=$tdnn_dim input=Append(-6,-3,0)
  output-layer name=output dim=$num_targets max-change=1.5
EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $tdnndir/configs/network.xconfig --config-dir $tdnndir/configs/
fi

# tdnn training
train_ivector_dir=
remove_egs=true
common_egs_dir=
reporting_email=
ali_dir=${gmmdir}_ali_${trset}_sp
if [ $stage -le 11 ]; then
  # I think we need to fine-tune the parameters for TDNN!!!
  tdnn_finaljob=4
  steps/nnet3/train_dnn.py --stage=-10 \
    --cmd="$decode_cmd" \
    --feat.online-ivector-dir="$train_ivector_dir" \
    --feat.cmvn-opts="--norm-means=false --norm-vars=false" \
    --trainer.srand=0 \
    --trainer.max-param-change=2.0 \
    --trainer.num-epochs=3 \
    --trainer.samples-per-iter=400000 \
    --trainer.optimization.num-jobs-initial=2 \
    --trainer.optimization.num-jobs-final=$tdnn_finaljob \
    --trainer.optimization.initial-effective-lrate=0.0015 \
    --trainer.optimization.final-effective-lrate=0.00015 \
    --trainer.optimization.minibatch-size=256,128 \
    --egs.dir="$common_egs_dir" \
    --cleanup.remove-egs=$remove_egs \
    --use-gpu=true \
    --reporting.email="$reporting_email" \
    --feat-dir=$data_dir/${trset}_sp_hires \
    --ali-dir=$ali_dir \
    --lang=$lang \
    --dir=$tdnndir  || exit 1;
fi

# decode
test_ivector_dir=
if [ $stage -le 12 ]; then
  for dset in $etset; do
    if [ ! -f ${tdnndir}/decode_${dset}/scoring_kaldi/best_wer ]; then
     steps/nnet3/decode.sh --nj $decode_nj --cmd "$decode_cmd"  --num-threads $thread_nj \
    	--online-ivector-dir "$test_ivector_dir" \
	--scoring-opts "$scoring_opts" --stage 0 \
  	$gmmdir/graph $data_dir/${dset}_hires ${tdnndir}/decode_${dset} || exit 1;
    fi
  done
fi


# ================================================================================
# chain setup without I-vectors
gmmdir=$exp_dir/$trset/tri3
chaindir=$exp_dir/$trset/chain
lang_chain=${lang}_chain

tree_dir=$exp_dir/$trset/chain/tree_sp
ali_dir=${gmmdir}_ali_${trset}_sp

xent_regularize=0.1
tdnn_dim=448
if [ $stage -le 20 ]; then
  num_targets=$(tree-info $tree_dir/tree |grep num-pdfs|awk '{print $2}')
  learning_rate_factor=$(echo "print 0.5/$xent_regularize" | python)
  opts="l2-regularize=0.05"
  output_opts="l2-regularize=0.01 bottleneck-dim=$bn_dim"

  mkdir -p $chaindir/configs
  cat <<EOF > $chaindir/configs/network.xconfig
  input dim=40 name=input

  # the first splicing is moved before the lda layer, so no splicing here
  relu-batchnorm-layer name=tdnn1 $opts dim=$tdnn_dim input=Append(-2,-1,0,1,2)
  relu-batchnorm-layer name=tdnn2 $opts dim=$tdnn_dim input=Append(-1,0,1)
  relu-batchnorm-layer name=tdnn3 $opts dim=$tdnn_dim
  relu-batchnorm-layer name=tdnn4 $opts dim=$tdnn_dim input=Append(-1,0,1)
  relu-batchnorm-layer name=tdnn5 $opts dim=$tdnn_dim
  relu-batchnorm-layer name=tdnn6 $opts dim=$tdnn_dim input=Append(-3,0,3)
  relu-batchnorm-layer name=tdnn7 $opts dim=$tdnn_dim input=Append(-3,0,3)
  relu-batchnorm-layer name=tdnn8 $opts dim=$tdnn_dim input=Append(-6,-3,0)

  ## adding the layers for chain branch
  relu-batchnorm-layer name=prefinal-chain $opts dim=$tdnn_dim target-rms=0.5
  output-layer name=output include-log-softmax=false $output_opts dim=$num_targets max-change=1.5

  # adding the layers for xent branch
  # This block prints the configs for a separate output that will be
  # trained with a cross-entropy objective in the 'chain' models... this
  # has the effect of regularizing the hidden parts of the model.  we use
  # 0.5 / args.xent_regularize as the learning rate factor- the factor of
  # 0.5 / args.xent_regularize is suitable as it means the xent
  # final-layer learns at a rate independent of the regularization
  # constant; and the 0.5 was tuned so as to make the relative progress
  # similar in the xent and regular final layers.
  relu-batchnorm-layer name=prefinal-xent input=tdnn8 $opts dim=$tdnn_dim target-rms=0.5
  output-layer name=output-xent $output_opts dim=$num_targets learning-rate-factor=$learning_rate_factor max-change=1.5
EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $chaindir/configs/network.xconfig --config-dir $chaindir/configs/
fi

chunk_width=140,100,160
chunk_left_context=0
chunk_right_context=0
train_ivector_dir=
train_data_dir=$data_dir/${trset}_sp_hires
if [ $stage -le 21 ]; then
  tdnn_finaljob=4
  steps/nnet3/chain/train.py --stage=-10 \
    --cmd="$decode_cmd" \
    --feat.online-ivector-dir=$train_ivector_dir \
    --feat.cmvn-opts="--norm-means=false --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient=0.1 \
    --chain.l2-regularize=0.0000 \
    --chain.apply-deriv-weights=false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --trainer.srand=0 \
    --trainer.max-param-change=2.0 \
    --trainer.num-epochs=4 \
    --trainer.frames-per-iter=3000000 \
    --trainer.optimization.num-jobs-initial=2 \
    --trainer.optimization.num-jobs-final=$tdnn_finaljob \
    --trainer.optimization.initial-effective-lrate=0.0005 \
    --trainer.optimization.final-effective-lrate=0.00005 \
    --trainer.optimization.shrink-value=1.0 \
    --trainer.num-chunk-per-minibatch=256,128,64 \
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
    --feat-dir=$train_data_dir \
    --tree-dir=$tree_dir \
    --lat-dir=$lat_dir \
    --dir=$chaindir  || exit 1;
fi


test_ivector_dir=
if [ $stage -le 22 ]; then
  frames_per_chunk=$(echo $chunk_width | cut -d, -f1)

  for dset in $etset; do
      if [ ! -f ${chaindir}/decode_${dset}/socring_kaldi/best_wer ]; then
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
          $tree_dir/graph $data_dir/${dset}_hires ${chaindir}/decode_${dset} || exit 1;
      fi
  done
fi




