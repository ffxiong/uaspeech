#!/bin/bash

# Feifei Xiong (f.xiong@sheffield.ac.uk) @ 2018, SPandH, University of Sheffield
# please contact (Feifei Xiong, Jon Barker, Heidi Christensen)

# specially for UASPEECH corpus
# please ensure that you downloaded the latest version of uaspeech dataset

# qsub -V -l h_rt=08:00:00 -o qsub_ctl -e qsub_ctl -j y ./run.sh --trainset "ctl" --nj 13 --decode-nj 13 --stage 0
# qsub -V -l h_rt=08:00:00 -o qsub_dys -e qsub_dys -j y ./run.sh --trainset "dys" --nj 15 --decode-nj 15 --stage 3

# Begin configuration section.
nj=13  # probably max 13 for ctl and max 15 for dys due to limited speakers
decode_nj=13  # 20
thread_nj=1  # 4
stage=0
trainset="ctl" 	# ctl: training with control speech data, or "dys": training with speech from dysarthric speakers
# End configuration section


. ./utils/parse_options.sh
. ./cmd.sh
. ./path.sh

set -e # exit on error


# path to the global dataset
uaspeech_src=/data/UASPEECH/audio # change to your corpus folder
home_dir=/data/kaldi/egs/uaspeech/s5
flist_dir=$home_dir/local/flist 	# check local dir for some specific scripts for uaspeech

# path to data, feature and exp
data_dir=$home_dir/data
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
nndepth=7
rbm_lrate=0.1
rbm_iter=3  	# smaller datasets should have more iterations!
hid_dim=2048	# according to the total pdfs (gmm-info tri3/final.mdl)
learn_rate=0.002
acwt=0.1 	# only affects pruning (scoring is on lattices)

if [[ "$settyp" = "ctl"* ]]; then
 etset="test_ctl test_dys" 	# test speech from both control and dysarthric speakers, you can switch off one if you'd like
 numLeavesTri1=500
 numGaussTri1=5000
 numLeavesMLLT=500
 numGaussMLLT=5000
 numLeavesSAT=500
 numGaussSAT=5000
 rbm_lrate=0.2
 rbm_iter=6  	# smaller datasets should have more iterations!
 learn_rate=0.004
fi

# ================================================================================
# prepare data and lang
if [ $stage -le 1 ]; then
  # generate the data files for training and test, in data/train*,test*
  local/prepare_uaspeech_data.sh --stage 0 --settyp "$settyp" --nj $nj --cleanup false $flist_dir $uaspeech_src $data_dir || exit 1;
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
    fi
    utils/fix_data_dir.sh $data_dir/$x
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
 utils/subset_data_dir_tr_cv.sh $data_fmllr/${trset} $data_fmllr/${trset}_tr90 $data_fmllr/${trset}_cv10 || exit 1 
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
    $data_fmllr/${trset}_tr90 $data_fmllr/${trset}_cv10 $lang $ali $ali $dir || exit 1;
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
# DNN training with FBANK features
subfix="_fbank"
gmmdir=$exp_dir/$trset/tri3
dnndir=$exp_dir/$trset/dnn${subfix}

if [ $stage -le 9 ]; then
 # calculate fbank based on conf/fbank.conf
 for x in $trset $etset; do
   utils/copy_data_dir.sh $data_dir/$x $data_dir/${x}${subfix}
   steps/make_fbank.sh --nj $nj --cmd "$train_cmd" --fbank-config conf/fbank.conf \
			$data_dir/${x}${subfix} $exp_dir/make_mfcc/${x}${subfix} $feat_dir/${x}${subfix}
   steps/compute_cmvn_stats.sh $data_dir/${x}${subfix} $exp_dir/make_mfcc/${x}${subfix} $feat_dir/${x}${subfix}
   utils/fix_data_dir.sh $data_dir/${x}${subfix}
 done

 # split the training data : 90% train 10% cross-validation (held-out)
 utils/subset_data_dir_tr_cv.sh $data_dir/${trset}${subfix} $data_dir/${trset}${subfix}/tr90 $data_dir/${trset}${subfix}/cv10 || exit 1 
fi

if [ $stage -le 10 ]; then
 # Pre-train DBN, i.e. a stack of RBMs
 dir=$dnndir/pretrain
 if [ ! -f $dir/${nndepth}.dbn ]; then
  (tail --pid=$$ -F $dir/log/pretrain_dbn.log 2>/dev/null)& # forward log
  $cuda_cmd $dir/log/pretrain_dbn.log \
     steps/nnet/pretrain_dbn.sh --nn-depth $nndepth --hid-dim $hid_dim --rbm-lrate $rbm_lrate --rbm-iter $rbm_iter --cmvn-opts "$cmvn_opts" \
				$data_dir/${trset}${subfix} $dir || exit 1;
 fi

 # Train the DNN optimizing per-frame cross-entropy.
 dir=$dnndir
 ali=${gmmdir}_ali
 if [ ! -f $ali/ali.1.gz ]; then
  steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" --boost-silence $boost_sil \
    $data_dir/${trset} $lang $gmmdir $ali || exit 1
 fi

 feature_transform=$dir/pretrain/final.feature_transform
 dbn=$dir/pretrain/${nndepth}.dbn
 (tail --pid=$$ -F $dir/log/train_nnet.log 2>/dev/null)& # forward log
 # Train
 $cuda_cmd $dir/log/train_nnet.log \
   steps/nnet/train.sh --feature-transform $feature_transform --dbn $dbn --hid-layers 0 --learn-rate $learn_rate \
    $data_dir/${trset}${subfix}/tr90 $data_dir/${trset}${subfix}/cv10 $lang $ali $ali $dir || exit 1;
fi

if [ $stage -le 11 ]; then
 # Decode (reuse HCLG graph)
 for dset in $etset; do
   if [ ! -f $dnndir/decode_${dset}/scoring_kaldi/best_wer ]; then
    steps/nnet/decode.sh --nj $decode_nj --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt $acwt \
	--scoring-opts "$scoring_opts" --stage 0 \
      	$gmmdir/graph $data_dir/${dset}${subfix} $dnndir/decode_${dset} || exit 1;
   fi
 done
fi
# print results:
# grep WER $dnndir/decode_*/scoring_kaldi/best_wer


# all done...

