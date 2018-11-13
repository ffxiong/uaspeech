#!/bin/bash
#
# Author: Feifei Xiong @ SPandH Sheffield

# Begin configuration section.
cleanup=true
settyp="ctl"
stage=0
nj=4
# End configuration section

echo "$0 $@"  # Print the command line for logging

. ./utils/parse_options.sh  # accept options.. you can run this run.sh with the
. ./path.sh

if [ $# != 3 ]; then
  echo "Usage: local/prepare_uaspeech_data.sh [options] <flist-dir> <audio-dir> <output-dir>"
  echo "main options (for others, see top of script file)"              
  echo "  --nj 		<nj>            # number of parallel jobs"
  echo "  --settyp 	<ctl|dys> 	# type of setup."
  exit 1;
fi

fdir=$1
sdir=$2
dir=$3
tmpdir=$dir/$settyp    # will be deleted if cleanup
mkdir -p $tmpdir

# uaspeech mlf
mlfdir=$fdir/mlf
if [[ $settyp == "ctl" ]]; then
 spksets="CF02 CF03 CF04 CF05 CM01 CM04 CM05 CM06 CM08 CM09 CM10 CM12 CM13"
 spkdir=$sdir/control
fi
if [[ $settyp == "dys" ]]; then
 spksets="F02 F03 F04 F05 M01 M04 M05 M07 M08 M09 M10 M11 M12 M14 M16"
 spkdir=$sdir
fi

if [ $stage -le 0 ]; then
 # from the provided .mlf to wav list and text
 # each speaker 
 for x in $spksets; do
   mlf=$mlfdir/$x/${x}_word.mlf
   [ ! -f $mlf ] && echo "prepare data: no such file $mlf, check your flist dir!" && exit 1;
  
   # generate text file
   textallfil=$tmpdir/${x}.textall
   for line in $(cat $mlf); do 
     if [[ $line == *$x* ]]; then
      line1=$(echo "$line"  | cut -d'/' -f2)
      line2=$(echo "$line1" | cut -d'.' -f1)
      linenext=$(grep -A1 $line $mlf | tail -n 1)
      echo $line2 $linenext
     fi
   done > $textallfil
   #nline=`cat $textallfil | wc -l`

   # segments
   segments=$mlfdir/$x/${x}.segments
   [ ! -f $segments ] && echo "prepare data: no such file $segments, check your flist dir!" && exit 1;
   segmentsfil=$tmpdir/${x}.segments
   cp $segments $segmentsfil

   # compare segments and text to update text
   textfil=$tmpdir/${x}.text
   awk '{print $1, $2}' < $segmentsfil > $tmpdir/${x}.utt2rec
   utils/apply_map.pl -f 2 $textallfil < $tmpdir/${x}.utt2rec >$textfil

   # text to wav.scp
   audiodir=$spkdir/$x/
   cut -d " " -f 2 $segmentsfil | sed -e "s|^|${audiodir}|" -e "s|$|.wav|" > $tmpdir/${x}.wav.lst || exit 1;
   perl -e 'while(<>){
     m:^\S+/(\w+)\w*\.wav$: || die "Bad line $_";
     $id = $1;
     print "$id $_";
   }' < $tmpdir/${x}.wav.lst | sort > $tmpdir/${x}.wav.scp || exit 1;

   # utt2spk
   cut -d " " -f 1 $segmentsfil | sed -e "s|^\(\([^_]\+\)_.*\)|\1 \2|g" > $tmpdir/${x}.utt2spk || exit 1;
  
   echo " ... generation for $x with length $nline done"
 done
fi


if [ $stage -le 1 ]; then
 # combine each speaker information
 tralldir=$dir/train_${settyp}all
 mkdir -p $tralldir
 for dd in wav.scp text utt2spk segments; do
   for x in $spksets; do
     cat ${tmpdir}/${x}.${dd}
   done | sort > ${tralldir}/${dd}
 done
  
 cat $tralldir/utt2spk | utils/utt2spk_to_spk2utt.pl > $tralldir/spk2utt || exit 1;
 nline=`cat $tralldir/text | wc -l`
 
 echo " ... generation all in $tralldir with length $nline done"
fi 


if [ $stage -le 2 ]; then
 # split B1 B3 as training and B2 as test  
 for ss in train test; do
   ddir=$dir/${ss}_${settyp}
   mkdir -p $ddir
   blinfo='_B1_|_B3_'
   if [[ "$ss" == "test" ]]; then
    blinfo='_B2_'
   fi

   for ff in wav.scp text utt2spk segments; do
     grep -E ${blinfo} $dir/train_${settyp}all/${ff} > ${ddir}/${ff} || exit 1
   done

   cat $ddir/utt2spk | utils/utt2spk_to_spk2utt.pl > $ddir/spk2utt || exit 1;
   utils/data/get_utt2dur.sh --nj $nj $ddir
   utils/validate_data_dir.sh --no-feats $ddir || exit 1
   nline=`cat $ddir/text | wc -l`

   echo " ... generation $ss in $ddir with length $nline done"
 done
fi


$cleanup && rm -rf $tmpdir
echo "Data preparation succeeded in $dir" 

exit 0;
