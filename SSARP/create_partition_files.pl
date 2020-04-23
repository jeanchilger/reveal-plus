#!/usr/bin/perl
#
use FileHandle;
$file_lac = $ARGV[0];
$file_features = $ARGV[1];
$numfiles = $ARGV[2];

if ($file_lac eq "") { $file_lac = '../lac_traintest_train.txt'; }
if ($file_features eq "") { $file_features = '../features_traintest.txt'; }
if ($numfiles eq "") { $numfiles=5 };

$numfeatures=64;

open (F1, $file_features) || die ("Could not open $file_features!");
open (F2, $file_lac) || die ("Could not open $file_lac!");

$linecount=0;

$linecount=0;
my @filehandles;

$line_features = <F1>;
(@features) = split ' ', $line_features;

#for($i=0; $i<=39; $i++) { print "$features[$i]\n" };

for ($i=1; $i<$numfiles+1; $i++) {
  $filehandles[$i] = new FileHandle;
  $filehandles[$i]->open(">>features_$i\_train.txt");
}

$started=0;

while ($line = <F2>)
{
  
  ($class,$qid,@fields) = split ' ', $line;
  
  if ($started==0) {
    for ($i=0; $i<@fields; $i++) {
      (@feat1) = split '=', $fields[$i];
      $feat = substr $feat1[0], 2, -1;
      $realfeatures{$feat}=$i;
    }
    $started=1;
  }
  
  $linecount++;
  
  for ($i=1; $i<$numfiles+1; $i++) {
    $filehandles[$i]->print("$class $qid ");
  }
 
  $k=1;
  for ($j=0; $j<$numfeatures; $j++) {
    
    if (exists $realfeatures{$features[($j)]}) {
      $i=$k % $numfiles;
      if ($i==0) { $i=$numfiles };
      $filehandles[$i]->print("$fields[$realfeatures{$features[($j)]}] ");
      $k++;
        
    }
  #  else:
   #     print "{$features[($j)]}\n";
  }
  
  for ($i=1; $i<$numfiles+1; $i++) {
    $filehandles[$i]->print("\n");
  }

}

#print "$linecount\n";
for ($i=1; $i<$numfiles+1; $i++) {
  $filehandles[$i]->close();
}
