#!/usr/bin/perl

my $infile = $ARGV[0];
my $outfile = $ARGV[1];
my $headfile = $ARGV[2];
my $numfeatures = $ARGV[3];
my $tempfile = 'temp.txt';

my %values;
my @featurenames;
my $linecount = 0;

open (F1, $infile) || die ("Could not open $file!");
open (F2, ">$tempfile");
open (F3, ">$headfile");

while ($line = <F1>) {
  ($id, $class, @features) = split ' ', $line;
  ($class1, $class2) = split '=', $class;
  print  ($id, $class)
  for ($i=0; $i<$numfeatures; $i++) {
    ($feat, $value) = split '=', $features[$i];
    $featurenames[$i] = $feat;
    if (! exists $values{$feat}{$value} ) {
      $values{$feat}{$value} = $value;
    }
    if ($feat ne "") { print F2 "'\\'$value\\'',"; }
  }
  print F2 "$class2\n";
  $linecount++;
  #if ($linecount % 100 == 0) { print "$linecount\n" };
}

$i=1;
print F3 "\@relation weka\n\n";
for ($i=0;$i<$numfeatures;$i++) {
#for $feat (keys %values) {
   #if ($feat != '') {
    if ($featurenames[$i] ne "") {
      print F3 "\@attribute F$featurenames[$i] {'\\'";
      foreach $val (keys %{$values{$featurenames[$i]}}) {
	  print F3 "$val";
	  print F3 '\\\'\',\'\\\'';
      }
      print F3 "\n";
    }
    #$i++;
  #}
}

print F3 "\@attribute class {0,1,2}\n\n\@data\n";

system("cat $headfile $tempfile > $outfile");

#system qq(bash -c ' sed 's/\,'\''\\'\''$/}/' $outfile > $outfile.arff');
