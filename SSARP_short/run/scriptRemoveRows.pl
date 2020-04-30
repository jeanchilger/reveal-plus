#!/usr/bin/perl

# my $infile = $ARGV[0];
my $train_file = $ARGV[0];
# my $featnum = $ARGV[2];
my $outfile = $ARGV[1];
my $outfile2 = $ARGV[2];
my $vez=$ARGV[3];
my $linecount = 1;

open (F1, $train_file) || die ("Could not open $train_file!");
open (F2, ">$outfile") || die ("Could not open $outfile!");
open (F3, ">$outfile2") || die ("Could not open $outfile2!");

while ($line = <F1>) {
    @vals = split(/ /, $line,2);
    
    if($vals[0] > 1000*$vez){
#       print "-----------$vals[0]-----------$vez\n";
      $j=$vals[0]-1000*$vez;
      print F2 "$j $vals[1]";
	print F3 "$line";

    }
#    else{
#      print F3 "$line";
#    }

}
