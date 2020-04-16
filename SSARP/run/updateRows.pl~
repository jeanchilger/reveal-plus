#!/usr/bin/perl

# my $infile = $ARGV[0];
my $train_file = $ARGV[0];
# my $featnum = $ARGV[2];
my $outfile = $ARGV[1];
my $vez=$ARGV[2];
my $linecount = 1;

open (F1, $train_file) || die ("Could not open $file!");
open (F3, ">$outfile") || die ("Could not open $file!");

while ($line = <F1>) {
    @vals = split(/ /, $line,2);
    $j=$vals[0]+1000*$vez;
    $class = $vals[0]+1000;
    $class =~ s/\n/ /;
    print F3 "$j $vals[1]";
}