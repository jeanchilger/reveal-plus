#!/usr/bin/perl

my $infile = $ARGV[0];
my $train_file = $ARGV[1];
my $featnum = $ARGV[2];
my $outfile = $ARGV[3];

my @bins;
my @vals;
my $linecount = 1;

open (F1, $train_file) || die ("Could not open $file!");
open (F3, ">$outfile") || die ("Could not open $file!");
#open (F4, ">>$outfile.weka") || die ("Could not open $file!");
$maxbins=0;

for ($i=1; $i<=$featnum; $i++) {
    if ( -e "$infile-$i-0LL.hist" ) {
        open (F2, "$infile-$i-0LL.hist");
        $j=0;
        $bins[$i][$j] = 0.0;
        #print "bins[$i][$j] = $bins[$i][$j] \n";
        $j++;
        $line = <F2>;
        while ($line = <F2>) {
            $line = <F2>; 
            if ($line ne "") {
                ($value,$density) = split ' ', $line;
                $bins[$i][$j] = $value;
                #print "bins[$i][$j] = $bins[$i][$j] \n";
                $j++;
                if ($maxbins < $j) { $maxbins = $j; }
            }
        }
    }
    else {
        print "$infile-$i-0LL.hist \n";
        $bins[$i][0]=-1;
    }
    close(F2);
}

while ($line = <F1>) {
    (@vals) = split ', ', $line;
    $class = $vals[@vals-1];
    $class =~ s/\n/ /;
    print F3 "$linecount CLASS=$class";
#    print F4 "$linecount CLASS=$class";
    for ($i=1; $i<=$featnum; $i++) {
      if ($bins[$i][0]>-1) {
          for ($j=1; $j<$maxbins; $j++) {
            if ($vals[$i-1] <= $bins[$i][$j] || ($vals[$i-1] == 1 && $bins[$i][$j] == 1)) {
              print F3 "w[$i]=$bins[$i][$j-1]-$bins[$i][$j] ";
#             print F4 "w[$i]=$bins[$i][$j-1]-$bins[$i][$j] ";
              last;
            }
            else {
              if ($j == ($maxbins - 1) && $bins[$i][$j] < 1) {
                print F3 "w[$i]=$bins[$i][$j]-1 ";
                #print "-1 inserted \n";
              }
            }
          }
     }
    }

    print F3 "\n";
#    print F4 "\n";
    $linecount++;
}

close(F1);
close(F3);
close(F4);



