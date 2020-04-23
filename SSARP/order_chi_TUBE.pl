#!/usr/bin/perl
#i
$file_chi = $ARGV[0];
#$file_chi = 'ChiSquared_Attributes.txt';

open (F2, $file_chi) || die ("Could not open $file!");

$linecount=0;

#open (F3, ">>perl_chi.txt");

while ($line = <F2>)
{
  (@field) = split ',', $line;
  
  for ($i=0; $i<@field; $i++) {
    if ($i>1) {
    #  print "$field[$i]\n";
      $value[$field[$i]] = $value[$field[$i]] + (1/log (10 * $i));
      #$value[$field[$i]] = $value[$field[$i]] + ($field[$i]/($field[$i] * log $i));
    }
    else {
      $value[$field[$i]] = $value[$field[$i]] + 1;
    }
    #print "$value[5]\n";
  }
  
  $linecount++;

}

#for ($i=1; $i<=@field; $i++) {
#  if ($value[$i] > 0) { print "$i = $value[$i]\n" };
#  }

#print "\n";
  
@sorted = sort ascend @value;

for ($i=1; $i<=@field; $i++) {           
  $j=$value[$i]/$sorted[0];                                                                                                               
  if ($value[$i] > 0) { print "$j\n" };                                                                                                    
  }                            

for ($i=0; $i<=@field; $i++) {
  for ($j=1; $j<=@field; $j++) {
    if ( $sorted[$i]==$value[$j] ) {
      #print "$j = $sorted[$i]\n";
      #if ($j!=47 && ($j<6 || $j>10)) {
	print "$j " ;
      #}
    }
  }
}  


#print "$linecount\n";
close (F2);

sub ascend {
  $b <=> $a;
  }
