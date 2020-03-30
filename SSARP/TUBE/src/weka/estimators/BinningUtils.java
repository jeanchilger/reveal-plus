//17.11 lap
/*
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 2 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program; if not, write to the Free Software
 *    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 *    BinningUtils.java
 *    Copyright (C) 2004 Gabi Schmidberger
 *
 */

package weka.estimators;

import java.io.FileOutputStream;
import java.io.PrintWriter;
import java.util.Vector;

import weka.core.Instances;
import weka.core.Utils;
import weka.filters.unsupervised.attribute.Bin;
import weka.filters.unsupervised.attribute.CutInfo;
 
/** 
 <!-- globalinfo-start -->
 * Contains static utility functions for Estimators.<p>
 <!-- globalinfo-leftEnd -->
 *
 * @author Gabi Schmidberger (gabi dot schmidberger at gmail dot com)
 * @version $Revision: 1.0 $
 */
public class BinningUtils extends EstimatorUtils{
 
   
  /**
   * Writes data to file that can be used to plot a histogram.
   * Filename is aprameter f + ".hist".
   *
   *@param f string to build filename
   *@param bins vector of bins
   *@param identicalValue value, is NaN if not all values were identical
   */
  public static void writeHistogram(String f, Vector bins, 
				    double min, double max,
				    double factor) throws Exception {
    double identicalValue = Double.NaN;
    if (min == max) {
      identicalValue = min;
    }
    //Oops.pln("writeHistogram "+ f);
    writeHistogram(f, bins, identicalValue, factor);
  }

  // all values are identical
  /**
   * Writes data to file that can be used to plot a histogram.
   * Filename is aprameter f + ".hist".
   *
   *@param f string to build filename
   *@param bins vector of bins
   *@param identicalValue value, is NaN if not all values were identical
   *@param factor 
   */
  public static void writeHistogram(String f, Vector bins, 
				    double identicalValue,
				    double factor) throws Exception {

    //Oops.pln("writehisto");
   // all values are identical
    if (!Double.isNaN(identicalValue)) {
      double veryLittle = 0.000001;
      double min = identicalValue - veryLittle; 
      double max = identicalValue + veryLittle;
      Vector helpBins = new Vector(1);
//  public Bin(int splitDepth,
// 	     double totalNum, double totalLen,
// 	     double num, int leftBegin, int leftEnd,
// 	     double min, boolean minIn, double max, boolean maxIn,
// 	     double entropy, double alpha) {
      Bin bin = new Bin(0, 
			1.0, 1.0, veryLittle * 2.0,
			1.0, 0, 0,
			min, true, max, true, 
			0.0, 1.0);
      helpBins.add(bin);
      writeHistogram(f, helpBins, factor);
    } else {
      writeHistogram(f, bins, factor);
    }
  }

  /**
   * Writes data to file that can be used to plot a histogram.
   * Filename is aprameter f + ".hist".
   *
   *@param f string to build filename
   *@param bins vector of bins
   */

  public static void writeHistogram(String f, Vector bins,
				    double factor) throws Exception {

    PrintWriter output = null;
    Bin bin = null;
    StringBuffer text = new StringBuffer("");
    
    if (f.length() != 0) {
      // add extension to filename
      String name = f + ".hist";
      output = new PrintWriter(new FileOutputStream(name));
    } else {
      return;
    }
    
    if (bins == null) return;
    if (bins.size() == 0) return;
    // first bin
    bin = (Bin) bins.elementAt(0);
    try {
      text.append("" + bin.getMinValue()+" "+0.0+" \n");
      
      for (int i = 0; i < bins.size(); i++) {
	double density = bin.getDensity() * factor;
	//Oops.pln("i "+i+" density "+density);
	text.append(""+bin.getMinValue()+" "+density+" \n");
	text.append(""+bin.getMaxValue()+" "+density+" \n");
	if (i + 1 < bins.size()) {
	  bin = (Bin) bins.elementAt(i + 1);
	}
      }
      text.append("" + bin.getMaxValue()+" "+0.0+" \n");
      // last bin
    } catch (Exception ex) {
      ex.printStackTrace();
      System.out.println(ex.getMessage());
    }
    output.println(text.toString());    

    // close output
    if (output != null) {
      output.close();
    }
  }

  /**
   * Writes data to file that can be used to plot a histogram.
   * Filename is aprameter f + ".hist".
   *
   *@param f string to build filename
   *@param bins vector of bins
   *@param factor 
   */

  public static void writeTestHistogram(String f, Vector bins, 
					double factor) throws Exception {

    PrintWriter output = null;
    Bin bin = null;
    StringBuffer text = new StringBuffer("");
    
    if (f.length() != 0) {
      // add extension to filename
      String name = f + ".TEST.hist";
      output = new PrintWriter(new FileOutputStream(name));
    } else {
      return;
    }
    
    if (bins == null) return;
    if (bins.size() == 0) return;
    // first bin
    bin = (Bin) bins.elementAt(0);
    try {
      text.append("" + bin.getMinValue()+" "+0.0+" \n");
      
      for (int i = 0; i < bins.size(); i++) {
	bin = (Bin) bins.elementAt(i);
	double density = bin.getInstDensity() * factor;
	text.append(""+bin.getMinValue()+" "+density+" \n");
	text.append(""+bin.getMaxValue()+" "+density+" \n");
      }
      text.append("" + bin.getMaxValue()+" "+0.0+" \n");
      // last bin
    } catch (Exception ex) {
      ex.printStackTrace();
      System.out.println(ex.getMessage());
    }
    output.println(text.toString());    

    // close output
    if (output != null) {
      output.close();
    }
  }

  /**
   * Output of an n points of a density curve.
   * Filename is parameter f + ".curv".
   *
   *@param f string to build filename
   *@param bins vector of bins
   */

  public static void writeCurve(String f, Estimator est, 
				double min, double max,
				int numPoints) throws Exception {

    PrintWriter output = null;
    StringBuffer text = new StringBuffer("");
    
    if (f.length() != 0) {
      // add attribute indexnumber to filename and extension .hist
      String name = f + ".curv";
      output = new PrintWriter(new FileOutputStream(name));
    } else {
      return;
    }

    double diff = (max - min) / ((double)numPoints - 1.0);
    try {
      text.append("" + min + " " + est.getProbability(min) + " \n");

      for (double value = min + diff; value < max; value += diff) {
	text.append("" + value + " " + est.getProbability(value) + " \n");
      }
      text.append("" + max + " " + est.getProbability(max) + " \n");
    } catch (Exception ex) {
      ex.printStackTrace();
      System.out.println(ex.getMessage());
    }
    output.println(text.toString());    

    // close output
    if (output != null) {
      output.close();
    }
  }

  /**
   * Output of an n points of a density curve.
   * Filename is parameter f + ".curv".
   *
   *@param f string to build filename
   *@param bins vector of bins
   */
  public static void writeCurve(String f, Estimator est, 
				Estimator classEst,
				double classIndex,
				double min, double max,
				int numPoints) throws Exception {

    PrintWriter output = null;
    StringBuffer text = new StringBuffer("");
    
    if (f.length() != 0) {
      // add attribute indexnumber to filename and extension .hist
      String name = f + ".curv";
      output = new PrintWriter(new FileOutputStream(name));
    } else {
      return;
    }

    double diff = (max - min) / ((double)numPoints - 1.0);
    try {
      text.append("" + min + " " + 
		  est.getProbability(min) * classEst.getProbability(classIndex)
		  + " \n");

      for (double value = min + diff; value < max; value += diff) {
	text.append("" + value + " " + 
		    est.getProbability(value) * classEst.getProbability(classIndex)
		    + " \n");
      }
      text.append("" + max + " " +
		  est.getProbability(max) * classEst.getProbability(classIndex)
		  + " \n");
    } catch (Exception ex) {
      ex.printStackTrace();
      System.out.println(ex.getMessage());
    }
    output.println(text.toString());    

    // close output
    if (output != null) {
      output.close();
    }
  }

 /**
   * Get a probability estimate for a value
   *
   * @param value the value to estimate the probability of
   * @return the estimated probability of the supplied value
   */
  public static double getProbability(double value, Vector bins,
				      double min, double max) {
    
    // special case; estimator was trained with identical values only
    //Oops.pln("min "+min+" max "+max);
    if (min == max) { 
      return 1.0;
    } 

    Bin bin = null;
    double prob = 0.0;

    // no data therefore no bins, probability is 0.0 
    if (bins == null) return prob;

    // outside the boundaries the probability is 0.0
    if (value < min) { 
      return 0.0;
    } else {
      if (value > max) { 
	return 0.0;
      } else {
        bin = findBin(value, bins, ((Bin)bins.elementAt(0)).getMinValue(), 
		      ((Bin)bins.elementAt(bins.size() - 1)).getMaxValue());
      }
    }
    try {
      prob = bin.getDensity();
    } catch (Exception ex) {
      ex.printStackTrace();
      System.out.println(ex.getMessage());
    }
    return prob;
  }

  /**
   * Return a copy of the bins 
   *
   * @param  oldBins the bins that should be copied
   * @return a copy of the bins
   */
  public static Vector copyBins (Vector oldBins) {
    Vector bins = null;
    bins = new Vector();

    for (int i = 0; i < oldBins.size(); i++) {
      Bin bin = (Bin)oldBins.elementAt(i);
      bins.add(bin);
    }
    
    return bins;
  }  

  /**
   * Get a probability estimate for a value
   *
   * @param data the value to estimate the probability of
   * @return the estimated probability of the supplied value
   */
  public static double getDensity(double data, Vector bins) {
    
    Bin bin;
    double prob = 0.0;

    // no data therefore no bins, probability is 0.0 
    if (bins == null) return prob;

    //data = round(data);

    // outside the boundaries the density is 0.0
    if (data < ((Bin)bins.elementAt(0)).getMinValue()) { 
      // bin = (Bin)m_bins.elementAt(0);
      //Oops.pln("< RANGE");
      return 0.0;
    } else {
      if (data > ((Bin)bins.elementAt(bins.size() - 1)).getMaxValue()) { 
	// bin = (Bin)m_bins.elementAt(m_bins.size() - 1);
	//Oops.pln("> RANGE");
	return 0.0;
      } else {

	// find the right bin
        bin = findBin(data, bins, ((Bin)bins.elementAt(0)).getMinValue(), 
		      ((Bin)bins.elementAt(bins.size() - 1)).getMaxValue());
      }
    }
    try {
      prob = bin.getDensity();
    } catch (Exception ex) {
      ex.printStackTrace();
      System.out.println(ex.getMessage());
    }
    return prob;
  }

  /**
   * Finds the bin a value fits into.
   * @param value the value to find the bin for
   * @param min the minimal value of this attribute 
   * @param max the maximal value of this attribute 
   * @return the bin the value fits into
   */
  public static Bin findBin(double value, Vector bins, double min, double max) {
    //Oops.p("findBin ");
    Bin bin = null;
    int bini = 0;
    double len = max - min;
    double part = len / bins.size();
    int index = (int)Math.floor((value - min) / part);
    if (index >=  bins.size()) {
      index =  bins.size() - 1;
    }
    bin = (Bin)bins.elementAt(index);
    int diff = 0;
    if (value < bin.getMinValue() ||
	(value == bin.getMinValue() && !bin.getMinIncl())) {diff = -1;}
    if (value > bin.getMaxValue() ||
	(value == bin.getMaxValue() && !bin.getMaxIncl())) {diff = 1;}
    //Oops.p(" index = "+index+" diff = "+diff);
    index = index + diff;
    if (diff != 0) {
      bin = (Bin) bins.elementAt(index);
      
//       boolean in1 = (value > bin.getMinValue() && value < bin.getMaxValue()); 
//       boolean in2 = (value == bin.getMinValue() && bin.getMinIncl());
//       boolean in3 = (value == bin.getMaxValue() && bin.getMaxIncl());
//       Oops.pln(" "+in1+"/"+in2+"/"+in3+" ");
      while (!((value > bin.getMinValue() && value < bin.getMaxValue()) || 
	       (value == bin.getMinValue() && bin.getMinIncl()) ||
	       (value == bin.getMaxValue() && bin.getMaxIncl()))) {
	// not yet the right bin found
	index = index + diff;
	bin = (Bin) bins.elementAt(index);
		bini = index;
// 	 in1 = (value > bin.getMinValue() && value < bin.getMaxValue());
// 	 Oops.pln("value "+value+" Min "+bin.getMinValue()+" Max "+bin.getMaxValue());
// 	 in2 = (value == bin.getMinValue() && bin.getMinIncl());
// 	 in3 = (value == bin.getMaxValue() && bin.getMaxIncl());
// 	Oops.pln(" "+in1+"/"+in2+"/"+in3+" ");
      }
    }
    //Oops.pln("#bin id "+bini);
    return bin;
  }

//   /**
//    * Finds the bin a value fits into.
//    * @param value the value to find the bin for
//    * @param min the minimal value of this attribute 
//    * @param max the maximal value of this attribute 
//    * @return the bin the value fits into
//    */
//   private static Bin findBin(double value, Vector bins, double min, double max) {
//     Bin bin = null;
//     int bini = 0;
//     double len = max - min;
//     double part = len / bins.size();
//     int index = (int)Math.floor((value - min) / part);
//     if (index >=  bins.size()) {
//       index =  bins.size() - 1;
//     }
//     bin = (Bin)bins.elementAt(index);
//     int diff = 0;
//     if (value < bin.getMinValue() ||
// 	(value == bin.getMinValue() && !bin.getMinIncl())) {diff = -1;}
//     if (value > bin.getMaxValue() ||
// 	(value == bin.getMaxValue() && !bin.getMaxIncl())) {diff = 1;}
//     index = index + diff;
//     if (diff != 0) {
//       bin = (Bin) bins.elementAt(index);
      
// //       boolean in1 = (value > bin.getMinValue() && value < bin.getMaxValue()); 
// //       boolean in2 = (value == bin.getMinValue() && bin.getMinIncl());
// //       boolean in3 = (value == bin.getMaxValue() && bin.getMaxIncl());
// //       Oops.pln(" "+in1+"/"+in2+"/"+in3+" ");
//       while (!((value > bin.getMinValue() && value < bin.getMaxValue()) || 
// 	       (value == bin.getMinValue() && bin.getMinIncl()) ||
// 	       (value == bin.getMaxValue() && bin.getMaxIncl()))) {
// 	// not yet the right bin found
// 	index = index + diff;
// 	bin = (Bin) bins.elementAt(index);
// 		bini = index;
// // 	 in1 = (value > bin.getMinValue() && value < bin.getMaxValue());
// // 	 Oops.pln("value "+value+" Min "+bin.getMinValue()+" Max "+bin.getMaxValue());
// // 	 in2 = (value == bin.getMinValue() && bin.getMinIncl());
// // 	 in3 = (value == bin.getMaxValue() && bin.getMaxIncl());
// // 	Oops.pln(" "+in1+"/"+in2+"/"+in3+" ");
//       }
//     }
//     //Oops.pln("#bin id "+bini);
//     return bin;
//   }

  /**
   * Transforms the cutpoints to cutinfo.
   * @param cutPoints the array of cutpoints
   * @param cutAndLeft info about borders, the same for all cutpoints
   * @return the cutinfo object
   */
  public static CutInfo cutPointsToCutInfo(double [] cutPoints, boolean cutAndLeft) {
    // no cutpoints here
    if (cutPoints == null) return null;
    boolean [] cal = new boolean[cutPoints.length];
    for (int i = 0; i < cal.length; i++) {
      cal[i] = cutAndLeft;
    }

    // transform bins into cutpoints
    CutInfo info = new CutInfo(cutPoints.length);
    for (int i = 0; i < info.numCutPoints(); i++) {
      info.m_cutPoints[i] = cutPoints[i];
      info.m_cutAndLeft[i] = cal[i];
      }
    return info;
  }

  /**
   * Transforms the bins to cutinfo.
   * @param bins the vecor of bins
   * @return the list of cutpoints
   */
  public static CutInfo binsToCutInfo(Vector bins) {

    // no cutpoints here
    if (bins == null) return null;

    // transform bins into cutpoints
    CutInfo info = new CutInfo(bins.size() - 1);

    for (int i = 0; i < info.numCutPoints(); i++) {
      info.m_cutPoints[i] = ((Bin)bins.elementAt(i)).getMaxValue();
      info.m_cutAndLeft[i] = ((Bin)bins.elementAt(i)).getMaxIncl();
      }
    return info;
  }


  /**
   * Use the given cutpoints and form bins and fill the instances into these bins. 
   * @param cutpoint all the cutpoints 
   * @param cutAndLeft flag if true the bin to the left is closed
   * @param test a set of instances, that are in the bins
   * @param attrIndex the index of the attribute the loglikelihood is asked for
   * @param min minimal value of the attributes value range
   * @param max maximal value of the attributes value range
   * @return the bins for the given attribute
   */
  public static Vector cutInfoToBins(CutInfo info, Instances inst, 
				     int attrIndex, double alpha) {
    // find min and max 
    double [] minMax = new double[2];

    try {
      int num = BinningUtils.getMinMax(inst, attrIndex, minMax);
    } catch (Exception ex) {
      ex.printStackTrace();
      System.out.println(ex.getMessage());
    }
    double minValue = minMax[0];
    double maxValue = minMax[1];

    Vector bins = cutPointsToBins(info.m_cutPoints, info.m_cutAndLeft,
				  inst, attrIndex, 
				  minValue, maxValue, alpha);
    return bins;
  }

  /**
   * Use the given cutpoints and form bins and fill the instances into these bins. 
   * @param cutpoint all the cutpoints 
   * @param cutAndLeft flag if true the bin to the left is closed
   * @param test a set of instances, that are in the bins
   * @param attrIndex the index of the attribute the loglikelihood is asked for
   * @param min minimal value of the attributes value range
   * @param max maximal value of the attributes value range
   * @return the bins for the given attribute
   */
  public static Vector cutPointsToBins(double [] cutPoints, Instances inst, 
				       int attrIndex, double alpha) {
    boolean [] cutAndLeft = null;
    if (!((cutPoints == null) || (cutPoints.length == 0))) {
      cutAndLeft = new boolean[cutPoints.length];
      for (int i = 0; i < cutAndLeft.length; i++) {
	cutAndLeft[i] = true;
      }
    }
    // find min and max 
    double [] minMax = new double[2];

    try {
      int num = BinningUtils.getMinMax(inst, attrIndex, minMax);
    } catch (Exception ex) {
      ex.printStackTrace();
      System.out.println(ex.getMessage());
    }
    double minValue = minMax[0];
    double maxValue = minMax[1];

    //Oops.pln("cutPointsToBins -minValue "+minValue+" maxValue "+maxValue);
    // handles the no-cutpoints-case also
    Vector bins = cutPointsToBins(cutPoints, cutAndLeft,
				  inst, attrIndex, 
				  minValue, maxValue, alpha);
    return bins;
  }

  /**
   * Use the given cutpoints and form bins and fill the instances into these bins. 
   * @param cutpoint all the cutpoints 
   * @param cutAndLeft flag if true the bin to the left is closed
   * @param test a set of instances, that are in the bins
   * @param attrIndex the index of the attribute the loglikelihood is asked for
   * @param min minimal value of the attributes value range
   * @param max maximal value of the attributes value range
   * @return the bins for the given attribute
   */
  public static Vector cutPointsToBins(double [] cutPoint, boolean [] cutAndLeft,
				Instances test, int attrIndex, 
				double min, double max, double alpha) {

    //Oops.pln("min "+min+" max "+max+" numinst "+ test.numInstances());
    //Oops.pln("cutPointsToBins");
    Bin bin = null;
    double totalLen = max - min;
    int totalNum = test.numInstances();
 
    Vector bins = new Vector();
    boolean onlyOne = false;
    if (cutPoint == null) { onlyOne = true; }
    if (onlyOne || (cutPoint.length == 0)) {
      // only one bin
      bin = new Bin(1, 
		    totalNum, totalNum, totalLen,
		    totalNum, 0, 0,
		    // num is totalnum, the other two are irrelevant
		    min, true, max, true,
		    0.0, alpha);
    bins.add(bin);
    return bins;
    }

    // first bin
    bin = new Bin(1, 
		  totalNum, totalNum, totalLen,
		  0, 0, 0,
		  // num not yet given, the other two are irrelevant
		  min, true, cutPoint[0], cutAndLeft[0],
		  0.0, alpha);
    bins.add(bin);
    for (int i = 1; i < cutPoint.length; i++) {
//   public Bin(int splitDepth,
// 	     double totalNum, double totalLen,
// 	     double num, int leftBegin, int leftEnd,
// 	     double min, boolean minIn, double max, boolean maxIn,
// 	     double entropy, double alpha) {
      bin = new Bin(1, 
		    totalNum, totalNum, totalLen,
		    0, 0, 0,
		    // num not yet given, the other two are irrelevant
		    cutPoint[i - 1], !cutAndLeft[i - 1], cutPoint[i], cutAndLeft[i],
		    0.0, alpha);
      bins.add(bin);
    }    
    bin = new Bin(1, 
		  totalNum, totalNum, totalLen,
		  0, 0, 0,
		  // num not yet given, the other two are irrelevant
		  cutPoint[cutPoint.length - 1], cutAndLeft[cutPoint.length - 1], 
		  max, true,
		  0.0, alpha);
    bins.add(bin);

    // fill instances into bins
    nextInstance:
    for (int i = 0; i < test.numInstances(); i++) {
      //Oops.pln("i "+i+" instance "+test.instance(i));
      if (test.instance(i).isMissing(attrIndex)) {
	continue;
      }
      double value = test.instance(i).value(attrIndex);
 
      // test if in first bin     
     if ((value < cutPoint[0]) || 
	  ((value == cutPoint[0]) && cutAndLeft[0])) {
	((Bin)bins.elementAt(0)).addBorderInstance(true, value);
	((Bin)bins.elementAt(0)).addWeight();
	//Oops.pln("#"+value +" into bin 0");
	continue nextInstance;
      }
      // find bin for the current value
      for (int j = 1; j < bins.size() - 1; j++) {
	if ((value < cutPoint[j]) || 
	    ((value == cutPoint[j]) && cutAndLeft[j])) {
	  ((Bin)bins.elementAt(j)).addInstance();
	  ((Bin)bins.elementAt(j)).addWeight();
	  //Oops.pln("#"+value +" into bin "+j);
	  continue nextInstance;
	}
      }

      // into the last bin
      ((Bin)bins.elementAt( bins.size() - 1)).addBorderInstance(true, value);
      ((Bin)bins.elementAt( bins.size() - 1)).addWeight();
      int jjj = bins.size() - 1;
      //Oops.pln("#"+value +" into bin " + jjj);
    }
      
    //Oops.pln("cutPointsToBins 2");
    return bins;    
  } 

  /**
   * Get training loglikelihood..
   *@return the train loglikelihood for these bins
   *@exception if computation of loglikelihood (getloglikelihood) doesn't work
   */
  public static double getTrainLoglkFromBins(Vector bins)  
    throws Exception {

    double entropy = 0.0;
    for (int i = 0; i < bins.size(); i++) {
      Bin bin = (Bin)bins.elementAt(i);
	entropy += bin.getWeightLoglikelihood();
    }
    return entropy;
  }


  /**
   * Use Leave-1-out CV loglikelihood to compute loglk.
   *@param bins the bins to compute the leave-1-outLklh from
   *@param numFolds the number of folds = number of instances.
   *@return the leave-1-out CV loglikelihood for these bins
   *@exception if computation of loglikelihood (getloglikelihood) doesn't work
   */
  public static double getLOOCVLoglkFromBins(Vector bins, int numInst)
    throws Exception {
    //Oops.pln("getLOOCVLoglkFromBins");

    int numFolds = numInst;
    double llk = 0.0;

    // test has one instances
    for (int i = 0; i < bins.size(); i++) {
      Bin bin = (Bin)bins.elementAt(i);
      llk += bin.getLOOCVLoglikelihood() * bin.getWeight();
      //Oops.pln("#loocvllk " +bin.getLOOCVLoglikelihood()+" weight "+bin.getWeight());
    }
    llk = llk / numFolds;
    return llk;
  }

  /**
   * Use Leave-1-out CV loglikelihood to compute loglk.
   *@param bins the bins to compute the leave-1-outLklh from
   *@param numFolds the number of folds = number of instances.
   *@return the leave-1-out CV loglikelihood for these bins
   *@exception if computation of loglikelihood (getloglikelihood) doesn't work
   */
  public static double getTrainLOOCVLoglkFromBins(Vector bins, int numInst)  
    throws Exception {

    int numFolds = numInst;
    double llk = 0.0;

    // test has one instances
    for (int i = 0; i < bins.size(); i++) {
      Bin bin = (Bin)bins.elementAt(i);
      llk += bin.getLOOCVLoglikelihood() * bin.getWeight(); 
      llk += bin.getWeightLoglikelihood() * (numInst - bin.getWeight());
      llk = llk / numInst; 
    }
    llk = llk / numFolds;
    return llk;
  }

  /**
   * Returns a dataset that contains of all instances of a certain value
   * for the given attribute.
   * @param data dataset to select the instances from
   * @param index the index of the attribute  
   * @param v the value 
   * @return a subdataset with only instances of one value for the attribute 
   */
  public static Instances getInstancesFromValue(Instances data, int index,
					  double v) {
    Instances workData = new Instances(data, 0);
    for (int i = 0; i < data.numInstances(); i++) {
      if (data.instance(i).value(index) == v) {
	workData.add(data.instance(i));
      }
    } 
    return workData;
  }

  /**
   *
   */
//   public static void diffBins(Vector bins1, Vector bins2) {

//     // get minimum

//     // get x normalizing factor

//     // get y normalizing factor

//     // get cutpoints
//     double [] cut1 = binsToCutpoints(bins1);
//     double [] dens1_L = densityLOfCutpoints(bins1);
//     double [] dens1_R = densityROfCutpoints(bins1);
//     double [] cut2 = binsToCutpoints(bins2);
//     double [] dens2_L = densityLOfCutpoints(bins2);
//     double [] dens2_R = densityROfCutpoints(bins2);

//     // merge cutpoints
//     double [] cutall;
//     boolean [] cutwhich;

//     int i1 = 0;
//     int i2 = 0;
//     int iA = 0;
//     double cutAll [] = new double [cut1.length + cut2.length];
//     double densAll_L [] = new double [dens1.length + dens2.length];
//     double densAll_R [] = new double [dens1.length + dens2.length];
//     boolean [] cutby1;

//     while (i1 < cut1.length && i2 < cut2.length) {
//       if (cut1[i1] <= cut2[i2]) {
// 	cutAll[iA] = cut1[i1];
// 	densAll_L[iA] = dens1_L[i1];
// 	densAll_R[iA] = dens1_R[i1]; i1++; iA++; 
// 	cutby1[iA] = true;
//       } else {
// 	cutAll[iA] = cut2[i2]; 
// 	densAll_L[iA] = dens2_L[i2]; 
// 	densAll_R[iA] = dens2_R[i2]; i2++; iA++;
// 	cutby1[iA] = false;
//       }    
//     }
//     // merge the rest
//     if (i1 < cut1.length()) {
//       while (i1 < cut1.length()) {
// 	cutAll[iA] = cut1[i1]; 
// 	densAll_L[iA] = dens1_L[i1];
// 	densAll_R[iA] = dens1_R[i1]; i1++; iA++; 
// 	cutby1[iA] = true;
//       }
//     }
//     if (i2 < cut2.length()) {
//       while (i2 < cut2.length()) {
// 	cutAll[iA] = cut2[i2]; 
// 	densAll_L[iA] = dens2_L[i2]; 
// 	densAll_R[iA] = dens2_R[i2]; i2++; iA++;
// 	cutby1[iA] = false;
//       }
//     }
//     for (int i = 1; i < cutAll.length; i++) {
//       double add = 0.0;
//       if (cutby1[i] != cutby1[i - 1]) {
//       }
//     }
//   }


  /**
   * Empty the entropy bins from all test instances (not the weights).
   *
   * @param bins the bins for the given attribute
   */
  public static void emptyBins(Vector bins) {
    //Oops.pln("emptyBins");
    if ((bins == null) || (bins.size() == 0)) return;
    for (int j = 0; j < bins.size(); j++) {
      Bin bin = (Bin) bins.elementAt(j);
      bin.emptyBin();
    }
  }

  /**
   * Use the given bins and fill the instances that the bins have been trained for
   * into. 
   * @param bins the bins for the given attribute
   * @param test a set of instances, need not be the one the bins have been build for
   * @param index the index of the relevant attribute 
   */
  public static void fillBinsWithTrain(Vector bins) {
    int nn = 0;
    Bin bin = null;
    if (bins.size() == 0) return;
    //EstimatorUtils.emptyBins(bins);
    for (int j = 0; j < bins.size(); j++) {
      bin = (Bin) bins.elementAt(j);
      //Oops.pln("weight  of bin "+bin.getWeight());
      //Oops.pln("numinst of bin "+bin.getNumInst());
      bin.setNumInst(bin.getWeight());
    }
  }
    
  /**
   * Use the given bins and fill the instances into the given bins. 
   * @param bins the bins for the given attribute
   * @param test a set of instances, need not be the one the bins have been build for
   * @param attrIndex the index of the relevant attribute 
   */
  public static int fillBinsWithTest(Vector bins, Instances test, int attrIndex) {
    
    int nn = 0;
    Bin bin = null;
    BinningUtils.emptyBins(bins);

    if ((bins == null) || (bins.size() == 0)) return test.numInstances();
    double cutPoint;
    boolean cutAndLeft;
    int numInst = 0;

    nextInstance:
    for (int i = 0; i < test.numInstances(); i++) {
      if (test.instance(i).isMissing(attrIndex)) {
	continue;
      }
      numInst++;
      double value = test.instance(i).value(attrIndex);
 
      // test if  bin     
      bin = (Bin) bins.elementAt(0);
      double min = bin.getMinValue();
      if (value < min) { continue;}

      cutPoint = bin.getMaxValue();
      cutAndLeft = bin.getMaxIncl();
      if ((value < cutPoint) || 
	  ((value == cutPoint) && cutAndLeft)) {
	bin.addBorderInstance(true, value);
	//Oops.pln("#"+value +" into bin 0"); nn++;
	continue nextInstance;
      }
      // find bin for the current value
      for (int j = 1; j < bins.size() - 1; j++) {
	bin = (Bin) bins.elementAt(j);
	 cutPoint = bin.getMaxValue();
	cutAndLeft = bin.getMaxIncl();
	if ((value < cutPoint) || 
	    ((value == cutPoint) && cutAndLeft)) {
	  bin.addInstance();
	  //Oops.pln("#"+value +" into bin "+j); nn++;
	  continue nextInstance;
	}
      }

      // into the last bin
      bin = (Bin) bins.elementAt(bins.size() - 1);

      // falls out of last bin
      double max = bin.getMaxValue();
      if (value > max) {continue;}

      bin.addBorderInstance(false, value);
      int jjj = bins.size() - 1;
      //Oops.pln("#"+value +" into bin " + jjj); nn++;
    }
    return numInst;
  } 

  /**
   * Use the given bins and fill the instances into the given bins. 
   * @param bins the bins for the given attribute
   * @param test a set of instances, need not be the one the bins have been build for
   * @param index the index of the relevant attribute 
   */
  public static int fillBins(Vector bins, Instances test, int index) {
    
    int nn = 0;
    Bin bin = null;
    BinningUtils.emptyBins(bins);
    if ((bins == null) || (bins.size() == 0)) return test.numInstances();
    double cutPoint;
    boolean cutAndLeft;
    int numInst = 0;    

    nextInstance:
    for (int i = 0; i < test.numInstances(); i++) {
      if (test.instance(i).isMissing(index)) {
	continue;
      }
      numInst++;
      double value = test.instance(i).value(index);
 
      // test if in first bin     
      bin = (Bin) bins.elementAt(0);
      cutPoint = bin.getMaxValue();
      cutAndLeft = bin.getMaxIncl();
      if ((value < cutPoint) || 
	  ((value == cutPoint) && cutAndLeft)) {
	bin.addBorderInstance(true, value);
	//Oops.pln("#"+value +" into bin 0"); nn++;
	continue nextInstance;
      }
      // find bin for the current value
      for (int j = 1; j < bins.size() - 1; j++) {
	bin = (Bin) bins.elementAt(j);
	 cutPoint = bin.getMaxValue();
	cutAndLeft = bin.getMaxIncl();
	if ((value < cutPoint) || 
	    ((value == cutPoint) && cutAndLeft)) {
	  bin.addInstance();
	  //Oops.pln("#"+value +" into bin "+j); nn++;
	  continue nextInstance;
	}
      }

      // into the last bin
      bin = (Bin) bins.elementAt(bins.size() - 1);
      bin.addBorderInstance(false, value);
      int jjj = bins.size() - 1;
      //Oops.pln("#"+value +" into bin " + jjj); nn++;
    }
    return numInst;
  } 

  /**
   * Use the given bins and fill the instances into the given bins. 
   * @param bins the bins for the given attribute
   * @param test a set of instances, need not be the one the bins have been build for
   * @param index the index of the relevant attribute 
   *
  public static int fillEntropyBins(Vector bins, Instances test, int attrIndex) {
    
    int nn = 0;
    //EntropyBin bin = null;
    BinningUtils.emptyBins(bins);
    int numBins = bins.size();
    if ((bins == null) || (numBins == 0)) return test.numInstances();


    double [] cutPoint = new double[numBins];
    boolean [] maxIncl = new boolean[numBins];

    int testNumInst = 0;    
    double min = ((Bin)bins.elementAt(0)).getMinValue();
    boolean minIncl = ((Bin)bins.elementAt(0)).getMinIncl();
    double max = ((Bin)bins.elementAt(numBins - 1)).getMaxValue();
    boolean endIncl = ((Bin)bins.elementAt(numBins - 1)).getMaxIncl();

    // prepare cutpoints
    for (int i = 0; i < numBins; i++) {
      cutPoint[i] = ((Bin)bins.elementAt(i)).getMaxValue();
      maxIncl[i] = ((Bin)bins.elementAt(i)).getMaxIncl();
    }
   
    nextInstance:
    for (int i = 0; i < test.numInstances(); i++) {
      if (test.instance(i).isMissing(attrIndex)) {
	continue;
      }
      testNumInst++;
      Instance instance = test.instance(i);
      double value = instance.value(attrIndex);

      // is value within the train range?
      if (!((minIncl && value < min) || (!minIncl && value <= min)
	    || (endIncl && value > max) || (!endIncl && value >= max))) {
	
	// test through bins     
	for (int j = 0; j < numBins; j++) {
	  if (value < cutPoint[j] || 
	      (value <= cutPoint[j] && maxIncl[j])) {
	    // add instance to that bin
	    ((EntropyBin)bins.elementAt(j)).addInstance(instance);
	    continue nextInstance;
	  }
	}
      }
    }
    return testNumInst;
  } 

  /**
   * Use the given bins and fill the instances into the given bins. Returns the
   * average loglikelihood, which is the sum of loglikelihoods divided by the 
   * number of instances.
   * @param bins the bins for the given attribute
   * @param test a set of instances, need not be the one the bins have been build for
   * @param index the index of the attribute the loglikelihood is asked for
   */
  public static double getLoglkFromBins(Vector bins, Instances test, int index) {
    //Oops.pln("bins-getloglkfrombins"+bins);

    // fill the bins with instances from test set
    // return number of non missing instances
    int numInst = BinningUtils.fillBinsWithTest(bins, test, index);
    //Oops.pln("bins-afterfill   bins"+bins);

    double loglk = 0.0;
    Bin bin = null;
    
    // compute likelihood
    for (int i = 0; i < bins.size(); i++) {
      bin = (Bin) bins.elementAt(i);
      //Oops.pln("Bin "+i+" has "+bin.getNumInst()+" instances");
      try {
	loglk += bin.getLoglikelihood();
        //Oops.pln("Bin "+i+" has "+bin.getNumInst()+" instances"+loglk);
      } catch (Exception ex) {
	ex.printStackTrace();
	System.out.println(ex.getMessage());
      }
    }

    // divide by number of instances
    if (numInst > 0)
      loglk = loglk / numInst;
    return loglk;
  }

  /**
   * Get the loglikelihood  with the training instances.
   * Average over the number of instances.
   * @param bins the bins for the given attribute
   * @param test a set of instances, need not be the one the bins have been build for
   * @param index the index of the attribute the loglikelihood is asked for
   */
  public static double getWeightLoglkFromBins(Vector bins, int numInstances) {

    double loglk = 0.0;
    Bin bin = null;
    
    // compute likelihood
    for (int i = 0; i < bins.size(); i++) {
      bin = (Bin) bins.elementAt(i);
      try {
	loglk += bin.getWeightLoglikelihood();
      } catch (Exception ex) {
	ex.printStackTrace();
	System.out.println(ex.getMessage());
      }
    }

    // divide by number of instances
    loglk = loglk / numInstances;
    return loglk;
  }


  /**
   * Defines an illegal cut.
   *@param numInst number of instances
   *@param width width of the bin
   */
  public static boolean isIllegalCut(int numInst, double width, double totalLen, double totalNum) {
    return isIllegalCut((double)numInst, width, totalLen, totalNum);
  }

  /**
   * Returns the threshold for an illegal cut.
   *@param totalNum number of all not missing instances
   */
  public static int getIllegalCutThreshhold(double totalNum) {
    double thresh = (((double)totalNum * 0.1) / 100.0) + 1.0;
    int intThresh = (int)(Math.sqrt(thresh));
    // intThresh = 5;
    return intThresh;
  }

  /**
   * Defines an illegal cut.
   *@param numInst number of instances
   *@param width width of the bin
   *@param totalLen length of total range
   *@param totalNum number of all not missing instances
   */
  public static boolean isIllegalCut(double numInst, double width, double totalLen,  double totalNum) {
    int intThresh = getIllegalCutThreshhold(totalNum);
    // Oops.pln("treshhold "+intThresh+" with totalnum "+totalNum);
    if ((numInst <= intThresh) && ((width / totalLen) < 0.001)) {
      return true;
    } else {
      return false;
    }
  }

  /**
   * Returns a string representing the cutpoints
   */
  public static String cutpointsToString(double [] cutPoints, boolean [] cutAndLeft) {
    StringBuffer text = new StringBuffer("");
    if (cutPoints == null) {
      text.append("\n# no cutpoints found - attribute \n"); 
    } else {
      text.append("\n#* "+cutPoints.length+" cutpoint(s) -\n"); 
      for (int i = 0; i < cutPoints.length; i++) {
	text.append("# "+cutPoints[i]+" "); 
	text.append(""+cutAndLeft[i]+"\n");
      }
      text.append("# leftEnd\n");
    }
    return text.toString();
  }

//   /**
//    * Returns a dataset with the distances between two of the instances after sorting them
//    * @param inst the instances to get the distances from
//    * @param attrIndex index of the attribute
//    */
//   public Instances String getDistances(Instances inst, int attrIndex) {

//     Instances workData = new Instances(inst);
//     workdata.sort();
//     int next = 1;
//     for (int i = 0; i < workData.numInstances(); i++) {
//       for (int j = next; 
// 	   j < workData.numInstances() && data.instance(last).isMissing(attrIndex; j++)) 
// 	next = j;
//     }
// diff = 
// 	     }

//     while ((last >= 0) && (data.instance(last).isMissing(attrIndex))) {
//       sp.numMissing++;
//       last--;
//     }


//   }


  /**
   * Tests whether the current estimation object is equal to another
   * estimation object
   *
   * @param binsA the first bins
   * @param binsB the second bins
   * @return true if the two bin vectors are equal
   */
  public static boolean equalBins(Vector binsA, Vector binsB) {
    
    if (binsA == null) {
        //DBO.pln("binsa is null");
        if (binsB != null) return false;
        else return true;
    }
    //if (binsB == null) { DBO.pln("binsab is null"); }
      if (binsA.size() != binsB.size()) {
      return false;
    }
    for (int i = 0; i < binsA.size(); i++) {
      Bin binA = (Bin) binsA.elementAt(i);
      Bin binB = (Bin) binsB.elementAt(i);
      if (!binA.equals(binB)) return false;
    }
    return true;
  }
  
  /**
   * Returns a string representing the bins
   */
  public static String binsToString(Vector bins) {
    int numIllegalCuts = 0;
    StringBuffer text = 
      new StringBuffer(
          "#|    Bin                    |    %    ||  Instances |   %     | Density | Probability | Loglk | loglk/one \n"+
      "#+------------------------------------------------------------++----------------------+-------------------+\n");
    for (int i = 0; i < bins.size(); i++) {
      Bin bin = (Bin) bins.elementAt(i);
      // count number of illegal cuts
      bin.setIllegalCut();
      if (bin.getIllegalCut()) {
        numIllegalCuts++;
      }
      
      text.append("#"+i+": "+bin.toString());
    }
    text.append("# "+numIllegalCuts+" illegal cuts.\n");
    return text.toString();
  } 

  /**
   * Build a representationstring of the estimator of these bins.
   *
   */
  public static String toString(Vector bins, String name) {
    double sum = 0.0;
    StringBuffer result = new StringBuffer("" + name +" Estimator. Counts = ");
    if (bins != null) {
      for (int i = 0; i < bins.size(); i++) {
	Bin bin = (Bin)bins.elementAt(i);
	double num = bin.getNumInst();
	sum += num;
 	result.append(" " + Utils.doubleToString(num, 2));
      }
      result.append("  (Total = " + Utils.doubleToString(sum, 2) + " / "
		    + bins.size()+ " bins).");
      return result.toString();
    } else {
      result.append(" (No Bins).");
    }
  return result.toString();
  }

  /**
   * Build a representationstring of the estimator of these bins.
   *
   */
  public static String toNumAndWidthString(Vector bins) {
    double sum = 0.0;
    StringBuffer result = new StringBuffer("");
    if (bins != null) {
    	int numBins = bins.size();
    	Bin bin = (Bin)bins.elementAt(0);
    	int num = (int)bin.getNumInst();
    	double width = bin.getWidth();
    	result.append(""+num);
    	result.append(" "+width);

    	if (numBins > 1) {
    		for (int i = 1; i < bins.size(); i++) {
    			bin = (Bin)bins.elementAt(i);
    			num = (int)bin.getNumInst();
    			width = bin.getWidth();
    			result.append(" "+num);
    			result.append(" "+width);
    		}
    	}
    }
    return result.toString();
  }
}
