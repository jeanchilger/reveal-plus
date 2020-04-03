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
 *    EqualWidthEstimator.java
 *    Copyright (C) 2004 University of Waikato
 *
 */


package weka.estimators;

import java.util.Enumeration;
import java.util.Vector;

import weka.core.Attribute;
import weka.core.AttributeStats;
import weka.core.Capabilities;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.RevisionUtils;
import weka.core.Utils;
import weka.core.Capabilities.Capability;
import weka.core.Debug.DBO;
 
/** 
 <!-- globalinfo-start -->
 * Density estimator, The range is splitted into equal width intervals.
 * <p>
 <!-- globalinfo-leftEnd -->
 *
 <!-- options-start -->
 <!-- options-leftEnd -->
 * 
 * @author Gabi Schmidberger (gabi dot schmidberger at gmail dot com)
 * @author Len Trigg (trigg@cs.waikato.ac.nz)
 * @author Eibe Frank (eibe@cs.waikato.ac.nz)
 * @version $Revision: 1.0 $
 */
public class EqualWidthEstimator extends BinningEstimator
  implements OptionHandler {
  

  /** list all cutpoints at leftEnd */
  public static int D_RESULTBINS      = 3; // 4

  /** num bins and offset */
  public static int D_NUMANDOFFSET    = 5; // 6 

  /** entropy over all num cuts   */
  public static int D_ENTROPYCURVE    = 6; // 7

  /** list all cutpoints at leftEnd */
  public static int D_HISTOGRAM       = 7; // 8

  /** special case, all values are the same */
  private double m_allIdenticalValue = Double.NaN;

  /** The number of bins to divide the attribute into */
  protected int m_numBins = 10;

  /** Store the current cutpoints */
  protected double [] m_cutPoints = null;

  /** The offset in percent of the bin size */
  protected double m_offset = 0.0;

  /** The level of uniqueness below which integer values are assumed */
  protected double m_intValuesPercent = 100.0;

  /** Find the number of bins using cross-validated entropy. */
  protected boolean m_findNumBins = false;

  /** Find the value of the offset using cross-validated entropy. 
   *  the offset values are 0, 10, 20, 30 .. 90 */
  protected boolean m_findOffset = false;

  /** Find the number of bins using loglikeliestimator. */
  protected boolean m_findNumBinsWithLogLL = false;

  /** The precision of numeric values (= minimum std dev permitted) 
      For example, if the precision is stated to be 0.1, the values in the
      interval (0.25,0.35] are all treated as 0.3. */ //todo
  private double m_Precision;

  /**
   * Returns a string describing this estimator 
   *
   * @return a description of the estimator suitable for
   * displaying in the explorer/experimenter gui
   */
  public String globalInfo() {
    return "An instance filter that discretizes a range of numeric"
      + " attributes in the dataset into nominal attributes."
      + " Discretization is by simple binning. Skips the class"
      + " attribute if set.";
  }
  
  /** Constructor */
  public EqualWidthEstimator() {
    // for the debug output
  }

  /** Another constructor */
  public EqualWidthEstimator(double precision) {
    m_Precision = precision;
    // for the debug output
    dbo.initializeRanges(20);
  }

  /**
   * Returns default capabilities of the classifier.
   *
   * @return      the capabilities of this classifier
   */
  public Capabilities getCapabilities() {
    Capabilities result = super.getCapabilities();
    
    // attributes
    result.enable(Capability.NUMERIC_ATTRIBUTES);
    result.enable(Capability.NOMINAL_CLASS);
    return result;
  }

  /**
   * Sets the value for the offset in percent.
   * @param newValue the new offset value
   */
  public void setOffset(double newValue) {
    m_offset = newValue;
  }

  /**
   * Gets the value for the offset in percent.
   * @return the offset value
   */
  public double getOffset() {
    return m_offset;
  }
  
  /**
   * Gets the default value for the offset in percent.
   * @return the offset value
   */
  public double getOffsetDefault() {
    return 0.0;
  }

  /**
   * Gets an enumeration describing the available options.
   *
   * @return an enumeration of all the available options.
   */
  public Enumeration listOptions() {

    Vector newVector = new Vector(7);

    newVector.addElement(new Option(
              "\tSpecifies the (maximum) number of bins to divide numeric"
	      + " attributes into.\n"
	      + "\t(default = 10)",
              "B", 1, "-B <num>"));

    newVector.addElement(new Option(
              "\tOptimize number of bins using leave-one-out estimate\n"+
	      "\tof estimated entropy (for equal-width discretization).\n"+
	      "\tIf set option -B sets the maximal number of bins.",
              "Y", 0, "-Y"));
    newVector.addElement(new Option(
              "\tOptimize the offset of the bins using leave-one-out estimate\n"+
	      "\tof estimated entropy (for equal-width discretization).\n",
              "Z", 0, "-Z"));
    newVector.addElement(
      new Option(
		 "\tSet the seed for the randomizer\n"+
		 "\tRandomize is used in cross-validation.\n",
		 "S", 1, "-S <num>"));
    
    newVector.addElement(new Option(
              "\tTake number of bins 2x the number of bins found by \n"+
	      "\tthe loglikelihood estimator.\n",
              "L", 0, "-L"));
    newVector.addElement(new Option(
              "\tIf value given smaller than 100 and the percentage of unique values"
	      +" is smaller than the value given the minimum distance is taken as bin width.\n"
	      + "\t(default = 100)",
              "P", 1, "-P <num>"));
    newVector.addElement(new Option(
              "\tSets the offset of the grid to value given in percent of the bin"
	      +" width.\n"
	      + "\t(default = 0)",
              "O", 1, "-O <num>"));
   Enumeration enu = super.listOptions();
   while (enu.hasMoreElements()) {
     newVector.addElement(enu.nextElement());
   }
   return newVector.elements();
  }


  /**
   * Parses the options for this object.<p>
   *
   * @param options the list of options as an array of strings
   * @exception Exception if an option is not supported
   */
  public void setOptions(String[] options) throws Exception {

    super.setOptions(options); 

    setNumBins(10);
 
    setFindNumBins(Utils.getFlag('Y', options));
    setFindOffset(Utils.getFlag('Z', options));

    setFindNumBinsWithLogLL(Utils.getFlag('L', options));

    if (getFindNumBins() && getFindNumBinsWithLogLL()) {
      throw new Exception("Cannot use find num bins option and find"+
			  " num bins with loglikeli option at the same time.");
    }
    if (getFindNumBinsWithLogLL()) {
      setNumBins(100);
    }

    String numBins = Utils.getOption('B', options);
    if (numBins.length() != 0) {
      setNumBins(Integer.parseInt(numBins));
    } 

    // set minimum percent of unique values
    String percentString = Utils.getOption('P', options);
    if (percentString.length() > 0)
      setIntValuesPercent(Double.parseDouble(percentString));

    // set offset in percent
    String offsetString = Utils.getOption('O', options);
    if (offsetString.length() > 0)
      setOffset(Double.parseDouble(offsetString));

    if (getOffset() < 0.0 || getOffset() > 100.0) {
      throw new Exception("Offset is percent, so must be a value in the range [0.0..100.0].");
    }
   
  }

  /**
   * Gets the current settings of the filter.
   *
   * @return an array of strings suitable for passing to setOptions
   */
  public String [] getOptions() {

    Vector result = new Vector();

    // options of the superclass
    String [] superOptions = super.getOptions();
    for (int i = 0; i < superOptions.length; i++) {
      result.add(superOptions[i]);
    }

    // find num of bins by cv
    if (getFindNumBins()) {
      result.add("-Y");
    }

    // find offset using cv
    if (getFindOffset()) {
      result.add("-Z");
    }

    // find number of bins using loglikelihood
    if (getFindNumBinsWithLogLL()) {
      result.add("-L");
    }

    // set number of bins
    if (getNumBins() != getNumBinsDefault()) {
      result.add("-B");
      result.add("" + getNumBins());
    }

    // offset in percent
    if (getOffset() != getOffsetDefault()) {
      result.add("-O");
      result.add("" + getOffset());
    }
 
    if (getIntValuesPercent() != getIntValuesPercentDefault()) {
      result.add("-P");
      result.add("" + getIntValuesPercent());
    }
    return (String[])result.toArray(new String[result.size()]);
  }

  /**
   * Returns the tip text for this property
   *
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String intValuesPercentTipText() {

    return "Assume that the values are integer values if the percentage"
      + " of unique vales are below a certain value and "
      +" and search a bin width.";
  }

  /**
   * Get the default value of the integer values flag.
   *
   * @return value of the integer values flag.
   */
  public double getIntValuesPercentDefault() {
    
    return 100.0;
  }
  
  /**
   * Get the value of the integer values flag.
   *
   * @return value of the integer values flag.
   */
  public double getIntValuesPercent() {
    
    return m_intValuesPercent;
  }
  
  /**
   * Set the value of percentage for integer values
   *
   * @param newPercent the percent value for .
   */
  public void setIntValuesPercent(double newPercent) {
    
    m_intValuesPercent = newPercent;
  }

  /**
   * Returns the tip text for this property
   *
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String findNumBinsTipText() {

    return "Optimize number of equal-width bins using leave-one-out. Doesn't " +
      "work for equal-frequency binning";
  }

  /**
   * Get the value of FindNumBins.
   *
   * @return Value of FindNumBins.
   */
  public boolean getFindNumBins() {
    
    return m_findNumBins;
  }
  
  /**
   * Set the value of FindNumBins.
   *
   * @param newFindNumBins Value to assign to FindNumBins.
   */
  public void setFindNumBins(boolean newFindNumBins) {
    
    m_findNumBins = newFindNumBins;
    setNumBins(100);
  }

  /**
   * Get the value of the flag findOffset.
   *
   * @return Value of FindOffset
   */
  public boolean getFindOffset() {
    
    return m_findOffset;
  }
  
  /**
   * Set the value of the flag findOffset.
   *
   * @param newFindOffset Value to assign to FindOffset.
   */
  public void setFindOffset(boolean newFindOffset) {
    
    m_findOffset = newFindOffset;
    setNumBins(100);
  }
  
  /**
   * Returns the tip text for this property
   *
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String findNumBinsWithLogLLTipText() {

    return "Optimize number of equal-width bins using leave-one-out. Doesn't " +
      "work for equal-frequency binning";
  }

  /**
   * Get the value of FindNumBinsWithLogLL.
   *
   * @return Value of FindNumBinsWithLogLL.
   */
  public boolean getFindNumBinsWithLogLL() {
    
    return m_findNumBinsWithLogLL;
  }
  
  /**
   * Set the value of FindNumBinsWithLogLL.
   *
   * @param newFindNumBinsWithLogLL Value to assign to FindNumBinsWithLogLL.
   */
  public void setFindNumBinsWithLogLL(boolean newFindNumBinsWithLogLL) {
    
    m_findNumBinsWithLogLL = newFindNumBinsWithLogLL;
  }
  
  /**
   * Returns the tip text for this property
   *
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String binsTipText() {

    return "Number of bins.";
  }

  /**
   * Gets the default number of bins numeric attributes will be divided into
   *
   * @return the default number of bins.
   */
  public int getNumBinsDefault() {

    return 10;
  }

  /**
   * Gets the number of bins numeric attributes will be divided into
   *
   * @return the number of bins.
   */
  public int getNumBins() {

    return m_numBins;
  }

  /**
   * Sets the number of bins to divide each selected numeric attribute into
   *
   * @param numBins the number of bins
   */
  public void setNumBins(int numBins) {

    m_numBins = numBins;
  }

  /**
   * Returns the Density of a value.
   *
   * @return the value of the density function at the given value
   */
  public double getDensity(double value) {
    //todo
    return 0.0;
  }
  /**
   * Returns the cut points.
   *
   * @return an array containing the cutpoints (or null if the
   * attribute requested has been discretized into only one interval.)
   */
  public double [] getCutPoints() {

    if (m_cutPoints == null) {
      return null;
    }
    return m_cutPoints;
  }

  /**
   * Initialize the estimator with a new dataset.
   *
   * @param data the dataset used to build this estimator 
   * @param attrIndex attribute the estimator is for
   * @exception if building of estimator goes wrong
   */
  public void addValues(int attrIndex, Instances inst) throws Exception {

   m_filePostfix = "EW";

    // find bin width
    double binWidth = -1.0;
    if (getIntValuesPercent() < 100.0) {
      binWidth = findIntBinWidth(inst, attrIndex, getIntValuesPercent());
      if (binWidth > 0.0) {
	m_minValue -= binWidth / 2.0;
        computeNumBins(binWidth, m_minValue, m_maxValue); 
      } 
    }

    // make cut points
    if (binWidth > 0.0) {
      makeCutPoints2(attrIndex, binWidth, m_minValue);
    } else {
      if (getFindNumBins() && getFindOffset()) {
	int numBins = cvOffsetAndNumBins(inst, attrIndex, m_minValue, m_maxValue);
	setNumBins(numBins);
      } else {
	if (getFindNumBins()) {
	  int numBins = cvNumBins(inst, attrIndex, m_minValue, m_maxValue);
	  setNumBins(numBins);
	  
	} else {
	  if (getFindNumBinsWithLogLL()) {
	    int numBins = logllNumBins(inst, attrIndex);
	    setNumBins(numBins * 2);
	  }
	}
      }

      // add to estimator to find cutpoints
      makeCutPoints(attrIndex, m_minValue, m_maxValue);
    }

    // transform cutpoints into bins
    boolean [] cutAndLeft = null;
    if (m_cutPoints != null) {
      cutAndLeft = new boolean[m_cutPoints.length];
      for (int i = 0; i < cutAndLeft.length; i++) {
	cutAndLeft[i] = true;
      }
    }
    m_bins = BinningUtils.cutPointsToBins(m_cutPoints, cutAndLeft, 
					    inst, attrIndex,
					    m_minValue, m_maxValue, m_alpha);

    dbo.dpln(D_RESULTBINS, BinningUtils.binsToString(m_bins));
    
    dbo.dpln(D_NUMANDOFFSET, "num bins = "+m_bins.size()+" offset = "+m_offset);
    
    if (dbo.dl(D_HISTOGRAM)) {
      try {
	writeHistogram("EW", inst.relationName(), attrIndex);
      } catch (Exception ex) {
	ex.printStackTrace();	
	System.out.println(ex.getMessage());
      }
    }
    }

//   /**
//    * Initialize the estimator with a new dataset. Minimum value,
//    * maximum value and bin width are predefined.
//    *
//    * @param data the dataset used to build this estimator 
//    * @param attrIndex attribute the estimator is for
//    */
//   public void addValues(Instances inst, int attrIndex, 
// 			double minValue, double maxValue,
// 			double binWidth) throws Exception {
    
//     // add to estimator to find cutpoints
//     makeCutPoints2(attrIndex, binWidth, minValue);

//     // transform into bins
//     boolean [] cutAndLeft = null;
//     if (m_cutPoints != null) {
//       cutAndLeft = new boolean[m_cutPoints.length];
//       for (int i = 0; i < cutAndLeft.length; i++) {
// 	cutAndLeft[i] = true;
//       }
//     }
//     m_bins = EstimatorUtils.cutPointsToBins(m_cutPoints, cutAndLeft, 
// 					    inst, attrIndex,
// 					    m_minValue, m_maxValue, m_alpha);

//     dbo.dpln(D_RESULTBINS, EstimatorUtils.binsToString(m_bins));

//     if (dbo.dl(D_HISTOGRAM)) {
//       String filename = inst.relationName();
//       if (filename.length() > 10)
// 	filename = filename.substring(0, 10);
//       if (m_fileName != null) filename = m_fileName;
//       try {
// 	DBO.pln("3min "+ m_minValue+" max "+m_maxValue);
// 	EstimatorUtils.writeHistogram(filename +"-"+attrIndex+"EW", m_bins, m_minValue, m_maxValue);

// 	EstimatorUtils.writeHistogram(filename +"EW3", m_bins, m_minValue, m_maxValue);
//       } catch (Exception ex) {
// 	ex.printStackTrace();

// 	System.out.println(ex.getMessage());
//       }
//     }
//   }


  /**
   * Set cutpoints for a single attribute.
   *
   *@param attrIndex index of the attribute
   *@param classIndex index of the class attribute
   *@param classValue value of the classattribute 
   */
  private void makeCutPoints(int attrIndex, 
			     double min, double max) { //throws Exception {
    //todo min == max
    double binWidth = (max - min) / getNumBins();
    
    makeCutPoints2(attrIndex, binWidth, min);
  }

  /**
   * Set cutpoints for a single attribute.
   *
   *@param attrIndex index of the attribute
   *@param classIndex index of the class attribute
   *@param classValue value of the classattribute 
   */
  private void makeCutPoints2(int attrIndex, 
			      double binWidth,
			      double min) { //throws Exception {

    // offset is in percent of the bin width
    if (m_offset == 0.0) { m_offset = 100.0; }
    if (m_offset < 100.0) { m_numBins++; }

    // compute the bin width of the first bin
    double firstBinWidth = binWidth * (m_offset / 100.0);
    double [] cutPoints = null;

    if ((m_numBins > 1) && (binWidth > 0)) {
      cutPoints = new double [m_numBins - 1];
      // first bin
      cutPoints[0] = min + firstBinWidth;

      for(int i = 1; i < m_numBins - 1; i++) {
	cutPoints[i] = min + firstBinWidth + binWidth * i;
      }
    }
    m_cutPoints = cutPoints;

    // -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    // debug output
    if (dbo.dl(D_RESULTBINS)) {
      if (cutPoints == null) {
	DBO.pln("\n# no cutpoints found - attribute "+attrIndex); 
      } else {
	DBO.pln("\n#* "+cutPoints.length+" cutpoint(s) - attribute "+attrIndex+" offset "+m_offset); 
	for (int i = 0; i < cutPoints.length; i++) {
	  DBO.pln("# "+cutPoints[i]+" "); 
	}
	DBO.pln("# leftEnd");
      }
    }
    // -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  }

  /**
   * Compute the number of bins for integer values.
   *
   * @param index the attribute index
   * @return number of bins that scored best
   * @exception in getLoglkLeaveOneOut
   */
  protected void computeNumBins(double binWidth, double minValue,
				double maxValue) {
    DBO.pln(""+ binWidth +" "+minValue+" "+maxValue);
    int numBins = 0;
    minValue += binWidth;
    while (minValue < maxValue) {
      numBins++;
      minValue += binWidth;
    }
    m_numBins = numBins;
  } 

  /**
   * Optimizes the number of bins using leave-one-out cross-validation.
   *
   * @param index the attribute index
   * @return number of bins that scored best
   * @exception in getLoglkLeaveOneOut
   */
  protected int cvNumBins(Instances inst, int attrIndex, 
			  double min, double max) throws Exception {

    //DBO.pln("cvnumbins");
    double trainEntropy = 0.0;
    double minEntropy = Double.MAX_VALUE;
    double minNumBins = 0.0;

    // try one bin only 
    m_cutPoints = null;
    boolean [] cutAndLeft = null;
    m_bins = BinningUtils.cutPointsToBins(m_cutPoints, cutAndLeft, 
					    inst, attrIndex,
					    min, max, m_alpha);

    // Compute cross-validated entropy
    double entropy = 0;
    double bestEntropy = BinningUtils.getLOOCVLoglkFromBins(m_bins, 
							      inst.numInstances());
    minEntropy = bestEntropy;
    int bestNumBins = 1;

//     if (outputTypeSet(D_ENTROPYCURVE)) {
//       trainEntropy = getLoglkLeaveOneOutTrain(m_bins, inst.numInstances());
//       DBO.pln("" + 1.0 + "  " + bestEntropy +" trainLlk "+trainEntropy);
//     }
    
    int saveNumBins = m_numBins;    
    for (int numBins = 2; numBins < saveNumBins; numBins++) {
      // DEBUG output
      m_numBins = numBins;
      
      // make cutpoints
      makeCutPoints(attrIndex, min, max);
      
      // make bins
      if (m_cutPoints == null) cutAndLeft = null;
      else {
	cutAndLeft = new boolean[m_cutPoints.length];
	for (int i = 0; i < cutAndLeft.length; i++) {
	  cutAndLeft[i] = true;
	}
      }
      m_bins = BinningUtils.cutPointsToBins(m_cutPoints, cutAndLeft, 
					      inst, attrIndex,
					      min, max, m_alpha);
      dbo.dpln(D_RESULTBINS, BinningUtils.binsToString(m_bins));
      
      
      // Compute cross-validated entropy
      entropy = BinningUtils.getLOOCVLoglkFromBins(m_bins, inst.numInstances());
      
      // -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
      if (dbo.dl(D_ENTROPYCURVE)) {
	// output for entropy curve
	trainEntropy = 
	  BinningUtils.getTrainLOOCVLoglkFromBins(m_bins,
						    inst.numInstances());
	dbo.dpln(D_ENTROPYCURVE, "" + numBins + "  " + entropy +" trainLlk "+trainEntropy);
	//  " err "+error+" aeB " + numAlmEmptyBins + " bins " 
	//    + numBins + " illegalcuts "+ numIllegalCuts +
	//    " totallyuniform "+ numTotallyUniform);
      }
      // -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
      
      // Best entropy so far?
      if (entropy > bestEntropy) {
	bestEntropy = entropy;
	bestNumBins = numBins;
      }
      
      // -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
      // output min and max
      if (entropy < minEntropy) {
	minEntropy = entropy;
	minNumBins = numBins;
      }
      dbo.dpln(D_ENTROPYCURVE, "#min "+bestNumBins+" max "+minNumBins);
      // -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
      
    }
    
    // -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    // output current best number of bins
    dbo.dpln(D_ENTROPYCURVE, "# bestnum "+bestNumBins);
    
    // -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    
    //   int m_numBins = saveNumBins;    
    return bestNumBins;
  }

  /**
   * Optimizes the number of bins using leave-one-out cross-validation.
   *
   * @param index the attribute index
   * @return number of bins that scored best
   * @exception in EstimatorUtils.getLoglkLeaveOneOut
   */
  protected int logllNumBins(Instances inst, int attrIndex) throws Exception {

    Estimator est = new TUBEstimator();
    ((TUBEstimator)est).setAlpha(m_alpha);
    ((TUBEstimator)est).setMaxNumBins(getNumBins());

    ((Estimator)est).addValues(inst, attrIndex);
    Vector bins = ((BinningEstimator)est).getBins();

    if (bins == null) return 0;
    return bins.size();
  }

  /**
   * Optimizes the number of bins using leave-one-out cross-validation.
   *
   * @param index the attribute index
   * @return number of bins that scored best
   * @exception in getLoglkLeaveOneOut
   */
  protected int cvOffsetAndNumBins(Instances inst, int attrIndex, 
				   double min, double max) throws Exception {

    //DBO.pln("cvOffsetAndNumBins");
    double trainEntropy = 0.0;
    double minEntropy = Double.MAX_VALUE;
    double minNumBins = 0.0;
    int bestoffset = 0;

    // try one bin only 
    m_cutPoints = null;
    boolean [] cutAndLeft = null;
    m_bins = BinningUtils.cutPointsToBins(m_cutPoints, cutAndLeft, 
					    inst, attrIndex,
					    min, max, m_alpha);

    // Compute cross-validated entropy
    double entropy = 0;
    double bestEntropy = BinningUtils.getLOOCVLoglkFromBins(m_bins, 
							      inst.numInstances());
    int bestNumBins = 1;

//     if (dbo.dpln((D_ENTROPYCURVE)) {
//       trainEntropy = getLoglkLeaveOneOutTrain(m_bins, inst.numInstances());
//       DBO.pln("" + 1.0 + "  " + bestEntropy +" trainLlk "+trainEntropy);
//     }

    
    int saveNumBins = m_numBins;
    for (int numBins = 2; numBins < saveNumBins; numBins++) {
      for (int offset = 0; offset < 10; offset++) {

	m_offset = 10.0 * (double)offset;    
	
	// DEBUG output
	m_numBins = numBins;
	
	// make cutpoints
	makeCutPoints(attrIndex, min, max);
	
	// make bins
	cutAndLeft = null;
	if (m_cutPoints != null) {
	  cutAndLeft = new boolean[m_cutPoints.length];
	  for (int i = 0; i < cutAndLeft.length; i++) {
	    cutAndLeft[i] = true;
	  }
	}
	m_bins = BinningUtils.cutPointsToBins(m_cutPoints, cutAndLeft, 
						inst, attrIndex,
						min, max, m_alpha);

	dbo.dpln(D_RESULTBINS, BinningUtils.binsToString(m_bins));

	// Compute cross-validated entropy
	entropy = BinningUtils.getLOOCVLoglkFromBins(m_bins, inst.numInstances());
	
	// -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
	if (dbo.dl(D_ENTROPYCURVE)) {
	  // output for entropy curve
	  trainEntropy = 
	    BinningUtils.getTrainLOOCVLoglkFromBins(m_bins,
						      inst.numInstances());
	  dbo.dpln(D_ENTROPYCURVE, "" + numBins + "  " + entropy +" trainLlk "+trainEntropy);
	  //  " err "+error+" aeB " + numAlmEmptyBins + " bins " 
	  //    + numBins + " illegalcuts "+ numIllegalCuts +
	  //    " totallyuniform "+ numTotallyUniform);
	}
	// -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
	
	// Best entropy so far?
	if (entropy > bestEntropy) {
	  bestEntropy = entropy;
	  bestoffset = offset;
	  bestNumBins = numBins;
	  //DBO.pln("#bestnumbins "+bestNumBins+" bestoffset "+bestoffset);
	}
	// -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
	// output min and max
	if (entropy < minEntropy) {
	  minEntropy = entropy;
	  minNumBins = numBins;
	}
	if (dbo.dl(D_ENTROPYCURVE)) {
	  DBO.pln("#min "+bestNumBins+" max "+minNumBins);
	}
	// -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
      }
      
      // output current best number of bins
      dbo.dpln(D_ENTROPYCURVE, "# bestnum "+bestNumBins+" bestoffset "+bestoffset);
    }

    //   int m_numBins = saveNumBins;
    m_offset = 10.0 * (double)bestoffset;
    setOffset(m_offset);    
    return bestNumBins;
  }

 /**
   * Get a probability estimate for a value
   *
   * @param value the value to estimate the probability of
   * @return the estimated probability of the supplied value
   */
  public double getProbability(double value) {
    if (m_bins == null) {
      return -1.0;
    }     
    double prob = BinningUtils.getProbability(value, m_bins,
						m_minValue, m_maxValue); 
    return prob;
  }

  /**
   * Find the binwidth of an 'integer' value
   *
   */
  public static double findIntBinWidth(Instances inst, int attrIndex,
				       double maxPercent) {
    Instances workData = null;
    double binWidth = -1.0;
    Attribute att = inst.attribute(attrIndex);
    AttributeStats as = inst.attributeStats(attrIndex);
    
    // only numeric attributes
    if (att.type() == Attribute.NUMERIC) {
      
      Vector distances = new Vector();
      // percent of values distinct
      double percent = 100.0 * as.uniqueCount / as.totalCount;
      if (percent <= maxPercent) {
	workData = new Instances(inst);
	workData.sort(attrIndex);

	// find smallest distance
	double small = Double.MAX_VALUE;
	for (int i = 0; i < workData.numInstances() - 1; i++) {
	  double dist =  workData.instance(i + 1).value(attrIndex) 
	    -  workData.instance(i).value(attrIndex);
	  if (dist > 0.0 && dist < small) {
	    small = dist;
	  } 
	}
	if (small > 0.0 && small < Double.MAX_VALUE)
	  binWidth = small;
      }
    }
    //	  DBO.p (" binWidth "+binWidth);
    return binWidth;
  }

  /**
   * Display a representation of this estimator.
   *@return a string giving a representation 
   */
  public String toString() {

    String text = BinningUtils.toString(m_bins, "Equal Width");
    return text;
  }

  /**
   * Main method for testing this class.
   *
   * @param argv should contain the options for the estimator
   */
  public static void main(String [] argv) {

    try {
      EqualWidthEstimator est = new EqualWidthEstimator();
      
      Estimator.buildEstimator((Estimator) est, argv, false);      
      System.out.println(est.toString());

    } catch (Exception ex) {
      ex.printStackTrace();
      System.out.println(ex.getMessage());
    }
  }
  
  /**
   * Returns the revision string.
   * 
   * @return		the revision
   */
  public String getRevision() {
    return RevisionUtils.extract("$Revision: 1.00 $");
  }

 }








