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
 *    TUBEstimator.java
 *    Copyright (C) 2004 Gabi Schmidberger
 *
 */

package weka.estimators;

import java.io.FileOutputStream;
import java.io.PrintWriter;
import java.io.Serializable;
import java.util.Enumeration;
import java.util.Random;
import java.util.Vector;

import weka.core.Capabilities;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.RevisionUtils;
import weka.core.Utils;
import weka.core.Capabilities.Capability;
import weka.core.Debug.DBO;
import weka.filters.unsupervised.attribute.Bin;
import weka.filters.unsupervised.attribute.CutInfo;
/** 
 <!-- globalinfo-start -->
 * Density estimator; the density is estimated by discretizing the attribute
 * using the TUBE algorithm.
 * TUBE builds a binary density estimation tree. It first selects an optimal
 * cut point in each range by maximizing the log-likelihood.
 * The number of splits is selected using crossvalidated log-likelihood.
 * The order of splits is given by taking as next split the one with highest
 * loglikelihood gain.
 <!-- globalinfo-leftEnd -->
 * 
 <!-- technical-bibtex-start -->
 * BibTeX:
 * <pre>
 * ;article{schmidberger05,
 *    author = {G.Schmidberger and E.Frank},
 *    title = {Unsupervised Discretization Using Tree Based Density Estimation},
 *    year = {2005}
 * }
 * 
 * </pre>
 * <p/>
 <!-- technical-bibtex-leftEnd -->
* See also:<p>
 * G.Schmidberger and E.Frank 2005 <i></i>.
 *
 * @author Gabi Schmidberger (gabi dot schmidberger at gmail dot com)
 * @version $Revision: 1.0 $
 */
public class TUBEstimator extends BinningEstimator
implements OptionHandler {
  
  /** for serialization */
  private static final long serialVersionUID = 8761672570540603851L;
  
  /** output cutpoint and entropy value there */
  public static int D_INFOVERBOSE     = 0; // 1 on the command line
  
  /** output before each split */
  public static int D_FOLLOWSPLIT     = 1; // 2 
  
  /** output 10 cutpoints beween instances */
  public static int D_TENVALUES       = 2; // 3
  
  /** list all cutpoints at leftEnd */
  public static int D_RESULTBINS      = 3; // 4
  
  /** difference between densities */
  public static int D_DIFFDENS        = 4; // 5 on the command line
  
  /** illegal cut as it happens */
  public static int D_ILLCUT          = 5; // 6 
  
  /** num cuts and its averaged entropy */
  public static int D_NUMCUTENTROPY   = 6; // 7
  
  /** print a histogram file */
  public static int D_HISTOGRAMS      = 7; // 8
  
  /** print priority queue info */
  public static int D_PRIORITY        = 8; // 9
  
  /** info from bins in getLoglkFromBins */
  public static int D_BININFO         = 9; // 10 on the command line
  
  /** output about all split possibilities */
  public static int D_ABOUTSPLIT      = 10; // 11 
  
  /** output 10 cutpoints beween instances */
  public static int D_SPLITACC        = 11; // 12
  
  /** list all cutpoints at leftEnd */
  public static int D_SUMILLCUTS      = 12; // 13
  
  /** list all cutpoints at leftEnd */
  public static int D_NUMCUTMISE      = 13; // 14
  
  /** trace through precedures */
  public static int D_TRACE           = 14; // 15
  
  /** trace through precedures */
  public static int D_ILLCUTCONTROL   = 15; // 16
  
  /** output cutpoint and entropy value there */
  public static int D_ATCUT           = 16; // 17 on the command line
  
  /** output cutpoint and entropy value there */
  public static int D_MINDIFF         = 17; // 18 on the command line
  
  /** look at at each real bin made */
  public static int D_LOOKATBINS      = 18; // 19
  
  /** output loglikeli and widths */
  public static int D_WID_LLK         = 19; // 20
  
  /** The class value index is > -1 if subset is taken with specific class value only*/
  protected double m_classValueIndex = -1.0;
  
  // Class that contains info about a split
  //
  
  private class SpInfos implements Serializable {
    
    int numMissing = 0; 
    int numIllegalCuts = 0;
    int numTotallyUniform = 0;
    double bigN = -1.0;
    double bigL = -1.0; 
    Vector bins = new Vector();
    Vector priorityQueue = new Vector();
    int splitDepth = 0;
    double totalLLK = 0.0;
    //double diff = 0.0;
    double minValue = 0.0;
    double maxValue = 0.0;
    Vector newBins = new Vector();
    
    /*
     * Resets splitting variables.
     */
    protected void resetSplitting() {
      numMissing = 0; 
      numIllegalCuts = 0;
      numTotallyUniform = 0;
      bigN = -1.0;
      bigL = -1.0; 
      bins = new Vector();
      priorityQueue = new Vector();
      splitDepth = 0;
      //totalLLK = 0.0;
      //diff = 0.0;
      minValue = 0.0;
      maxValue = 0.0;
      newBins = new Vector();
    }
    
  } // leftEnd class SpInfos
  
  /** Store the current cutpoints */
  protected double [] m_cutPoints = null;
  
  /** Store for each cutpoint if point at cut goes to left bin */
  protected boolean [] m_cutAndLeft = null;
  
  /** The maximum splitting depth */
  protected int m_maxDepth = Integer.MAX_VALUE;
  
  /** The maximum number of splits */
  protected int m_maxSplits = 99;
  
  /** The current maximum number of splits */
  protected int m_currentMaxSplits = -1;
  
  /** Split after class values before discretizing. */
  //protected boolean m_splitClass = false;
  
  /** Store the current cutpoints in the order they are done */
  protected Vector m_cutPointsTree = new Vector();
  protected boolean m_storeSplits = false;
  
  /** Dont allow illegal cuts. */
  protected boolean m_forbidIllegalCut = true;
  
  /** Default for forbiddenCut. */
  private static final int M_NOFORBIDDENCUT      = -1;
  
 /** using EW 10 to define illegal cuts */
  protected static final int C_TOOHIGH_EW10      = 1;
  
  /** using a min eps to define illegal cuts */
  protected static final int C_MIN_EPS           = 2;
  
  /** using a min eps to define illegal cuts */
  protected static final int C_WIDTH_EW10        = 3;
  
  /** using a min eps to define illegal cuts */
  protected static final int C_HALFWIDTH_EW10    = 4;
   
  /** Dont allow different types of cuts. */
  protected int m_forbiddenCut = M_NOFORBIDDENCUT;
  
  /** threshhold for some kinds of forbidden cuts */
  protected double m_TreshCutHeight = Double.MAX_VALUE;
  
  /** The seed used for discretization */
  protected int m_seed;
  
  /** different splitting criteria */
  private static final int STANDARD_SPLIT = 0;  
  private static final int CV_SPLIT = 1;
  private static final int FULL_SPLIT = 2;
 // private static final int BICRELATED_SPLIT = 3;
  private static final int WEIRD_SPLIT = 4;
  private static final int NOEMPTY_SPLIT = 5;
  
  /** holds the choice of the splitting methods */
  private int m_splitMethod = FULL_SPLIT;
  
  /** flag, if set instances at the splitpoint are put into the right bin */
  private boolean m_RightFlag = false;
  
  /** flag, if set instances at the splitpoint are put once into the right bin,
   once into the left */
  private boolean m_bothFlag = true;
  
  /**  flag true if epsilon is set to half minimal distance. */
  protected boolean m_minDistEpsilon = false;
  
  /** flag, if set min and max are set epsilon away from min and max using m_epsilon 
   as distance */
  private double m_epsilonBorder = 0.0;
  
  /** flag, if set split point is set beside the point using m_epsilon 
   as distance */
  private boolean m_epsilonCutting = true;
  
  private double m_epsilon = 1.0E-4;
  
  /** flag, if set split point is set in the weighted middle between 
   the points */
  private boolean m_middleCutting = false;
  
  /** flag, if cuts are set on the grid */
  private boolean m_gridCutting = false;
 
 /** flag, if set decision on splitting depends on global criterion */
  private boolean m_global = true;
  
  /** the n used if global is set or not */
  private double m_theN = 0;
  
///** flag, if set decision on split depends on crossvalidation*/
//private boolean m_crossValidating = false;
  
  /** flag, if set crossvalidation for number of splits is switched off*/
  private boolean m_noCVNumSplits = false;
  
  /** flag, if set crossvalidation is done with the least squares method */
  private boolean m_leastSquaresCV = false;
  
  /** flag, use the least squares criterion to find a cut point  */
  private boolean m_leastSquaresCut = false;
  
  /** number of grid cells */
  private int m_gridNum = -1;
  
 /** leftBegin value of the presplitgrid */
  private double m_gridBegin = Double.NaN;
  
  /** distance between the splits with presplit-splitting */
  private double m_gridWidth = Double.NaN;
  
  /** special case: range has only one value - value */
  private double m_oneValueOnlyValue = Double.NaN;
  
  /** special case: range has only one value - flag */
  private boolean m_oneValueOnlyFlag = false;
  
  
  /**
   * Constructor.
   */
  public TUBEstimator() {
    
  }
  
  /**
   * Get a probability estimate for a value using the density from the histogram
   *
   * @param value the value to estimate the probability of
   * @return the estimated probability of the supplied value
   */
  public double getProbability(double value) {
    
    // special case; estimator was trained with identical values only
    if (m_oneValueOnlyFlag) {
      if (m_oneValueOnlyValue == value) { 
        return 1.0;
      } else {
        return 0.0;
      }
    }
    
    Bin bin = null;
    double prob = 0.0;
    
    // no data therefore no bins, probability is 0.0 
    if (m_bins == null) return prob;
    
    //value = round(value);
    
    // outside the boundaries the probability is 0.0
    if (value < getMinValue()) { 
      // dbo.dpln("< RANGE " + value);
      return 0.0;
    } else {
      if (value > getMaxValue()) { 
        // dbo.dpln("> RANGE " + value);
        return 0.0;
      } else {
        
        // find the right bin   
        bin = BinningUtils.findBin(value, m_bins, 
            ((Bin)m_bins.elementAt(0)).getMinValue(), 
            ((Bin)m_bins.elementAt(m_bins.size() - 1)).getMaxValue());
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
   * Display a representation of this estimator.
   *
   *@return a string giving a representation of the estimator
   */
  public String toString() {
    
    String text = BinningUtils.toString(m_bins, "Loglikeli") + " LLK= "+m_resultLLK;
    if (dbo.dl(D_WID_LLK)) {
    	String numAndWidths = BinningUtils.toNumAndWidthString(m_bins);
    	dbo.pln(numAndWidths);
    	LoglikeliComputer comp = new LoglikeliComputer();
    	double llk = comp.computeLikelihood(numAndWidths);
    	int numInst = comp.getNumInst();
    	dbo.pln("Num instance "+numInst+" LLK = " + llk);
    }
    if (dbo.dl(D_NUMCUTENTROPY) || dbo.dl(D_ABOUTSPLIT) || dbo.dl(D_TENVALUES) || dbo.dl(D_NUMCUTMISE)) {
      return "# "+text;
    }
    return text;
  }
  
  /*
   * Class to represent a split.
   */
  private class Split implements Serializable {
    
    // Loglikelihoods
    double leftLLK = Double.MAX_VALUE;
    double rightLLK = Double.MAX_VALUE;
    
    // LLK of the sum of both bins
    double oldLLK = Double.MAX_VALUE;
    
    // LLK of the sum of both bins
    double newLLK = Double.MAX_VALUE;
    
    // LLK before/after split difference
    double trainLLKDiff = Double.MAX_VALUE;
    
    // if true cut and put instances at cutpoint to the right
    boolean rightFlag = false;
    
    // index of the instance where it is split
    int index = -1;
    
    // the middle leftBegin and ends, they differ according to right flag
    int lastLeft = -1;
    int firstRight = -1;
    
    // number of instances to the left of the split
    double leftNum = -1.0;
    
    // number of instances to the right of the split
    double rightNum = -1.0;
    
    // distance of the split towards the left leftEnd
    double leftDist = -1.0;
    
    // distance of the split towards the right leftEnd
    double rightDist = -1.0; 
    
    // value at which the split is performed
    double cutValue = -1.0;
    
    // the bin in which the split happened
    Bin bin = null;
    int splitDepth = -1;
  } // leftEnd of class Split
  
  
  /**
   * Set splitting method to standard.
   * @param standard boolean if true method is set to standard splitting
   */
  public void setStandardSplitting(boolean standard) {
    if (standard) {
      m_splitMethod = STANDARD_SPLIT;
    } 
  }
  
  /**
   * Set splitting method to 'weird' splitting.
   * @param weird boolean if true method is set to weird splitting
   */
  public void setWeirdSplitting(boolean weird) {
    if (weird) {
      m_splitMethod = WEIRD_SPLIT;
    } 
  }
  
  /**
   * Get true if splitting method is set to 'weird' splitting.
   *
   * @return boolean, if true method is currently set to weird splitting,
   */
  public boolean getWeirdSplitting() {
    return (m_splitMethod == WEIRD_SPLIT);
  }
  
  /**
   * Set splitting method to 'no empty bins allowed' splitting.
   * @param f boolean if true method is set to no-empty splitting
   */
  public void setNoEmptySplitting(boolean f) {
    if (f) {
      m_splitMethod = NOEMPTY_SPLIT;
    } 
  }
  
  /**
   * Returns true if splitting method is set to 'no empty bins allowed'.
   * @return boolean, if true method is currently set to no-empty splitting,
   */
  public boolean getNoEmptySplitting() {
    return (m_splitMethod == NOEMPTY_SPLIT);
  }
  
  /**
   * Set splitting method to full splitting.
   * @param full boolean if true method is set to full splitting
   */
  public void setFullSplitting(boolean full) {
    if (full) {
      m_splitMethod = FULL_SPLIT;
    }
  }
  
  /**
   * Returns true if splitting method is set to full splitting
   * @return true if splitting method is currently set to full splitting
   */
  public boolean getFullSplitting() {
    return (m_splitMethod == FULL_SPLIT);
  }
  
  /**
   * Set splitting method to full splitting.
   * @param full boolean if true method is set to full splitting
   */
  public void setGridCutting(boolean newFlag) {
    m_gridCutting = newFlag;
   }
  
  /**
   * Returns true if splitting method is set to grid cutting
   * @return true if splitting method is currently set to grid cutting
   */
  public boolean getGridCutting() {
    return (m_gridNum > 1);
  }
  
  /**
   * Set number of grid cells for grid cutting
   * @param full int number of gridcells
   */
  public void setGridNum(int num) {
    m_gridNum = num; 
    setGridCutting(getGridCutting());
  }
  
  /**
   * Returns number of grid cells for grid cutting 
   * @return number of grid cells for grid cutting
   */
  public int getGridNum() {
    return m_gridNum;
  }
  
  /**
   * Sets the parameters for grid cutting
   * @param minValue minimal value of this attribute
   * @param maxValue maximal value of this attribute
   * @exception if the grid cannot be build 
   */
  public void setGrid(double minValue, double maxValue) throws Exception {
    double num = (double) getGridNum();
    if (num < 1.0) throw new Exception("Grid cannot be initialized for grid cutting");
    m_gridWidth = (maxValue - minValue) / num;
    m_gridBegin = minValue + m_gridWidth;
  }
  
  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String rightTipText() {
    
    return "Instance at split point is put into the right bin.";
  }
  
  /**
   * Gets the flag if the instances at the split point are put into the right bin
   * @return the setting of the right flag
   */
  public boolean getRightFlag() {
    
    return m_RightFlag;
  }
  
  /**
   * Sets the flag, if set to true the instances at each split point are put into the right bin
   *
   * @param newFlag the new flag value
   */
  public void setRightFlag(boolean newFlag) {
    
    m_RightFlag = newFlag;
  }
  
  /**
   * Gets the flag if the CV method used to determine the number of splits is set
   * to least squares
   * @return the setting of the least squares CV flag
   */
  public boolean getLeastSquaresCV() {
    
    return m_leastSquaresCV;
  }
  
  /**
   * Sets the flag, if set to true the CV method used to determine the number of splits is
   * least squares
   *
   * @param newFlag the new flag value
   */
  public void setLeastSquaresCV(boolean newFlag) {
    
    m_leastSquaresCV = newFlag;
  }
  
  
  /**
   * Gets the flag if set to true the method to find the split in the range is
   * LeastSquaresCut
   *
   * @return the setting of the leave one out CV flag
   */
  public boolean getLeastSquaresCut() {
    
    return m_leastSquaresCut;
  }
  
  /**
   * Sets the flag, if set to true the method to find the split in the range is
   * LeastSquaresCut
   *
   * @param newFlag the new flag value
   */
  public void setLeastSquaresCut(boolean newFlag) {
    
    m_leastSquaresCut = newFlag;
  }
  
  /**
   * Gets the flag, if set to true epsilon splitting is set and
   * epsilon is set to half of the minimal distance > 0.0 found in the
   * data set
   * @return the setting of the minDistEpsilon cutting flag
   */
  public boolean getMinDistEpsilon() {
    
    return m_minDistEpsilon;
  }
  
  /**
   * Sets the flag, if set to true epsilon splitting is set and
   * epsilon is set to half of the minimal distance > 0.0 found in the
   * data set
   *
   * @param newFlag the new flag value
   */
  public void setMinDistEpsilon(boolean newFlag) {
    
    m_minDistEpsilon = newFlag;
    if (m_minDistEpsilon) {
      m_epsilonCutting = true;
      m_middleCutting = true;
    }
  }
  
  /**
   * Gets the flag if
   * @return the setting of the right flag
   */
  public double getEpsilonBorder() {
    return m_epsilonBorder;
  }
  
  /**
   * Sets the flag, if set to true 
   * @param newFlag the new flag value
   */
  public void setEpsilonBorder(double border) {
    m_epsilonBorder = border;
  }
  
  /**
   * Gets the flag if
   * @return the setting of the epsilon cutting
   */
  public double getEpsilon() {
    return m_epsilon;
  }
  
  /**
   * Sets the epsilon value for epsiloncutting
   * @param eps the new value for epsilon
   */
  public void setEpsilon(double eps) {
    m_epsilon = eps;
  }
  
  /**
   * Gets the flag if
   * @return the setting of the epsilon cutting
   */
  private boolean getEpsilonCutting() {
    return !Double.isNaN(m_epsilon);
  }
  
  
  /**
   * Gets true if the cutting is set to weighted middle
   * @return the setting of the middle cutting flag
   */
  public boolean getMiddleCutting() {
    return m_middleCutting;
  }
  
  /**
   * Sets the flag, if set to true cutting is set to weighted middle
   *
   * @param newFlag the new flag value
   */
  public void setMiddleCutting(boolean newFlag) {
    m_middleCutting = newFlag;
    if (m_middleCutting) m_epsilonCutting = false;
  }
  
  /**
   * Gets the value for the random seed
   *
   * @return the setting of the seed
   */
  public int getSeed() {
    return m_seed;
  }
  
  /**
   * Sets the value for the random seed
   * @param seed the new seed value
   */
  public void setSeed(int seed) {
    m_seed = seed;
  }
  
///**
//* Sets the value for the alpha uniform noise.
//* @param newValue the new alpha value
//*/
//public void setAlpha(double newValue) {
//m_alpha = newValue;
//}
  
///**
//* Gets the value for the alpha uniform noise.
//* @return the alpha value
//*/
//public double getAlpha() {
//return m_alpha;
//}
 
  /**
   * Gets true if decision on split is not done by cross validating
   * @return the setting of the cross validation flag
   */
  public boolean getNoCVNumSplits() {
    return m_noCVNumSplits;
  }
  
  /**
   * Sets the flag, if set to true cross validating is switched off
   * @param newFlag the new flag value
   */
  public void setNoCVNumSplits(boolean newFlag) {
    m_noCVNumSplits = newFlag;
    if (newFlag)
      setFullSplitting(true);
  }
  
  /**
   * Sets the maximum number of bins the range can be split into
   *
   * @param max the maximum number of splits
   */
  public void setMaxNumBins(int max) {
    m_maxSplits = max - 1;
  }
  
  /**
   * Returns the maximum number of bins 
   *
   * @return max the maximum number of binss
   */
  public int getMaxNumBins() {
    return m_maxSplits + 1;
  }
  
///**
//* Sets the number of instances for illegal cut computation
//*
//* @param numInst  number of instances for illegal cut computation
//*/
//public void setNumInstForIllCut(double numInst) {
//m_numInstForIllCut = numInst;
//}
  
///**
//* Returns the number of instances for illegal cut computation
//*
//* @return max the maximum number of binss
//*/
//public double getNumInstForIllCut() {
//return m_numInstForIllCut;
//}
  
///**
//* Switches the outputs on that are requested from the option V
//* if list is empty switches on the verbose mode only
//* @param list list of integers, all are used for an output type
//*/
//public void setVerboseLevels(String list) { 
//dbo.setOutputTypes(list);
//}
  
///**
//* Gets the current output type selection
//*
//* @return a string containing a comma separated list of ranges
//*/
//public String getVerboseLevels() {
//return dbo.getOutputTypes();
//}
  
///**
//* Sets the filename for output
//*
//* @param n the new file name
//*/
//public void setFileName(String n) {
//m_fileName = n;
//}
  
///**
//* Returns the filename for all info output
//*
//* @return filename that would be used for info output
//*/
//public String getFileName() {
//return m_fileName;
//}
  
  /** 
   * Sets which kind of illegal cuts should be disallowed
   * 
   * @param num new mode value
   */
  public void setForbiddenCut(int num) {
    m_forbiddenCut = num;
  }
  
  /** 
   * Sets whether illegal cuts are dissallowed
   * @return true if illegal cuts are set to be disallowed
   */
  public int getForbiddenCut() {
    return m_forbiddenCut;
  }
  
  /** 
   * Sets whether illegal cuts (<10% of length and <= 2 instances) should be
   * dissallowed
   * @param flag new flag value
   */
  public void setForbidIllegalCut(boolean flag) {
    m_forbidIllegalCut = flag;
  }
  
  /** 
   * Sets whether illegal cuts are dissallowed
   * @return true if illegal cuts are set to be disallowed
   */
  public boolean getForbidIllegalCut() {
    return m_forbidIllegalCut;
  }
  
  /**
   * Gets an enumeration describing the available options.
   *
   * @return an enumeration of all the available options.
   */
  public Enumeration listOptions() {

    Vector newVector = new Vector(16);

    newVector.addElement(new Option(
	"\tSet seed for cross-validation (Default 0).",
	"S", 0, "-S <seed>"));
    newVector.addElement(new Option(
	"\tSwitch off cross-validation to determine the number of splits.\n"
	+ "\t(Default is CV is switched on). \n"
	+ "\tSwitches on Standard split method. Standard split method means\n"
	+ "\tloglikelihood must increase more than -(log(N)+log(2))",
	"N", 0, "-N"));
    newVector.addElement(new Option(
	"\tSwitch on Full split method:\n"
	+ "\tThe split is accepted as long as the loglikelihood is increased.\n"
	+ "\tThis is the default setting. Does not affect if CV or not.",
	"F", 0, "-F"));
    newVector.addElement(new Option(
	"\tSwitch on W split method:\n"
	+ "\tThe split is accepted as long as the loglikelihood is increased by more than\n"
	+ "\t-((N * log(sum/N))/10). Sum is number of instances of current range.\n" 
	+ "\tN is total number of instances. Does not affect if CV or not.",
	"W", 0, "-W"));
    newVector.addElement(new Option(
	"\tDisallow empty splits:\n"
	+ "\tDisallow splits that result in one of the subranges empty.",
	"Y", 0, "-Y"));
    newVector.addElement(new Option(
	"\tSplit in the middle:\n"
	 + "\tSplit always in the middle of two instances.",
	"M", 0, "-M"));
    newVector.addElement(new Option(
	"\tChoose grid cutting method and set number of grid partitions.",
	"G", 0, "-G <num>"));
    newVector.addElement(new Option(
	"\tDisallow illegal cuts. Illegal cut is defined that 1 or 2 is true.\n" 
	+"\t1. Number of instances below threshhold with threshold is\n"
	+"\tsqrt(10% of total number of instances\n"
	+"\t2. Width is smaller 0.1% of total width.",
	"L", 0, "-L"));   
    newVector.addElement(new Option(
	"\tSwitch on other methods to forbid cuts. (only 1 implemented)\n"
	+"\t-U 1 .. density gets higher than 2x the max. density of 10 EqualWidth",
	"U", 0, "-U <num>"));
    newVector.addElement(new Option(
	"\tUse Least Squares criterion for CV.",
	"Q", 0, "-Q"));
    newVector.addElement(new Option(
	"\tUse Least Squares criterion for cut.",
	"O", 0, "-O"));
    newVector.addElement(new Option(
	"\tSet the cutting distance (distance cut to next instance)\n"
	+"\tas the 1/2 * minimal distance between two instances.",
	"C", 0, "-C"));
    newVector.addElement(new Option(
	"\tExtend the border (default 0.0).",
	"E", 0, "-E <value>"));
    newVector.addElement(new Option(
	"\tSet the distance to the instance where the cut is set (default 1.0E-4).",
	"Z", 0, "-Z <value>"));
    newVector.addElement(new Option(
	"\tCut at each instance only once with leaving the instance in the right bin.",
	"R", 0, "-R"));
    newVector.addElement(new Option(
	"\tSet the maximum number of bins. Note: If CV is switched on,\n"
	+"\tthe result might have less bins. (default 100)",
	"B", 0, "-B <num>"));
    Enumeration enu = super.listOptions();
    while (enu.hasMoreElements()) {
      newVector.addElement(enu.nextElement());
    }
    return newVector.elements();
  }

  /**
   * Parses the options for this object. Valid options are: <p>
   * @param options the list of options as an array of strings
   * @exception Exception if an option is not supported
   */
  public void setOptions(String[] options) throws Exception {  
    
    m_splitMethod = FULL_SPLIT;
    
    super.setOptions(options); 
    
    // set seed for random
    String seedString = Utils.getOption('S', options);
    if (seedString.length() > 0)
      setSeed(Integer.parseInt(seedString));
    
    setNoCVNumSplits(Utils.getFlag('N', options));
    //if (getNoCVNumSplits()) { m_splitMethod = STANDARD_SPLIT; }
    setFullSplitting(Utils.getFlag('F', options));
    setWeirdSplitting(Utils.getFlag('W', options));
    setNoEmptySplitting(Utils.getFlag('Y', options));
    setMiddleCutting(Utils.getFlag('M', options));
    
    String gridNum = Utils.getOption('G', options);
    if (gridNum.length() > 0) {
      setGridNum(Integer.parseInt(gridNum));
    }
    
    setForbidIllegalCut(Utils.getFlag('L', options));
    
    String fCuts = Utils.getOption('U', options);  
    if (fCuts.length() != 0) {
      setForbiddenCut(Integer.parseInt(fCuts));
    } 
    
    setLeastSquaresCV(Utils.getFlag('Q', options));
    setLeastSquaresCut(Utils.getFlag('O', options)); // was V in earlier version
    
    // set Epsilon cutting
    setMinDistEpsilon(Utils.getFlag('C', options));
    String borderString = Utils.getOption('E', options);
    if (borderString.length() > 0) {
      setEpsilonBorder(Double.parseDouble(borderString));
    }
    String epsilonString = Utils.getOption('Z', options);
    if (epsilonString.length() > 0) {
      setEpsilon(Double.parseDouble(epsilonString));
    }
    
    
    // R option, put instance at border into right bin 
    setRightFlag(Utils.getFlag('I', options));
    
    String numBins = Utils.getOption('B', options);
    if (numBins.length() != 0) {
      setMaxNumBins(Integer.parseInt(numBins));
    }     
  }
  
  /**
   * Gets the current settings of the filter.
   *
   * @return an array of strings suitable for passing to setOptions
   */
  public String [] getOptions() {
    
    Vector        result;
    String[]      options;
    int           i;

    result  = new Vector();
    options = super.getOptions();
    for (i = 0; i < options.length; i++)
      result.add(options[i]);

    result.add("-S"); 
    result.add(" " + getSeed());
      
    if (getNoCVNumSplits()) {
      result.add("-N");
    }
    
    if (getFullSplitting()) {
      result.add("-F");        
    }
    
    if (getWeirdSplitting()) {
      result.add("-W");
    }

    if (getNoEmptySplitting()) {
      result.add("-Y");
    }

    if (getMiddleCutting()) {
      result.add("-M");
    }

    if (getGridCutting()) {
      result.add("-G");
      result.add(" "+getGridNum());
    }

    if (getForbidIllegalCut()) {
      result.add("-L");
    }

    if (getForbiddenCut() != M_NOFORBIDDENCUT) {
      result.add("-U");
      result.add(" " + getForbiddenCut());
    }

    if (getLeastSquaresCV()) {
      result.add("-Q");
    }
    
    if (getLeastSquaresCut()) {
      result.add("-O");
    }
    
    if (getMinDistEpsilon()) {
      result.add("-C");
    }
    
    if (getEpsilonBorder() > 0.0) {
      result.add("-E");
      result.add(" " + getEpsilonBorder());
    }
    
     
    if (getEpsilonCutting()) {
      result.add("-Z");
      result.add(" " + getEpsilon());
    }
    
    // R option, put instance at border into right bin 
    if (getRightFlag()) {
      result.add("-I");
    }
    
    if (getMaxNumBins() > 0) {
      result.add("-B");
      result.add(" " + getMaxNumBins());
    }     

    
    return (String[]) result.toArray(new String[result.size()]);
   }
  
  /**
   * Returns a string describing this filter
   *
   * @return a description of the filter suitable for
   * displaying in the explorer/experimenter gui
   */
  public String globalInfo() {
    
    return ".";
  }
  
///**
//* Get the minimal value of this estimator
//* @return the minimal value of an attribute
//*/
//public double getMinValue() {
  
//if (m_bins == null) return Double.MIN_VALUE;
//return ((Bin)m_bins.elementAt(0)).getMinValue();
//}
  
///**
//* Get the maximal value of this estimator
//* @return the maximal value of an attribute
//*/
//public double getMaxValue() {
  
//if (m_bins == null) return Double.MAX_VALUE;
//return ((Bin)m_bins.elementAt(m_bins.size() - 1)).getMaxValue();
//}
    
  /**
   * Mix vector of bins int the global vecor of bins
   * in m_bins, the second is given by parameter. 
   * If two cutpoints are at the same value and the flag is the same, 
   * the two become one point.
   * @param bins2 the additional vector of bins
   */
  private CutInfo mixBins(Vector bins2) {
    
    CutInfo info2 = BinningUtils.binsToCutInfo(bins2);
    mixCutPoints(info2.m_cutPoints, info2.m_cutAndLeft);
    info2.m_cutPoints = m_cutPoints;
    info2.m_cutAndLeft = m_cutAndLeft;
    //    m_bins = EstimatorUtils.cutInfoToBins(info2);
    return info2;
  }
  
  /**
   * Mix a cutpoint list and their flags into the global lists stored
   * in m_cutPoints, the second is given by parameter. 
   * If two cutpoints are at the same value and the flag is the same, 
   * the two become one point.
   *@param cut2 the additional list of cutpoints
   *@param left2 the additional list of flags
   */
  private void mixCutPoints(double [] cut2, boolean [] left2) {
    
    if (cut2 == null) return;
    if (m_cutPoints == null) {
      m_cutPoints = cut2;
      m_cutAndLeft = left2;
      return;
    }
    int len1 = m_cutPoints.length;
    int len2 = cut2.length;
    double [] cut1 = m_cutPoints;
    boolean [] left1 = m_cutAndLeft;
    double [] newCut = new double[(len1 + len2) * 2];
    boolean [] newLeft = new boolean[(len1 + len2) * 2];
    int i = 0;
    int begin1 = 0; int first1 = 0;
    int begin2 = 0; int first2 = 0;
    double upTo = 0.0;
    
    while (begin1 < len1 || begin2 < len2) {
      
      // set the barrier value 
      if (first2 < cut2.length) { upTo = cut2[first2]; }
      else { upTo = Double.POSITIVE_INFINITY;}
      
      // select all cutpoints in cut1 that are smaller than the next in cut2
      while ((begin1 < len1) && (cut1[begin1] < upTo)) {
        newCut[i] = cut1[begin1]; 
        newLeft[i] = left1[begin1]; i++;
        begin1++;
      }
      
      // values are the same
      if ((begin1 < len1) && (cut1[begin1] < upTo)) {
        if (left1[begin1] == left2[begin2]) {
          newCut[i] = cut1[begin1];
          newLeft[i] = left1[begin1]; i++;
        } else {
          newCut[i] = cut1[begin1];
          newLeft[i] = false; i++;
          newCut[i] = cut1[begin1];
          newLeft[i] = true; i++;
          newCut[i] = cut1[begin1];
          newLeft[i] = false; i++;
        }
        begin1++; begin2++;
      }
      first1 = begin1;
      
      // mark down the next barrier
      if (first1 < cut1.length) { upTo = cut1[first1]; }
      else { upTo = Double.POSITIVE_INFINITY;}
      
      // select all cutpoints in cut2 that are smaller than the next in cut1
      while ((begin2 < len2) && (cut2[begin2] < upTo)) {
        newCut[i] = cut2[begin2];
        newLeft[i] = left2[begin2]; i++;
        begin2++;
      }
      
      // values are the same
      if ((begin2 < len2) && (cut1[begin1] < upTo)) {
        if (left1[begin1] == left2[begin2]) {
          newCut[i] = cut1[begin1];
          newLeft[i] = left1[begin1]; i++;
        } else {
          newCut[i] = cut1[begin1];
          newLeft[i] = false; i++;
          newCut[i] = cut1[begin1];
          newLeft[i] = true; i++;
          newCut[i] = cut1[begin1];
          newLeft[i] = false; i++;
        }
        begin1++; begin2++;
      }
      first2 = begin2;
      
    }
    
    // make the arays fitting
    double [] helpCut = new double[i];
    boolean [] helpLeft = new boolean[i];
    System.arraycopy(newCut, 0, helpCut, 0, helpCut.length); 
    System.arraycopy(newLeft, 0, helpLeft, 0, helpLeft.length);
    m_cutPoints = helpCut;
    m_cutAndLeft = helpLeft;
  }
  
  
  /**
   * Initialize the estimator with a set of values.
   *
   * @param attrIndex attribute the estimator is for
   * @param data the dataset used to build this estimator 
   */
  public void addValues(int attrIndex, Instances data) throws Exception {
    
    Instances workData = new Instances(data);
    
    /* special case- only one value */
    if (m_minValue == m_maxValue)
    {
      m_oneValueOnlyValue = m_minValue;
      m_oneValueOnlyFlag = true;
      // make a bin for all 
      addValues_specialCase(m_oneValueOnlyValue, workData);
      return;
    } // end special case
 
    //  find the epsilon, using half of the minimal distance between the values
    if (m_minDistEpsilon) {
      workData.sort(attrIndex);
      //dataSorted = true;
      double min = BinningUtils.findMinDistance(workData, attrIndex);
      double epsilon = min / 2.0;
      // -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
      dbo.dpln(D_MINDIFF, "# minimal distance is "+min);
      setEpsilon(epsilon);
      setEpsilonBorder(epsilon);
    }
    dbo.dpln(D_MINDIFF, "# epsilon is "+m_epsilon);
    
    
    // widen borders
    m_minValue = getMinValue() - m_epsilonBorder;
    m_maxValue = getMaxValue() + m_epsilonBorder;
    
    // make bin for special case
    if (m_gridCutting) {
      setGrid(m_minValue, m_maxValue);
    }
    
    
    m_filePostfix = "LL";
    dbo.dpln(D_TRACE, "addValues -TUBEstimator");
    
    // prepare to forbid some types of cuts
    defineForbiddenCut(data, attrIndex);
    
    m_storeSplits = false;
    int numFolds = 10;
    double numIllegalCuts = 0.0;    
    double avgCVIllegalCuts = 0.0;    
    
    try {
      Random random = new Random(m_seed);
      double [] cutPoints = null;
      boolean [] cutAndLeft = null;
      
      // cross validation over number of splits
      if (!m_noCVNumSplits &&  (workData.numInstances() > numFolds)) {
        boolean stopCV = false;
        Instances train = null;
        Instances test = null;
        int cvNumInstances = workData.numInstances() * (numFolds - 1) / numFolds;
        
        // zero split entropy 
        double llk = 0.0;
        double llk_lsq = 0.0;
        double sqr = 0.0;
        //double uniformLlk = 0.0;
        double trainLlk = 0.0;
        double minLlk = Double.MAX_VALUE;
        double maxLlk = - Double.MAX_VALUE;
        double numTotallyUniform;   
        double numAlmEmptyBins = 0.0; 
        double error = 0.0; 
        double oldNumBins = -1.0; 
        double numBins = 0.0; 
        int bestNumber = -1;
        int minNumber = -1;
        
        m_currentMaxSplits = -1;
        Instances [] trainSet = new Instances[10];
        Instances [] testSet = new Instances[10];
        SpInfos [] spInfos = new SpInfos[10];
        SpInfos sp_sqr = new SpInfos();
        
        workData.randomize(random);
        //dataSorted = false;	
        
        // get all train and testsets
        for (int i = 0; i < 10; i++) {
          trainSet[i] = workData.trainCV(numFolds, i, random);
          
          // Sort input data
          trainSet[i].sort(attrIndex);
          
          testSet[i] = workData.testCV(numFolds, i);
          
          // prepare splitting information
          spInfos[i] = new SpInfos();    
        }
        // initialize split infos and training sets
        // already sorted
        for (int i = 0; i < numFolds; i++) {
          initializeForSplits(spInfos[i], trainSet[i], attrIndex, 
              m_minValue, m_maxValue);
        }
        
        // special prep for least squared cross-validation
        //if (getLeastSquaresCV()) {
        workData.sort(attrIndex);
        initializeForSplits(sp_sqr, workData, attrIndex, m_minValue, m_maxValue);
        //}
        
        double [] aveDensity = null;
        
        // ****************************************************************************
        // cross validation over number of splits
        do {
          oldNumBins = numBins;
          m_currentMaxSplits++;
          llk = 0.0; trainLlk = 0.0; numBins = 0.0; numIllegalCuts = 0.0;
          numTotallyUniform = 0;
          numAlmEmptyBins = 0;
          error = 0.0;
          for (int i = 0; i < numFolds; i++) {
//          if (i == 1) {
//          dbo.dpln("#split fold "+m_currentMaxSplits+"  \n"+
//          EstimatorUtils.binsToString(spInfos[i].bins));
//          }
            
            // Build the discretization using the train fold
            trainLlk += performSplits(spInfos[i], trainSet[i], attrIndex);
            
            numBins += spInfos[i].bins.size();
            aveDensity = new double[spInfos[i].bins.size()];
            numIllegalCuts += (double)spInfos[i].numIllegalCuts;
            numTotallyUniform += (double)spInfos[i].numTotallyUniform;
            
            double n = getNumAlmEmptyBins(spInfos[i].bins, trainSet[i].numInstances());
            numAlmEmptyBins += n;
            
            // test discretization
            double l = 0.0;
            double l_lsq = 0.0;
            if (m_minValue == m_maxValue) {
            } else {
              // -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
              l_lsq = getLkFromBins(spInfos[i].bins, testSet[i], attrIndex);
              // -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
              if (getLeastSquaresCV()) {
                l = getLkFromBins(spInfos[i].bins, testSet[i], attrIndex);
              } else {
                l = getLoglkFromBins(spInfos[i].bins, testSet[i], attrIndex);
              }
//            if (dbo.dl(D_NUMCUTENTROPY)) { 
//            dbo.dpln(EstimatorUtils.binsToString(spInfos[i].bins));
//            dbo.dpln("#loglikeli "+i+" = "+l);
//            }
            }
            setAveDensity(aveDensity, spInfos[i].bins);
            
            llk_lsq += l_lsq;
            llk += l;
            error += getErrorsFromBins(spInfos[i].bins);
          }
          
          // set averages
          llk = llk / numFolds;
          llk_lsq = llk_lsq / numFolds;
          
          double firstPart = llk;
          double squared = 0.0;
          if (getLeastSquaresCV()) {
            // perform splits on full data
            m_storeSplits = true;
            performSplits(sp_sqr, workData, attrIndex);
            //dbo.dpln("sp.bins "+sp.bins);
            
            squared = getSquaredLkFromBins(sp_sqr.bins, workData.numInstances());
            llk = (- squared + 2.0 * llk);	    
          }
          
          // -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
          // compute further statistics
          trainLlk = trainLlk / numFolds;
          numBins = numBins / numFolds;
          numIllegalCuts = numIllegalCuts / numFolds;
          numTotallyUniform = numTotallyUniform  / numFolds;
          numAlmEmptyBins = numAlmEmptyBins / numFolds;
          error = error / numFolds;
          if (dbo.dl(D_NUMCUTENTROPY)) { 
            dbo.dpln("" + m_currentMaxSplits + "  " + llk + " squa "+squared+" firstP "+firstPart+
                " trainLlk "+trainLlk+" err "+error+" aeB " + numAlmEmptyBins + " bins " 
                + numBins + " illegalcuts "+ numIllegalCuts +
                " totallyuniform "+ numTotallyUniform);
//          dbo.dpln("" + m_currentMaxSplits + "  " + llk +
//          " trainLlk "+trainLlk+" err "+error+" aeB " + numAlmEmptyBins + " bins " 
//          + numBins + " illegalcuts "+ numIllegalCuts +
//          " totallyuniform "+ numTotallyUniform);
            
          }
          // -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
          if (dbo.dl(D_NUMCUTMISE)) { 
            m_storeSplits = true;
            performSplits(sp_sqr, workData, attrIndex);
            sqr = getSquaredLkFromBins(sp_sqr.bins, workData.numInstances());
            double lsq = (- sqr + 2.0 * llk_lsq);	    
            dbo.dpln("" + m_currentMaxSplits + "  " + llk + " lsq " +lsq+
                " lsqllkpart "+llk_lsq+" lsqsquared "+sqr);
          }
          // -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
          
          // found new maximum
          if (llk > maxLlk) {
            maxLlk = llk;
            m_resultLLK = llk;
            bestNumber = m_currentMaxSplits;
            avgCVIllegalCuts = numIllegalCuts;
          }
          
          // -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
          // output min and max
          if (llk < minLlk) {
            minLlk = llk;
            minNumber = m_currentMaxSplits;
          }
          if (dbo.dl(D_NUMCUTENTROPY)) {
            dbo.dpln("#max "+bestNumber+" min "+minNumber);
          }
          // -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
          
          stopCV = false;
          if  ((m_maxSplits > -1) && (m_currentMaxSplits >= m_maxSplits)) { 
            stopCV = true; 
          }
          //if (m_currentMaxSplits > 100)  { stopCV = true; }
          if (m_currentMaxSplits > cvNumInstances)  { stopCV = true; }
        } while (!stopCV);
        m_currentMaxSplits = bestNumber;
      }
      // leftEnd cross-validation
      //dbo.dpln("leftEnd cross-validation");
      
      
      // set maximum number of splits, might have been computed by cross-validation
      if (m_currentMaxSplits == -1) {
        if (m_maxSplits != -1) {
          m_currentMaxSplits =  m_maxSplits;
        } else {
          m_currentMaxSplits =  workData.numInstances();
        }
      }
      
      // perform splits on full data
      m_cutPointsTree = new Vector();
      
      workData.sort(attrIndex);
      SpInfos sp = new SpInfos();
      initializeForSplits(sp, workData, attrIndex, m_minValue, m_maxValue);
      m_storeSplits = true;
      performSplits(sp, workData, attrIndex);
      
      // memorize average number of illegal cuts of CV plus actual
      if (!m_noCVNumSplits) {
        m_avgCVIllegalCuts = (double)avgCVIllegalCuts;
        m_numIllegalCuts = (double)sp.numIllegalCuts;
        m_diffNumIllegalCuts = (double)avgCVIllegalCuts - (double)sp.numIllegalCuts;
      } else {
        m_avgCVIllegalCuts = -1.0;
        m_numIllegalCuts = (double)sp.numIllegalCuts;
        m_diffNumIllegalCuts = -1.0;
      }
      
      // -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
      if (dbo.dl(D_SUMILLCUTS)) {
        // output number of illegal cuts
        dbo.dpln("#Average of Illegal cuts in CV: "+ avgCVIllegalCuts);
        dbo.dpln("#Illegal cuts:                : "+ sp.numIllegalCuts);
        m_diffNumIllegalCuts = avgCVIllegalCuts - (double)sp.numIllegalCuts;
        dbo.dpln("#Difference                   : "+ m_diffNumIllegalCuts);
      }
      // -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
      
      
      // fill global bins
      m_bins = sp.bins;
      
      // -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
      if (dbo.dl(D_RESULTBINS)) {
        // output cutpoints
        CutInfo info = getCutInfo();
        if (info.m_cutPoints == null) {
          dbo.dpln("\n# no cutpoints found - attribute "+attrIndex); 
        } else {
          dbo.dpln("\n#* "+info.m_cutPoints.length+" cutpoint(s) - attribute "
              +attrIndex); 
//        for (int i = 0; i < info.m_cutPoints.length; i++) {
//        dbo.dp("# "+info.m_cutPoints[i]+" "); 
//        dbo.dpln(""+info.m_cutAndLeft[i]);
//        }
          for (int i = 0; i < m_cutPointsTree.size(); i++) {
            Split split = (Split)  m_cutPointsTree.elementAt(i);
            dbo.dpln("# " + split.cutValue);
          }
          dbo.dpln("# leftEnd");
          dbo.dpln(BinningUtils.binsToString(m_bins));
        }
      }
      // -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
      
      
      // store cut points away
      m_cutPoints = cutPoints;
      m_cutAndLeft = cutAndLeft;
    } catch (Exception ex) {
      ex.printStackTrace();
      System.out.println(ex.getMessage());
    }
    
  }
  
  /**
   * special case - only one value
   * make a small bin for all in one
   */
  protected void addValues_specialCase(double value, Instances data) {

    //does the same as initializeForSplits(.....
    double bigN = data.numInstances() - 1.0;
    double numInstForIllCut = Double.NaN;
    double eps = m_epsilonBorder;
    if (m_epsilonBorder == 0.0)
      eps = m_epsilon;
    if (eps == 0.0)
      eps = 1E-4;
    double minValue = value - eps;
    double maxValue = value + eps;
    double bigL = maxValue - minValue;
    
    double totalLLK = getCutCriteria(bigN, maxValue - minValue, bigN) 
    			/ bigN;
    Bin bin = new Bin(0, bigN, bigL, numInstForIllCut, bigN, 0, 
        (int) (bigN - 1), minValue, true, 
        maxValue, true, totalLLK, m_alpha);
    bin.setSplitPath("P");
    
    if (dbo.dl(D_LOOKATBINS)) {
      dbo.dpln("#Firstbin\n"+bin.toString());
     }

    // add one and only to bins list
    m_bins = new Vector();
    m_bins.add(0, bin);
  }
  
  /**
   * Checks if it is a forbidden cut.
   *
   *@param numInst number of instances
   *@param width width of the bin
   *@param totalLen length of total range
   *@param totalNum number of all not missing instances
   */
  public boolean isForbiddenCut(double numInst, double width, 
      double totalLen,  double totalNum) {
    
    dbo.dpln(D_ILLCUTCONTROL,"check-type " + getForbiddenCut());
    switch (getForbiddenCut()) {
    case C_TOOHIGH_EW10:
      dbo.dpln(D_ILLCUTCONTROL,"CHECKING if--too high after 2x 10EW");
      
      // density = the height of the bin
      double density = getDensity(numInst, width, totalNum);
      
      // compare height with threshhold
      if (density > m_TreshCutHeight) {
        dbo.dpln(D_ILLCUTCONTROL, "too high after 2x 10EW");
        return true;
      }
      break;
    case C_MIN_EPS:
      dbo.dpln(D_ILLCUTCONTROL,"CHECKING if--width smaller than epsilon; eps = "+m_epsilon);
      
      if (width < m_epsilon) {
        dbo.dpln(D_ILLCUTCONTROL, "width smaller than epsilon; eps = "+m_epsilon);
        return true;
      }
      break;
    case C_WIDTH_EW10:
      dbo.dpln(D_ILLCUTCONTROL,"CHECKING if--width smaller than EW10 width; EW10width = "+m_epsilon);
      
      if (width < m_epsilon) {
        dbo.dpln(D_ILLCUTCONTROL, "width smaller than epsilon; eps = "+m_epsilon);
        return true;
      }
      break;
    case C_HALFWIDTH_EW10:
      dbo.dpln(D_ILLCUTCONTROL,"CHECKING if--width smaller than half EW10 width; EW10width = "+m_epsilon);
      
      if (width < m_epsilon) {
        dbo.dpln(D_ILLCUTCONTROL, "width smaller than epsilon; eps = "+m_epsilon);
        return true;
      }
      break;
    default:
      dbo.dpln(D_ILLCUTCONTROL,"Illegal cut condition not defined"); 
      break;
    }// switch
    
    return false;
  }
  
   /**
    * 
    * @param inst the data used
    * @param attrIndex index of the attribute
    * @throws Exception if definition does not work 
    */  
  public void defineForbiddenCut(Instances inst, int attrIndex) throws Exception {
    
    EqualWidthEstimator est = new EqualWidthEstimator();
    est.setFindNumBins(true);
    est.setNumBins(100);
    
    int v = getForbiddenCut();
    int v2 = C_WIDTH_EW10;
    int v3 = C_TOOHIGH_EW10;
    int v4 = C_HALFWIDTH_EW10;
    double value = 0;
    switch (getForbiddenCut()) {
    case C_TOOHIGH_EW10:
      //dbo.dpln("C_TOOHIGH_EW10");
      Estimator.buildEstimator((Estimator) est, inst, attrIndex, 
	  -1, -1, false);  
      dbo.dpln(D_ILLCUTCONTROL, est.toString());   
      value = ((BinningEstimator)est).getMaxHeight();
      m_TreshCutHeight = value * 2.0;
      dbo.dpln(D_ILLCUTCONTROL, "Height-threshhold "+m_TreshCutHeight);
      break;
    case C_WIDTH_EW10:
      Estimator.buildEstimator((Estimator) est, inst, attrIndex, 
	  -1, -1, false);  
      dbo.dpln(D_ILLCUTCONTROL, est.toString());  
      BinningEstimator b_est = (BinningEstimator)est;
      // set new epsilon
      value = b_est.getEWWidth();
      m_epsilon = value;
      Vector bins = b_est.getBins();
      setMaxNumBins(bins.size());
      // changes into min eps control
      setForbiddenCut(C_MIN_EPS);
      //dbo.pln("Width-threshhold "+m_epsilon+" numbins "+getMaxNumBins());
      dbo.dpln(D_ILLCUTCONTROL, "Width-threshhold "+m_epsilon);
      break;
    case C_HALFWIDTH_EW10:
      Estimator.buildEstimator((Estimator) est, inst, attrIndex, 
	  -1, -1, false);  
      dbo.dpln(D_ILLCUTCONTROL, est.toString());   
      b_est = (BinningEstimator)est;
      // set new epsilon
      value = ((BinningEstimator)est).getEWWidth();
      m_epsilon = value / 2.0;
      bins = b_est.getBins();
      setMaxNumBins((bins.size()));
      // changes into min eps control
      setForbiddenCut(C_MIN_EPS);
      //dbo.pln("Half-Width-threshhold "+m_epsilon+" numbins "+getMaxNumBins());
      dbo.dpln(D_ILLCUTCONTROL, "Half-Width-threshhold "+m_epsilon);
      break;
    default:
      break;
    }// switch
    
  }
  
  /**
   * Computes the average density for each bin with the new set of bins.
   * @param aveDensity the average densities sofar, actually the return value
   * @param bins a new set of bins
   */
  private void setAveDensity(double [] aveDensity, Vector bins) {
    
    for (int i = 0; i < bins.size(); i++) {
      Bin bin = (Bin) bins.elementAt(i);
      aveDensity[i] = (aveDensity[i] + bin.getNumInst())/2;
    }
  }
  
  /**
   * Method used for debug output.
   */
  private void printAveDensity(String f, double [] aveDensity, Vector bins) throws Exception {
    PrintWriter output = null;
    Bin bin = null;
    StringBuffer text = new StringBuffer("");
    
    if (f.length() != 0) {
      // add attribute indexnumber to filename and extension .arff
      String name = f + "H.txt";
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
        text.append("" + bin.getMinValue() + " " + aveDensity[i]+" \n");
        text.append("" + bin.getMaxValue() + " " + aveDensity[i]+" \n");
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
   * Merges neighboring bins that are both empty.
   * Used for debug purposes.
   * @return the new set of bins
   */
  private Vector checkBinsForNulls() {
    int lastNum = 1;
    for (int i = 0; i < m_bins.size(); i++) {
      int num = (int) ((Bin)m_bins.elementAt(i)).getNumInst();
      if (num == 0 && lastNum == 0) {
        m_bins.remove(i);
        dbo.dpln("remove-bins");
      } else {
        lastNum = num;
      }
    }
    return m_bins;
  }
  
  
  /*
   * Perfom the first split.
   * @param data the data set, used for data model information
   * @param attrIndex the index of the attribute that is discretized
   * @param minValue minimal value of this attribute
   * @param maxValue maximal value of this attribute
   * @return the total entropy of the result split
   */
  protected void initializeForSplits(SpInfos sp, Instances data, int attrIndex, 
      double minValue, double maxValue) {
    
    // reset all values in splitinfo
    sp.resetSplitting();   
    
    sp.minValue = minValue;
    sp.maxValue = maxValue;
    //double maxValue = data.instance(last).value(attrIndex);
    
    // exclude all missing values
    int last = data.numInstances() - 1;
    while ((last >= 0) && (data.instance(last).isMissing(attrIndex))) {
      sp.numMissing++;
      last--;
    }
    //dbo.dpln("last " + last);
    
    // totalnumber of instances might have some missing
    sp.bigN = (double)(last + 1);
    sp.bigL = sp.maxValue - sp.minValue; 
//  if (dbo.dl(D_RESULTBINS)) {
//  dbo.dpln("#Threshhold "+EstimatorUtils.getIllegalCutThreshhold(sp.bigN));
//  }
    
    // unsplit entropy
    sp.totalLLK = getCutCriteria(sp.bigN, sp.maxValue - sp.minValue, sp.bigN) 
    / sp.bigN;
    
    // start with one bin for the whole range
    Bin bin = new Bin(0, sp.bigN, sp.bigL, m_numInstForIllCut, sp.bigN, 0, 
        (int) (sp.bigN - 1), sp.minValue, true, 
        sp.maxValue, true, sp.totalLLK, m_alpha);
    
    bin.setSplitPath("P");
    
    if (dbo.dl(D_LOOKATBINS)) {
      dbo.dpln("#Firstbin\n"+bin.toString());
     }

    sp.bins.add(0, bin);
    sp.newBins.add(0, bin);
    
  }
  
  
  /*
   * Perfom the splits.
   * @param spInfo holds all the relevant 'global' values
   * @param data the data set, used for data model information
   * @param attrIndex the index of the attribute that is discretized
   * @param bigN the total number of instances
   * @return the total entropy of the result split
   */
  protected double performSplits(SpInfos spInfo, Instances data, int attrIndex) {
    
    //dbo.dpln("performsplits "+spInfo.bins.size()+" max splits "+m_currentMaxSplits+" totalLLK "+ spInfo.totalLLK);
    // -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
    // output: illegal cut
    if (dbo.dl(D_ILLCUT)) {
      dbo.dpln("# num inst for illegal cuts "+m_numInstForIllCut);
    }
    
    boolean maxSplit = false;
    double diff = 0.0;
    
//  Vector newBins = new Vector(); 
//  for (int i = 0; i < spInfo.bins.size(); i++) {
//  Bin bin = (Bin) spInfo.bins.elementAt(i);
//  newBins.add(bin);
//  }
    
    // number of splits already larger or equal than allowed splits
    if (spInfo.bins.size() - 1 >= m_currentMaxSplits) {
      maxSplit = true;
    } else {
      //
      // split until no more new splits found
      do {
        // look through all the new bins to find new possible splits
        for (int i = 0; i < spInfo.newBins.size(); i++) {
          Bin bin = (Bin)spInfo.newBins.elementAt(i);
          //dbo.dpln("new bin size " + spInfo.newBins.size());
          
          // find new split
          Split split = findSplit(spInfo, bin, data, attrIndex);	  
          //dbo.dpln("#after findsplit "+split);
          
          // split was found
          if (split != null) {
            // dbo.dpln(" split.entropy "+split.entropy);
            
            // add to priority queue, smallest difference, with all differences negative,
            // is highest priority  
            addToPriorityQueue(spInfo.priorityQueue, split);
            
          }
          //else   dbo.dpln(" split NOT found");
        }
        
        // all new bins have been examined
        spInfo.newBins = new Vector();
        
        //dbo.dpln("spInfo.priorityQueue.size() "+spInfo.priorityQueue.size());
        //dbo.dpln("maxSplit "+maxSplit);
        
        // get next split 
        if ((spInfo.priorityQueue.size() > 0) && !maxSplit) {
          
          // -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
          // output priority queue 
          if (dbo.dl(D_PRIORITY)) {
            dbo.dpln("# ");
            for (int j = 0; j < spInfo.priorityQueue.size(); j++) {
              Split s = (Split)spInfo.priorityQueue.elementAt(j);
              dbo.dp("#:" + spInfo.priorityQueue.size() + "::---");
              double total = spInfo.totalLLK + (s.trainLLKDiff / spInfo.bigN);
              dbo.dpln("# TOTAL-LL K="+total+" trainDiff "+s.trainLLKDiff+" @ cutvalue "+s.cutValue);
            }
            dbo.dpln("# ");
          }
          // -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
          
          // take first from priority queue and perform the split
          Split split = (Split)spInfo.priorityQueue.elementAt(0);
          spInfo.priorityQueue.remove(0);
          if (dbo.dl(D_PRIORITY)) {
            dbo.dpln("#::---"+split.cutValue+" taken.");
          }	  
          Bin bin = split.bin;
          
          
          
          // -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
          // output: illegal cut
          if (dbo.dl(D_ILLCUT) && bin.getIllegalCut()) {
            dbo.dpln("# Illegal cut chosen!!! is "+spInfo.bins.size()+"'s cut.");
          }
          // -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
          
          
          // compute new current LLK with adding LLK difference from split
          //spInfo.totalLLK += split.trainLLKDiff;
          
          // make two new bins, really perform next split
          //  public Bin(int splitDepth,
          // 	     double totalNum, double num, int leftBegin, int leftEnd,
          // 	     double min, boolean minIn, double max, boolean maxIn,
          // 	     double entropy) {
          String path = bin.getSplitPath();
          Bin leftBin = new Bin(split.splitDepth,
              spInfo.bigN, m_numInstForIllCut, spInfo.bigL, split.leftNum, 
              bin.getBegin(), split.lastLeft, bin.getMinValue(), bin.getMinIncl(), 
              split.cutValue, !split.rightFlag, split.leftLLK, m_alpha);
          leftBin.setSplitPath(path + "L");
          //dbo.dpln("Binleft\n"+leftBin.toString());
          Bin rightBin = new Bin(split.splitDepth,
              spInfo.bigN, m_numInstForIllCut, spInfo.bigL, split.rightNum, 
              split.firstRight, bin.getEnd(), split.cutValue, 
              split.rightFlag, bin.getMaxValue(), bin.getMaxIncl(), split.rightLLK,
              m_alpha);
          rightBin.setSplitPath(path + "R");
          //dbo.dpln("Binright\n"+rightBin.toString());
          
          
           // -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
          if (m_storeSplits) {
            // store cutpoints in order made
            m_cutPointsTree.add(split);
          }
          // -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
          
          // test if illegal cut was chosen
          leftBin.setIllegalCut();
          if (leftBin.getIllegalCut()) {
            spInfo.numIllegalCuts++;
          }
          rightBin.setIllegalCut();
          if (rightBin.getIllegalCut()) {
            spInfo.numIllegalCuts++;
          }
          
          // after split, two new bins, that will have to be examined
          spInfo.newBins.add(leftBin);
          spInfo.newBins.add(rightBin);
          // -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
          if (dbo.dl(D_LOOKATBINS)) {
            dbo.dpln("#Binleft\n"+leftBin.toString());
            dbo.dpln("#Binright\n"+rightBin.toString());         
          }
          // -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
                          
          // new total llk
          spInfo.totalLLK = spInfo.totalLLK 
          + (split.trainLLKDiff / spInfo.bigN);
          
          // delete the old bin in bins and put in the two new ones
          int binIndex = 0;
          for (int i = 0; i < spInfo.bins.size(); i++) {
            if (split.bin == (Bin)spInfo.bins.elementAt(i)) {
              binIndex = i;
            }
          }
          spInfo.bins.remove(binIndex);
          spInfo.bins.addAll(binIndex, spInfo.newBins);
          // -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
          // output total entropy 
          if (dbo.dl(D_PRIORITY)) {
            dbo.dpln("#Loglk = "+getTrainLoglkFromBins(spInfo.bins, attrIndex, spInfo.bigN));
          }
          
          // leftEnd if max splits have been done
          if ((spInfo.bins.size() - 1) >= m_currentMaxSplits) {
            maxSplit = true;
          }
        }
        
        
      } while (((spInfo.priorityQueue.size() > 0) || (spInfo.newBins.size() > 0))
          && (!maxSplit));
    }
    
    // perform splits on bins
    double loglk =  spInfo.totalLLK;
    double numInstances = data.numInstances() - spInfo.numMissing;
    return loglk / numInstances;
  }
  
  /**
   * Add to priority queue at the position of priority.
   * @param priorityQueue the priority queue
   * @param newSplit the new split element
   */
  protected void addToPriorityQueue(Vector priorityQueue, Split newSplit) {
    
    for (int i = 0; i < priorityQueue.size(); i++) {
      Split split = (Split)priorityQueue.elementAt(i);
      if (newSplit.trainLLKDiff > split.trainLLKDiff) {
        priorityQueue.add(i, newSplit);
        
//      dbo.dp("/"+i+"/");
//      for (int j = 0; j < priorityQueue.size(); j++) {
//      Split s = (Split)priorityQueue.elementAt(j);
//      dbo.dpln("cut@  "+s.cutValue+" entropy "+s.totalEntropy+" totEntValue "+s.entropy);
//      dbo.dp("///");
        
//      }
//      dbo.dpln("");
        return;
      }
    }
    //    dbo.dpln("#:"+priorityQueue.size()+":split added :last "+newSplit.cutValue);
    priorityQueue.add(newSplit);
  }
  
  /**
   * finds one split
   * @param sp global data used for the split
   * @param bin the bin to find the split in
   * @param data the data set 
   * @param attrIndex the attributes index
   * @return the details of the split
   */
  protected Split findSplit (SpInfos sp, Bin bin, Instances data, int attrIndex) {
    int currentDepth = bin.getSplitDepth();
    
    // if max depth given check if split should be done
    if (currentDepth > m_maxDepth) {
      return null;
    }
    // set which N to use
    if (m_global) {
      m_theN = sp.bigN;
    } else {
      m_theN =  bin.getWeight();
    }
    
    int begin = bin.getBegin();
    int end = bin.getEnd(); 
    double numInst = bin.getWeight();
    double minValue = bin.getMinValue();
    double maxValue = bin.getMaxValue();
    
//  if (m_crossValidating) {
//  dbo.dpln("crossValidating min "+minValue+" max "+maxValue);
//  double oldLoglikeli = getCriteria(numInst, maxValue - minValue, m_theN);;
//  double splitLoglikeli = 0.0;
//  // test if splitting by cross validation
//  for (int test = leftBegin; test <= leftEnd; test++) {
//  Instances testdata = new Instances(data, 0);
//  for (int i = leftBegin; i <= leftEnd; i++) {
//  if (i != test) testdata.add(data.instance(i));
//  }
//  //dbo.dpln("testinst "+data.instance(test));
//  //  public Bin(int splitDepth,
//  // 	     double totalNum, double num, int leftBegin, int leftEnd,
//  // 	     double min, boolean minIn, double max, boolean maxIn,
//  // 	     double entropy) {
    
//  Bin cvBin = new Bin(0, 0.0, m_bigL, numInst - 1, 0, (int) (numInst - 1), minValue, 
//  true, maxValue, true, oldLoglikeli, m_alpha);
//  Split cvSplit = findMinCriteria(bin, testdata, attrIndex, minValue, maxValue, bigN);
//  splitLoglikeli += testEntropy(data.instance(test), attrIndex, cvSplit, numInst - 1);
//  }
//  if (splitLoglikeli >= oldLoglikeli) {
//  // crossvalidated loglikelihood is not smaller than unsplit, therefore no split
//  return null;
//  }
//  } 
    Split split = null;
    if (!m_gridCutting)
      split = findMinCriteria(bin, data, attrIndex, minValue, maxValue, sp.bigN);
    else
      split = findMinCriteria(m_gridBegin, bin, data, attrIndex, minValue, maxValue, sp.bigN);
         
    if (split.index == -1) {
      // -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
      // reason no split found = no minimium found
      dbo.dpln(D_SPLITACC, "Split not accepted : no minimum found ["+minValue+":"+maxValue+"]");
      
      return null;
    }
    
    // new min found
    
    // -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
    // output cutvalue and entropy at cutvalue for each new split found (split not executed!)
    if (dbo.dl(D_FOLLOWSPLIT)) {
      dbo.dpln("#new max found " + split.cutValue + " minLeftNum " + split.leftNum + 
          " minRightNum " + split.rightNum + " minIndex " + split.index +
          " m_theN "+m_theN);
      double newLLK = getCutCriteria(split.leftNum, split.leftDist, 
    		  split.rightNum, split.rightDist, m_theN);

      dbo.dpln("newLLk "+newLLK);
    }
    // -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
    
    // check splitting criteria
    double newTotalLLK = sp.totalLLK + (split.trainLLKDiff / sp.bigN);
    //  - (split.oldLLK / sp.bigN) + (split.newLLK / sp.bigN);
    double diff = (split.newLLK / sp.bigN) - (split.oldLLK / sp.bigN);
    double diff2 = split.trainLLKDiff/sp.bigN;
    //dbo.dpln("diff "+ diff +" or "+ diff2);
    //dbo.dpln("split.oldLLK "+ split.oldLLK + " split.newLLK "+ split.newLLK );
    if (splitAccepted(sp, sp.totalLLK, newTotalLLK, split.leftNum, split.rightNum, 
        sp.bigN, split.leftDist, split.rightDist)) {
      // new splitpoint accepted
      if (dbo.dl(D_SPLITACC)) {
        dbo.dpln("#Split accepted "+split.cutValue);
      }
      split.bin = bin;
      split.splitDepth = currentDepth + 1;
      return split;
      
    } else {
      // -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
      // reason no split found = split accepted failed
      if (dbo.dl(D_SPLITACC)) {
        dbo.dpln("#Split not accepted "+split.cutValue);
      }
      // -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
      return null;
    }
  }
  
  	  /*
   * Compute the entropy for the cutpoint 
   * @param data the data set, used for data model information
   * @param index the index of the attribute that is discretized
   * @param cutPoints the values order in increasing way
   * @return how much the number of bins has increased in this branch
   */
  protected Split findMinCriteria(Bin bin, Instances data, int index, 
      double minValue, double maxValue, double bigN) {
    
    if (dbo.dl(D_ABOUTSPLIT)) {
      if (getLeastSquaresCut()) {
        DBO.p("#MISE-criteria: ");
      } else {
        DBO.p("#LLK -criteria: ");
      }
      DBO.pln("find min in range "+minValue+"--"+maxValue);
    }
    
    int begin = bin.getBegin();
    int end = bin.getEnd();
    
    //int numInBin = bin.getWeight();
    int totalN = (int)bin.getTotalNum();
    double totalL = bin.getTotalLen();
    double numInst = bin.getWeight();
    //dbo.dpln("findMinCriteria "+leftBegin+" -" +leftEnd+" // " +minValue+" - " +maxValue+" numinst "+numInst); 
    
    double leftNum = 0.0;
    double rightNum = 0.0;
    double leftDist = 0.0;
    double rightDist; 
    double oldLLK = getCriteriaFromBin(bin);
    //dbo.dpln("oldLLK = "+oldLLK);
    double newLLK = - Double.MAX_VALUE;
    double leftLLK = 0.0;
    double rightLLK = 0.0;
    double cutValue = 0.0;
    double lastCutValue = Double.NaN;
    
    boolean minRightFlag = false;
    
    //
    double width = bin.getWidth();
    double weight = bin.getWeight();
    double l = getLoglikelihood (weight, width, bigN);
    //dbo.dpln("oldLLK = "+oldLLK+" or " +l);
    
    Split split = new Split();
    split.newLLK = - Double.MAX_VALUE;
    split.oldLLK = oldLLK;
    split.index = -1;
    if (begin > end) return split;
    
    boolean toggle = false;
    int testRun = 0; // helps for toggling to count the tests done, every second toggle
    boolean rightFlag = true; 
    
    // three possible modes, always put instance at cutpoint to the left (default) or always to the right
    // or compare both ways and take the better one
    // if middle cutting is set then rightFlag and toggle must stay false
    rightFlag = getRightFlag();
    
    if (m_bothFlag) { toggle = true; rightFlag = true;}
    
    split.leftDist = 0.0;
    
    // offset is the index in the instance list
    int offset = 0;;
    
    if (getEpsilonCutting()) {
      rightDist = maxValue - minValue;
      offset = begin;
    } else {
      // find first point away from left border
      offset = begin - 1;
      do {
        offset++;
        if (offset <= end) {
          cutValue = getCutValue(data, index, offset, m_middleCutting); 
          //cutValue = data.instance(offset).value(index);
          leftDist = cutValue - minValue;
        }
      } while ((leftDist <= 0.0) && (offset <= end));
      
      rightDist = maxValue - cutValue; 
    }
    
    // set this for epsilon cutting
    double twotimes_epsilon = 0.0;
    boolean wasBorder = false;
    if (getEpsilonCutting()) { 
      twotimes_epsilon = 2.0 * m_epsilon;
      lastCutValue = minValue;
      if (minValue != getCutValue(data, index, begin, false)) wasBorder = true;
    }
    int numEpsilonCuts = 0;
    
    // ************************************************************************
    // check over all instances starting from the first instance that is away 
    // from the left range border,
    // cutting at the value of the instance 
    // and instance at cut point is put into the left partition
    int ii = offset;
    // dbo.dpln("######minValue "+minValue+" maxValue "+maxValue+" leftBegin "+leftBegin+" leftEnd "+leftEnd+"  ii "+ii+" rightDist "+rightDist);
    while ((ii <= end) && (rightDist > 0.0)) {
      // count the runs to help toggling
      testRun++;
      
      // if epsiloncuts
      if (getEpsilonCutting()) {
        cutValue = getCutValue(data, index, ii, false);
        double saveCutValue = cutValue;
        
        // dbo.dpln("#eps cutting ii:"+ii+" cutvalue "+getCutValue(data, index, ii, false)+
        //          " numepsiloncuts "+numEpsilonCuts+" lastCutValue "+lastCutValue);
        // new region	
        if (numEpsilonCuts == 0) {
          //dbo.dpln("numEpsilonCuts"+numEpsilonCuts);
          if (lastCutValue == cutValue) { 
            if (ii == end) numEpsilonCuts = 2;
            else ii++; /*dbo.dpln("break");*/ 
            continue; } 
          
          // special treatment for first from border
          if (wasBorder) { 
            // dbo.dpln("wasborder");
            wasBorder = false; 
            // make one or no cuts but count two
            numEpsilonCuts = 2;
            if ((cutValue - lastCutValue) > twotimes_epsilon) {
              cutValue = cutValue - m_epsilon;
            } else {
              // make no cut at border middle
              
              // special case leftEnd
              if (ii != end) { ii++; numEpsilonCuts = 0; }
              lastCutValue = saveCutValue;
              continue;
            }
            
          } else {
            // all other intervals
            if ((cutValue - lastCutValue) > twotimes_epsilon) {
              cutValue = lastCutValue + m_epsilon;
              numEpsilonCuts = 1;
            } else {
              // cut in the middle
              ////dbo.dpln("#cut in the middle");
              cutValue = (cutValue + lastCutValue) / 2.0;
              // only one cut, counts like two
              numEpsilonCuts = 2;
            }
          }
        } else {
          // second cut in region between two points
          if (numEpsilonCuts == 1) {
            //dbo.dpln("numEpsilonCuts"+numEpsilonCuts);
            cutValue = getCutValue(data, index, ii, false) - m_epsilon;
            numEpsilonCuts = 2;
          } else {
            
            // cut after leftEnd instance
            if (numEpsilonCuts == 2) {
              //dbo.dpln("numEpsilonCuts"+numEpsilonCuts);
              cutValue = maxValue;
              lastCutValue = getCutValue(data, index, ii, false);
              ii++; // leftEnd loop with ii > leftEnd
              if (lastCutValue == cutValue)  { continue; } 
              if ((cutValue - lastCutValue) > twotimes_epsilon) {
                cutValue = lastCutValue + m_epsilon;
              } else {
                // don't cut in the middle 
                continue;
              }
            }
            
          }
        }
        leftNum = ii - begin;
        rightNum = numInst - leftNum;
        if (numEpsilonCuts == 2) lastCutValue = saveCutValue;
        //dbo.dpln("#END cutting ii:"+ii+" cutvalue "+cutValue+
        //	 " numepsiloncuts "+numEpsilonCuts+" lastCutValue "+lastCutValue);
        // leftEnd epsilon cutting treatment
        
      } else {
        ////dbo.dpln("#NOT eps cutting");
        
        // before the cut and left, test if next instance has the same value
        if (!rightFlag && (ii + 1 < end)) {// therefore middle flag can also be set
          double valueNext = data.instance(ii + 1).value(index);
          if (valueNext == data.instance(ii).value(index)) {
            ii++;
            if (toggle) {
              // don't do this cut-and-left and the next cut-and-right
              testRun++; 
            }
            continue;
          }
        }
        // set the important values
        cutValue = getCutValue(data, index, ii, m_middleCutting);
        
        //cutValue = data.instance(ii).value(index);
        leftDist = cutValue - minValue;
        rightDist = maxValue - cutValue;
        if (rightDist <= 0.0) break; 
        
        // set number of instance on both sides, reset left/right distances if epsilon
        // cutting is on
        if (!rightFlag) {
          leftNum = ii - begin + 1;
          rightNum = numInst - leftNum;
        } else {
          leftNum = ii - begin ;
          rightNum = numInst - leftNum;
        }
        
      }
      
      // correct distances for all cases
      leftDist = cutValue - minValue;
      rightDist = maxValue - cutValue; 
      if (rightDist <= 0.0) break; 
      
      // -- finished setting the cut and all values 
      
      // get entropy value for this cut
      if (getLeastSquaresCut()) {
        newLLK = getLeastSquaresCutCriteria(leftNum, leftDist, rightNum, rightDist, 
            m_theN);
      } else {
        newLLK = getCutCriteria(leftNum, leftDist, rightNum, rightDist, m_theN);
      }
      
      
      
      // -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
      // output all LLKs
      if (dbo.dl(D_ABOUTSPLIT)) {
        double squaredLK = 0.0;
        if (getLeastSquaresCut()) {
          rightLLK = rightNum * getLikelihood(rightNum, rightDist, m_theN) /  m_theN;
          leftLLK = leftNum * getLikelihood(leftNum, leftDist, m_theN) / m_theN;
          squaredLK = getSquaredCutCriteria(rightNum, rightDist, leftNum, leftDist, m_theN);
          
        } else {
          rightLLK = getCutCriteria(rightNum, rightDist, m_theN);
          leftLLK = getCutCriteria(leftNum, leftDist, m_theN);
        }
        
        dbo.dpln(""+cutValue + " "+ newLLK +" "+leftLLK+" "+rightLLK+" "+squaredLK+
            " left "+leftNum+"/"+leftDist+" right "+rightNum+"/"+rightDist+
            " cut# "+cutValue);
      }
      // -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
      
      
      // -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
      // output entropy of 10 (or divide) points inbetween the instances
      if (dbo.dl(D_TENVALUES)) {
        int divide = 10;
        double diffInBetween = getDiffInBetween(data, index, ii, divide); 
        dbo.dpln("#diffInBetween "+diffInBetween);
        for (int oo = 1; oo < divide; oo++) {
          double ooDiff = diffInBetween * oo; 
          
          double ooLeftDist = leftDist + ooDiff;
          double ooRightDist = rightDist - ooDiff; 
          double ooRightLLK = getCutCriteria(rightNum, ooRightDist, m_theN);
          double ooLeftLLK = getCutCriteria(leftNum, ooLeftDist, m_theN);
          double ooTotalLLK = getCutCriteria(leftNum, ooLeftDist, rightNum, ooRightDist, m_theN);
          double ooCutValue = cutValue + ooDiff;
          dbo.dpln(""+ooCutValue + " "+ ooTotalLLK +" "+ooLeftLLK+" "+ooRightLLK+
              " ooleftDist "+ooLeftDist+" ooright "+rightNum+"/"+ooRightDist);
        }
      }
      // -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
      
      
      // -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
      // output diff density left > right = 358 else 350
      if (dbo.dl(D_DIFFDENS)) {
        double leftD = leftNum / leftDist;
        double rightD = rightNum / rightDist;
        
        if (leftD > rightD) {
          dbo.dpln(""+ cutValue+" "+358);
        } else {
          dbo.dpln(""+ cutValue+" "+350);
        }
      }
      // -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
      
      // see if a new maximum
      if (newLLK > split.newLLK) {
        
        // test if illegal cut
        boolean forbiddenCut = false;
        if (m_forbidIllegalCut && 
            ((BinningUtils.isIllegalCut(leftNum, leftDist, totalL, totalN)) 
                || (BinningUtils.isIllegalCut(rightNum, rightDist, totalL, totalN)))) 
          forbiddenCut = true;
        
        // test if other forbidden cuts
        if ((m_forbiddenCut != M_NOFORBIDDENCUT) && 
            ((isForbiddenCut(leftNum, leftDist, totalL, totalN)) 
                || (isForbiddenCut(rightNum, rightDist, totalL, totalN)))) 
          forbiddenCut = true;
        
        
        
        if (!forbiddenCut) {
          // current cut is new minimum 
          split.rightFlag = rightFlag;
          split.newLLK = newLLK;
          split.leftLLK = leftLLK;
          split.rightLLK = rightLLK;
          split.trainLLKDiff = (newLLK - oldLLK); // / (double)numInBin;
          split.index = ii;
          split.cutValue = cutValue;
          split.leftDist = leftDist;
          split.rightDist = rightDist;
          split.leftNum = leftNum;
          split.rightNum = rightNum;
        }
      }
      
      // toggle or set next cut point
      if (getEpsilonCutting()) {
        if ((ii == end) && (numEpsilonCuts == 2)) {
          ///dbo.dpln("#is leftEnd");
          ; // do nothing to make cut after last point possible
        } else {
          // move to next instance
          if (numEpsilonCuts == 2) {
            ////dbo.dpln("#set to 0" + lastCutValue);
            numEpsilonCuts = 0;
            ii++;
          } 
        }
        ///dbo.dpln("#BOT cutting ii:"+ii+" cutvalue "+cutValue+
        ///	 " numepsiloncuts "+numEpsilonCuts+" lastCutValue "+lastCutValue);
      } else {
        // if it was epsiloncut dont toggle and don't increase ii;
        if (toggle) {
          rightFlag = !rightFlag;
          if ((testRun % 2) == 0) { ii++; } 
        } else {
          ii++;
        }
      }
      //dbo.dpln("ii "+ii+" leftEnd "+leftEnd);
      
    } // leftEnd of the big while loop
    //xx    dbo.dpln("# new split "+ split.cutValue+" rightFlag "+split.rightFlag+
    //xx     "\n#"+" leftNum "+split.leftNum+" rightNum "+split.rightNum+"\n# instance"+
    //xx     data.instance(split.index)+" cutvalue "+split.cutValue);
    
    if (split.rightFlag) {
      split.lastLeft = split.index - 1;
      split.firstRight = split.index;
    } else {
      split.lastLeft = split.index;
      split.firstRight = split.index + 1;
    }
    
    // -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
    // output where split was found
    if (dbo.dl(D_ABOUTSPLIT)) {
      dbo.dpln("# max found at "+split.leftNum+" "+split.newLLK+" cut@ "+split.cutValue +
          " minValue "+minValue+" maxValue "+maxValue);
      if (BinningUtils.isIllegalCut(split.leftNum, split.leftDist, totalL, totalN)) {
        dbo.dpln("#is illegal left\n\n");
      }
      if (BinningUtils.isIllegalCut(split.rightNum, split.rightDist, totalL, totalN)) {
        dbo.dpln("#is illegal right\n\n");
      }
    }
    // -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
    
    return split;  
  }
  
  /*
   * Find the cut point in the range with min criteria
   * @param gridBegin left most = leftBegin value of the grid
   * @param bin the bin where the split should be found
   * @param data the data set, used for data model information
   * @param attrIndex the index of the attribute that is discretized
   * @param cutPoints the values order in increasing way
   * @return how much the number of bins has increased in this branch
   */
  protected Split findMinCriteria(double gridBegin, Bin bin, Instances data, int attrIndex, 
      double minValue, double maxValue, double bigN) {  
    
    if (dbo.dl(D_ABOUTSPLIT)) {
      if (getLeastSquaresCut()) {
        DBO.p("#MISE-criteria: ");
      } else {
        DBO.p("#LLK -criteria: ");
      }
      DBO.pln("find min in range "+minValue+"--"+maxValue);
    }
    
    int begin = bin.getBegin();
    int end = bin.getEnd();
    
    //int numInBin = bin.getWeight();
    int totalN = (int)bin.getTotalNum();
    double totalL = bin.getTotalLen();
    double numInst = bin.getWeight();
    //dbo.dpln("findMinCriteria "+leftBegin+" -" +leftEnd+" // " +minValue+" - " +maxValue+" numinst "+numInst);
    
    double leftNum = 0.0;
    double rightNum = numInst;
    double leftDist = 0.0;
    // double rightDist; 
    double newLLK = -Double.MAX_VALUE;
    double leftLLK = 0.0;
    double rightLLK = 0.0;
    double cutValue = 0.0;
    double lastCutValue = Double.NaN;
    
 
    // compute first criteria
    double oldLLK = getCriteriaFromBin(bin);
    
    // prepare split object
    Split split = new Split();
    split.newLLK = -Double.MAX_VALUE;
    split.oldLLK = oldLLK;
    split.index = -1;
    if (begin > end) return split;
        
    split.leftDist = 0.0;
    
    // offset is the index in the instance list
    int offset = begin;
    cutValue = getFirstCutValue(gridBegin, minValue);
    if (cutValue <= minValue)
      cutValue = getNextCutValue(cutValue);
      
    if (cutValue > maxValue) return split;
    leftDist = cutValue - minValue;
    double rightDist = maxValue - cutValue; 
    if (rightDist <= 0.0) return split;

    // find the offset of the last point left of the cut
    offset = findNewOffset(cutValue, offset, end, data, attrIndex);      
    
    leftNum = offset - begin + 1;
    rightNum = numInst - leftNum;
        
   // **************************************************************************
    // check over all instances starting from the first cutpoint within the range
    // from the left range border, cutting at equal distances 
    // dbo.dpln("######minValue "+minValue+" maxValue "+maxValue+" leftBegin "+leftBegin+" leftEnd "+leftEnd+"  ii "+ii+" rightDist "+rightDist);
    while ((rightDist > 0.0) && (cutValue < maxValue)) {
      // get entropy value for this cut
      if (getLeastSquaresCut()) {
        newLLK = getLeastSquaresCutCriteria(leftNum, leftDist, rightNum, rightDist, 
            m_theN);
      } else {
        newLLK = getCutCriteria(leftNum, leftDist, rightNum, rightDist, m_theN);
      }
           
      // -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
      // output all LLKs
      if (dbo.dl(D_ABOUTSPLIT)) {
        double squaredLK = 0.0;
        if (getLeastSquaresCut()) {
          rightLLK = rightNum * getLikelihood(rightNum, rightDist, m_theN) /  m_theN;
          leftLLK = leftNum * getLikelihood(leftNum, leftDist, m_theN) / m_theN;
          squaredLK = getSquaredCutCriteria(rightNum, rightDist, leftNum, leftDist, m_theN);
          
        } else {
          rightLLK = getCutCriteria(rightNum, rightDist, m_theN);
          leftLLK = getCutCriteria(leftNum, leftDist, m_theN);
        }
        
        dbo.dpln(""+cutValue + " "+ newLLK +" "+leftLLK+" "+rightLLK+" "+squaredLK+
            " left "+leftNum+"/"+leftDist+" right "+rightNum+"/"+rightDist+
            " cut# "+cutValue);
      }
      // -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
         
      // see if a new maximum
      if (newLLK > split.newLLK) {
        
        // test if illegal cut
        boolean forbiddenCut = false;
        if (m_forbidIllegalCut && 
            ((BinningUtils.isIllegalCut(leftNum, leftDist, totalL, totalN)) 
                || (BinningUtils.isIllegalCut(rightNum, rightDist, totalL, totalN)))) 
          forbiddenCut = true;
        
        // test if other forbidden cuts
        if ((m_forbiddenCut != M_NOFORBIDDENCUT) && 
            ((isForbiddenCut(leftNum, leftDist, totalL, totalN)) 
                || (isForbiddenCut(rightNum, rightDist, totalL, totalN)))) 
          forbiddenCut = true;
               
        if (!forbiddenCut) {
          // current cut is new minimum 
          split.newLLK = newLLK;
          split.leftLLK = leftLLK;
          split.rightLLK = rightLLK;
          split.trainLLKDiff = (newLLK - oldLLK); // / (double)numInBin;
          split.index = offset;
          split.cutValue = cutValue;
          split.leftDist = leftDist;
          split.rightDist = rightDist;
          split.leftNum = leftNum;
          split.rightNum = rightNum;
        }
      }
  
      // find next cut value
      // offset is the index in the instance list
      cutValue = getNextCutValue(cutValue);
      if (cutValue > maxValue) break;
      leftDist = cutValue - minValue;
      rightDist = maxValue - cutValue; 
      if (rightDist <= 0.0) break;

      offset = findNewOffset(cutValue, offset, end, data, attrIndex);      
      
      leftNum = offset - begin + 1;
      rightNum = numInst - leftNum;             
    } // leftEnd of the big while loop
     
    if (split.rightFlag) {
      split.lastLeft = split.index - 1;
      split.firstRight = split.index;
    } else {
      split.lastLeft = split.index;
      split.firstRight = split.index + 1;
    }
    
    // -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
    // output where split was found
    if (dbo.dl(D_ABOUTSPLIT)) {
      dbo.dpln("# max found at "+split.leftNum+" "+split.newLLK+" cut@ "+split.cutValue +
          " minValue "+minValue+" maxValue "+maxValue);
      if (BinningUtils.isIllegalCut(split.leftNum, split.leftDist, totalL, totalN)) {
        dbo.dpln("#is illegal left\n\n");
      }
      if (BinningUtils.isIllegalCut(split.rightNum, split.rightDist, totalL, totalN)) {
        dbo.dpln("#is illegal right\n\n");
      }
    }
    // -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
    
    return split;  
  }
  
 
  /**
   * Return the number of almost empty bins (density < 0,1).
   * @param bins the bins for the given attribute
   * @return the number of almost empty bins
   */
  public double getNumAlmEmptyBins (Vector bins, double total) throws Exception {
    
    int num = 0;
    Bin bin = null;
    double p;
    //dbo.dp("#");
    for (int j = 0; j < bins.size(); j++) {
      bin = (Bin) bins.elementAt(j);
      p = (bin.getWeight() * 100.0) / total;
      if (p <= 1.0) { num++; }
    }
    //dbo.dpln("");
    return (double)num;
  }
  
  /**
   * Return the sum of total errors.
   * @param bins the bins for the given attribute
   * @return the number of total errors
   */
  public double getErrorsFromBins (Vector bins) throws Exception {
    
    double err = 0.0;
    Bin bin = null;
    for (int j = 0; j < bins.size(); j++) {
      bin = (Bin) bins.elementAt(j);
      err += bin.getError();
    }
    return (double)err;
  }
  
  /**
   * Use the given bins and fill the instances into the given bins. Returns the
   * average loglikelihood, which is the sum of loglikelihoods divided by the 
   * number of instances.
   * @param bins the bins for the given attribute
   * @param test a set of instances, need not be the one the bins have been build for
   * @param index the index of the attribute the loglikelihood is asked for
   */
  public double getLoglkFromBins(Vector bins, Instances test, int index) {
    BinningUtils.fillBins(bins, test, index);
    double loglk = 0.0;
    Bin bin = null;
    
    // compute likelihood
    for (int i = 0; i < bins.size(); i++) {
      bin = (Bin) bins.elementAt(i);
      //dbo.dpln("Bin "+i+" has "+bin.getNumInst()+" instances");
      try {
        loglk += bin.getLoglikelihood();
        // -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
        if (dbo.dl(D_BININFO)) {
          double width = bin.getMaxValue() - bin.getMinValue();
          dbo.dpln(bin.getWeight()+", "+width+", "+bin.getTotalNum());
        }
        // -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
        
      } catch (Exception ex) {
        ex.printStackTrace();
        System.out.println(ex.getMessage());
      }
    }
    
    // divide by number of instances
    double num = (double) test.numInstances();
    loglk = loglk / num;
    return loglk;
  }
  
  /**
   * Use the given bins and fill the instances into the given bins. Returns the
   * average loglikelihood, which is the sum of loglikelihoods divided by the 
   * number of instances.
   * @param bins the bins for the given attribute
   * @param numTrainInstances the number of instances
   */
  public double getSquaredLkFromBins(Vector bins, int numTrainInstances) {
    //EstimatorUtils.fillBinsWithTrain(bins);
    double slk = 0.0;
    Bin bin = null;
    
    // compute likelihood
    for (int i = 0; i < bins.size(); i++) {
      bin = (Bin) bins.elementAt(i);
      //dbo.dpln("Bin "+i+" has "+bin.getNumInst()+" instances and weight "+bin.getWeight());
      try {
        slk = slk + bin.getSquaredArea(); 
        //dbo.dpln("slk "+slk);
      } catch (Exception ex) {
        ex.printStackTrace();
        System.out.println(ex.getMessage());
      }
    }
    
    // divide by number of instances
    //slk = slk / numTrainInstances;
    return slk;
  }
  
  /**
   * Use the given bins and fill the instances into the given bins. Returns the
   * average loglikelihood, which is the sum of loglikelihoods divided by the 
   * number of instances.
   * @param bins the bins for the given attribute
   * @param test a set of instances, need not be the one the bins have been build for
   * @param index the index of the attribute the loglikelihood is asked for
   */
  public double getLkFromBins(Vector bins, Instances test, int index) {
    
    BinningUtils.fillBins(bins, test, index);
    double lk = 0.0;
    Bin bin = null;
    
    // compute likelihood
    for (int i = 0; i < bins.size(); i++) {
      bin = (Bin) bins.elementAt(i);
      try {
        lk += bin.getLikelihood(); 
      } catch (Exception ex) {
        ex.printStackTrace();
        System.out.println(ex.getMessage());
      }
    }
    
    // divide by number of instances
    double num = (double) test.numInstances();
    lk = lk / num;
    return lk;
  }
  
  /**
   * get the Llk for the trainingsdata from the given bins. Returns the
   * average loglikelihood, which is the sum of loglikelihoods divided by the 
   * number of instances.
   * @param bins the bins for the given attribute
   * @param index the index of the attribute the loglikelihood is asked for
   */
  public double getTrainLoglkFromBins(Vector bins, int index, double bigN) {
    
    double lk = 0.0;
    Bin bin = null;
    
    // compute likelihood
    for (int i = 0; i < bins.size(); i++) {
      bin = (Bin) bins.elementAt(i);
      try {
        lk += bin.getWeightLoglikelihood(); 
      } catch (Exception ex) {
        ex.printStackTrace();
        System.out.println(ex.getMessage());
      }
    }
    
    // divide by number of instances
    lk = lk / bigN;
    return lk;
  }
  
  /**
   * Get the current cut value.
   * @param data current data set
   * @param attrIndex the index of the attribute used 
   * @param offset
   * @param middleFlag
   * @return 
   */
  private double getCutValue(Instances data, int attrIndex, int offset, boolean middleFlag) { 
    
    double cutValue;
    if (!middleFlag) { 
      cutValue = data.instance(offset).value(attrIndex);
    } else {
      if (offset >= data.numInstances() - 1) {
        cutValue = data.instance(offset).value(attrIndex);
      } else {
        cutValue = (data.instance(offset).value(attrIndex) 
            + data.instance(offset + 1).value(attrIndex)) / 2.0;
      } 
    }
    return cutValue;
  }
 
  /**
   * Get the next cut value, for the gridcut method.
   * @param oldCut the old cut value
   * @return 
   */
  private double getNextCutValue(double oldCut) { 
    
    double cutValue = oldCut + m_gridWidth;
    return cutValue;
  }
  
  /**
   * Get the first leftmost cut value using the grid method.
   * @param gridStart start = left most of grid
   * @param minValue minimal value in range
   * @return the first cut value in the range
   */
  private double getFirstCutValue(double gridStart, double minValue) { 
    double cutValue = gridStart;
    if (cutValue < minValue) {
      double gridNums = Math.ceil((minValue - gridStart) / m_gridWidth); 
       cutValue = gridStart + (gridNums * m_gridWidth);
      if (cutValue <= minValue) cutValue += m_gridWidth;
    }
    return cutValue;
  }
  
  /**
   * Get the current cut value.
   * @param cutValue current cut value
   * @param oldOffset the old offset
   * @param endOffset the last offset in the range
   * @param data current data set
   * @param attrIndex the index of the attribute used 
   * @return the new offset
   */
  private int findNewOffset(double cutValue, int oldOffset, int endOffset, Instances data, int attrIndex) { 
    
    int ii = oldOffset;
    double value = data.instance(ii).value(attrIndex);
    boolean newFound = false;
    while (!newFound) {
      ii++;
      if (ii >= endOffset) {
        newFound = true; ii--;
        break;
      }
      value = data.instance (ii).value(attrIndex);
      if (value >= cutValue) {
        newFound = true; ii--;
        break;
      }
    }  
    
    return ii;
  }
  
  /**
   * Get the current cut value.
   * @param data
   * @param index
   * @param offset
   * @param middleFlag
   * @return 
   */
  private double getDiffInBetween(Instances data, int attrIndex, 
      int offset, int divide) { 
    double diff = 0.0;
    if (offset < data.numInstances() - 1) {
      diff = data.instance(offset + 1).value(attrIndex) 
      - data.instance(offset).value(attrIndex);
      diff = (diff / (double)divide);
    }     
    return diff;
  }
  
  /*
   * Compute the criterion for one cutpoint. 
   *
   * @param num number of instances left off the cut point
   * @param length volume (length) of the area 
   * @param numRight number of instances in that area
   */
  protected  double getCriteriaFromBin(Bin bin) {
    double criteria = 0.0;
//  ***
//  if (getLeastSquaresCV()) {
//  criteria = bin.getSquaredLikelihood();
//  double criteria2 = bin.getLikelihood();
//  criteria = criteria - 2.0 * criteria2;
//  } else {
    try {
      //dbo.dpln("getCriteriaFromBin");
      criteria = bin.getWeightLoglikelihood();
    } catch (Exception ex) {
      ex.printStackTrace();
      System.out.println(ex.getMessage());
    }
    //   }
    return criteria;
  }
  
  /*
   * Compute the criteria for one cutpoint. 
   *
   * @param rightNum number of instances right to the cut point
   * @param rightWidth length of the right range 
   * @param leftNum number of instances left off the cut point
   * @param rightNum volume (length) of the area 
   * @param totalNum number of instances in the total range
   */
  protected  double getCutCriteria(double rightNum, double rightWidth, 
      double leftNum, double leftWidth, double totalNum) {
    double rCriteria = getLoglikelihood(rightNum, rightWidth, totalNum);
    double lCriteria = getLoglikelihood(leftNum, leftWidth, totalNum);
    double criteria = rCriteria + lCriteria;
    return criteria;
  }
  
  /*
   * Compute the criteria for one cutpoint. 
   *
   * @param num number of instances left off the cut point
   * @param length volume (length) of the area 
   * @param numRight number of instances in that area
   */
  protected  double getCutCriteria(double num, double width, double totalNum) {
    if (num == 0.0) return 0.0;
    double criteria = 0.0;
    criteria = getLoglikelihood(num, width, totalNum);
    return criteria;
  }
  
  /*
   * Compute the mise criteria for one cutpoint. 
   *
   * @param rightNum number of instances right to the cut point
   * @param rightWidth length of the right range 
   * @param leftNum number of instances left off the cut point
   * @param leftNum volume (length) of the area 
   * @param totalNum number of instances in the total range
   */
  protected  double getLeastSquaresCutCriteria(double rightNum, double rightWidth, 
      double leftNum, double leftWidth, 
      double totalNum) {
    double rCriteria = getLikelihood(rightNum, rightWidth, totalNum);
    double lCriteria = getLikelihood(leftNum, leftWidth, totalNum);
    double lk = ((rightNum * rCriteria)  + (leftNum * lCriteria)) / totalNum;
    
    double squared = getSquaredCutCriteria(rightNum, rightWidth, leftNum,  leftWidth, totalNum);
    double criteria = - squared + 2.0 * lk;
    return criteria;
  }
  
  /*
   * Compute the suared part of the least squares cut criteria. 
   *
   * @param rightNum number of instances right to the cut point
   * @param rightWidth length of the right range 
   * @param leftNum number of instances left off the cut point
   * @param leftNum volume (length) of the area 
   * @param totalNum number of instances in the total range
   */
  protected  double getSquaredCutCriteria(double rightNum, double rightWidth, 
      double leftNum, double leftWidth, 
      double totalNum) {
    double rSquared = getSquaredArea(rightNum, rightWidth, totalNum);
    double lSquared = getSquaredArea(leftNum, leftWidth, totalNum);
    double squared = rSquared + lSquared;
    return squared;
  }
  
  /*
   * Compute the squared likelihood for one cutoff area.
   * @param num number of instances left off the cut point
   * @param length volume (length) of the area 
   * @param numRight number of instances in that area
   */
  protected  double getSquaredLikelihood(double num, double width, double totalNum) {
    if (num == 0.0) return 0.0;
    double llh = num / (width * totalNum);
    return llh * llh;
  }
  
  /*
   * Compute the squared likelihood for one cutoff area.
   * @param num number of instances left off the cut point
   * @param length volume (length) of the area 
   * @param numRight number of instances in that area
   */
  protected  double getSquared(double num, double width, double totalNum) {
    if (num == 0.0) return 0.0;
    double llh =  num / totalNum;
    llh = llh * llh;
    llh =  llh / width;
    return llh;
  }
  
  /*
   * Compute the squared likelihood for one cutoff area.
   * @param num number of instances left off the cut point
   * @param length volume (length) of the area 
   * @param numRight number of instances in that area
   */
  protected  double oopsgetSquared(double num, double width, double totalNum) {
    if (num == 0.0) return 0.0;
    double llh =  num / totalNum;
    dbo.dp(""+ num+ " "+width+" n/N "+llh);
    llh = llh * llh;
    dbo.dp(" sq "+llh);
    llh =  llh / width;
    dbo.dpln(" /width "+llh);
    return llh;
  }
  
  /*
   * Compute the density for one cutoff area. 
   *
   * @param num number of instances in the bin
   * @param width width of the bin 
   * @param totalNum total number of instances in that area
   */
  protected double getDensity(double num, double width, double totalNum) {
    if (num == 0.0) return 0.0;
    double llh = num / (width * totalNum);
    // double llh = num / (totalNum);
    return llh;
  }
  
  /*
   * Compute the likelihood for one cutoff area. 
   *
   * @param num number of instances left off the cut point
   * @param length volume (length) of the area 
   * @param numRight number of instances in that area
   */
  protected  double getLikelihood(double num, double width, double totalNum) {
    if (num == 0) return 0.0;
    double llh = num / (width * totalNum);
    return llh;
  }
  
  /**
   * Get area underneath the squared likelihood.
   * @param num number of instances in the sub range
   * @param width width of the sub range of the area 
   * @param totalNum number of instances in that area
   * @return the area underneath the squared likelihood 
   */
  public double getSquaredArea(double num, double width, double totalNum) {
    if (num == 0) return 0.0;
    double llh = num / (width * totalNum);
    double squared = llh * llh;
    squared = squared * width;
    //    Oops.pln("squared "+squared  );    
    return squared;
  }
  
  /*
   * Compute the loglikelihood for one cutoff area. 
   *
   * @param num number of instances left off the cut point
   * @param width width of the sub range of the area 
   * @param totalNum number of instances in that area
   */
  protected  double getLoglikelihood(double num, double width, double totalNum) {
    //dbo.dpln("getLoglikelihood num "+num+" width "+width+" totalNum "+totalNum);
    if (num == 0) return 0.0;
    double density = num / (width * totalNum);
    double llk  = (num) * (Math.log(density));
    
    //dbo.dpln("Loglikelihood "+llk);
    return llk;
  }
  
 /**
  * Tests if split can be accepted
  * 
  * @param sp global splitting infos
  * @param oldEntropy entropy without the split
  * @param splitEntropy entropy after the split
  * @param numLeft number of instances left
  * @param numRight number of instances right
  * @param bigN the total number of instances in data set
  * @param lenLeft the length of left part
  * @param lenRight the length og the right part
  * @return true if split is accepted
  */ 
   protected boolean splitAccepted(SpInfos sp,
      double oldEntropy, double splitEntropy, 
      double numLeft,  double numRight, double bigN, 
      double lenLeft, double lenRight) {
    //dbo.dpln("splitAccepted: oldEntropy "+oldEntropy+
    //	     " splitEntropy "+splitEntropy);
    double sum = numLeft + numRight;
    double penalty = 0.0;
    
    /** output difference of entropys, before/after split
     double diff = (oldEntropy - splitEntropy);
     dbo.dpln(" || DIFF "+ diff+" penalty "+penalty);*/
    
    //dbo.dpln("m_splitMethod "+m_splitMethod);
    switch (m_splitMethod) {
    case STANDARD_SPLIT:
      //dbo.dpln("STANDARD_SPLIT");
      penalty = Math.log(m_theN) + Math.log(2.0);
      break;
    case CV_SPLIT:
      //dbo.dpln("CV_SPLIT");
      // decision is made before the split is selected
      return true;
    case FULL_SPLIT:
      //dbo.dpln("FULL_SPLIT");
      penalty = 0.0;
      break;
      
    case WEIRD_SPLIT:
      //dbo.dpln("WEIRD_SPLIT");
      // todo + instead of minus
      penalty =  - (bigN * (Math.log(sum / bigN))) / 10.0;
      break;
    case NOEMPTY_SPLIT:
      //dbo.dpln("NOEMPTY_SPLIT");
      if ((numRight == 0.0) || (numLeft == 0.0)) return false;
      return true;
    }// switch
    
    //** output if invalid cut at the border was attempted
    if (dbo.dl(D_ILLCUT)) {
      if (BinningUtils.isIllegalCut((int)numRight, lenRight, sp.bigL, sp.bigN)) {
        double l = lenRight / sp.bigL;
        dbo.dp(" Illegal cut RIGHT:");
        dbo.dpln("\nl/l " + l +" len "+lenRight+" num "+numRight);
      }
      if (BinningUtils.isIllegalCut((int)numLeft, lenLeft, sp.bigL, sp.bigN)) {
        double l = lenLeft / sp.bigL;
        dbo.dp(" Illegal cut LEFT:");
        dbo.dpln("\nl/l " + l +" len "+lenLeft+" num "+numLeft);
      }
    }
    
    // illegal cut
    if (BinningUtils.isIllegalCut((int)numRight, lenRight, sp.bigL, sp.bigN) 
        || BinningUtils.isIllegalCut((int)numLeft, lenLeft, sp.bigL, sp.bigN)) {
      //m_numIllegalCuts++;
      if (m_forbidIllegalCut) { return false; }
    }
    
    if (oldEntropy < (splitEntropy - penalty)) {
      // make the split
      return true;
    }
    
    if (dbo.dl(D_FOLLOWSPLIT)) {
      dbo.dpln("#entropy didn't increase ");
      dbo.dpln("#penalty "+penalty+" oldEntropy "+oldEntropy+" splitEntropy "+splitEntropy);
    }
    
    // entropy didn't decrease
    sp.numTotallyUniform++;
    
    // reject the split
    return false;
  }
  
  /**
   * List cutpoints and left flags after bins
   * @param bins the bins to be used
   * @param cutPoints array of double to hold the cut values
   * @param cutAndLeft array of flags, true if left bin is closed
   * @exception arrays cutPoints and cutAndLeft not of the right size
   */
  public void binsToCutPoints (Vector bins, double [] cutPoints,
      boolean [] cutAndLeft) throws Exception {
    
    // transform bins into cutpoints
    if (bins != null) {
      
      if ((cutPoints.length != bins.size() - 1) ||
          (cutAndLeft.length != bins.size() - 1)) 
        throw new Exception("Error in binsToCutPoints: array not the right size.");
      for (int i = 0; i < cutPoints.length; i++) {
        cutPoints[i] = ((Bin)m_bins.elementAt(i)).getMaxValue();
        cutAndLeft[i] = ((Bin)m_bins.elementAt(i)).getMaxIncl();
      }
    }
    
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
  public Vector cutPointsToBins(double [] cutPoint, boolean [] cutAndLeft,
      Instances test, int attrIndex, 
      double min, double max) {
    
    Bin bin = null;
    double totalLen = max - min;
    int totalNum = test.numInstances();
    
    Vector bins = new Vector();
    // first bin
    bin = new Bin(1, 
        totalNum, m_numInstForIllCut, totalLen,
        0, 0, 0,
        // num not yet given, the other two are irrelevant
        min, true, cutPoint[0], cutAndLeft[0],
        0.0, getAlpha());
    bins.add(bin);
    for (int i = 0; i < cutPoint.length; i++) {
//    public Bin(int splitDepth,
//    double totalNum, double totalLen,
//    double num, int leftBegin, int leftEnd,
//    double min, boolean minIn, double max, boolean maxIn,
//    double entropy, double alpha) {
      bin = new Bin(1, 
          totalNum, m_numInstForIllCut, totalLen,
          0, 0, 0,
          // num not yet given, the other two are irrelevant
          cutPoint[i - 1], !cutAndLeft[i - 1], cutPoint[i], cutAndLeft[i],
          0.0, getAlpha());
      bins.add(bin);
    }    
    bin = new Bin(1, 
        totalNum, m_numInstForIllCut, totalLen,
        0, 0, 0,
        // num not yet given, the other two are irrelevant
        cutPoint[cutPoint.length - 1], cutAndLeft[cutPoint.length - 1], 
        max, true,
        0.0, getAlpha());
    bins.add(bin);
    
    // fill instances into bins
    nextInstance:
      for (int i = 0; i < test.numInstances(); i++) {
        if (test.instance(i).isMissing(attrIndex)) {
          break;
        }
        double value = test.instance(i).value(attrIndex);
        
        // test if in first bin     
        if ((value < cutPoint[0]) || 
            ((value == cutPoint[0]) && cutAndLeft[0])) {
          ((Bin)bins.elementAt(0)).addBorderInstance(true, value);
          //dbo.dpln("#"+value +" into bin 0"); nn++;
          continue nextInstance;
        }
        // find bin for the current value
        for (int j = 1; j < bins.size() - 1; j++) {
          if ((value < cutPoint[j]) || 
              ((value == cutPoint[j]) && cutAndLeft[j])) {
            ((Bin)bins.elementAt(j)).addInstance();
            //dbo.dpln("#"+value +" into bin "+j); nn++;
            continue nextInstance;
          }
        }
        
        // into the last bin
        ((Bin)bins.elementAt(0)).addBorderInstance(true, value);
        //int jjj = bins.size() - 1;
        //dbo.dpln("#"+value +" into bin " + jjj); nn++;
      }
    
    return bins;    
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
    return result;
  }

 
  /**
   * Main method for testing this class.
   *
   * @param argv should contain the options for the estimator
   */
  public static void main(String [] argv) {
    
    try {
//    DBO.pln("argument 0 "+argv[0]);
//    DBO.pln("argument 1"+argv[1]);
//    DBO.pln("");
      
      TUBEstimator est = new TUBEstimator();
      
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
