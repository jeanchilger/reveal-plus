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
 *    BinningEstimator.java
 *    Copyright (C) 2004
 *
 */

package weka.estimators;

import java.util.Enumeration;
import java.util.Vector;

import weka.core.Instances;
import weka.core.Option;
import weka.core.Utils;
import weka.core.Debug.DBO;
import weka.filters.unsupervised.attribute.Bin;
import weka.filters.unsupervised.attribute.CutInfo;

/**
 <!-- globalinfo-start -->
 * Abstract class of an estimator that creates bins.
 <!-- globalinfo-leftEnd -->
 *
 * @author Gabi Schmidberger (gabi dot schmidberger at gmail dot com)
 * @version $Revision: 1.0 $
 */
public abstract class BinningEstimator extends Estimator 
   {

  /** additional output element (for verbose modes) */
  protected DBO dbo = new DBO();

  /** print a histogram file */
  public static int D_HISTOGRAMS      = 7; // 8

  /** trace through precedures */
  public static int D_TRACE           = 14; // 15

  /** trace through precedures */
  protected String m_filePostfix = null;

  /** the bins that result from discretization*/
  protected Vector m_bins;

  /** The minimalvalue */
  protected double m_minValue;

  /** The maximalvalue */
  protected double m_maxValue;

  /** number of illegal cuts in CV in average */
  protected double m_avgCVIllegalCuts = 0.0;

  /** number of illegal cuts in final model */
  protected double m_numIllegalCuts = 0.0;

  /** difference between the number of illegal cuts in CV and leftEnd model */
  protected double m_diffNumIllegalCuts = 0.0;

  /** filename used for output */
  protected String m_fileName = null;

  /** The amount of noise added */
  protected double m_alpha = 1.0;

  /** number of illegal cuts in final model */
  protected double m_numInstForIllCut = -1.0;
  
  /** result llk */
  protected double m_resultLLK = Double.NaN;

  /** Constructor */
  public BinningEstimator() {

    // for the verbose (DBO) output
    dbo.initializeRanges(30);
  }

  /**
   * Switches the outputs on that are requested from the option V
   * if list is empty switches on the verbose mode only
   * @param list list of integers, all are used for an output type
   */
  public void setVerboseLevels(String list) { 
    dbo.setOutputTypes(list);
  }

  /**
   * Gets the current output type selection
   *
   * @return a string containing a comma separated list of ranges
   */
  public String getVerboseLevels() {
    return dbo.getOutputTypes();
  }

 /**
   * Sets the filename for output
   * @param n the new file name
   */
  public void setFileName(String n) {
    m_fileName = n;
  }

  /**
   * Returns the filename for all info output
   * @return filename that would be used for info output
   */
  public String getFileName() {
    return m_fileName;
  }

  /**
   * Get the cut points for an attribute and the boundary information.
   * @return an object containing the cutpoints and info if boundaries
   * are open or closed
   */
  public CutInfo getCutInfo() {
    
    // no cutpoints here
    if (m_bins == null) return null;
    
    // transform bins into cutpoints
    CutInfo info = BinningUtils.binsToCutInfo(m_bins);
    return info;
  }
  

  /**
   * Returns the bins.
   * @return a vector containing the bins.
   */
  public Vector getBins() {
    if (m_bins == null) {
      return null;
    }
    return m_bins;
  }

  /**
   * Get the minimal value of this estimator
   * @return the minimal value of an attribute
   */
  public double getMinValue() {

    return m_minValue;
  }

  /**
   * Get the maximal value of this estimator
   * @return the maximal value of an attribute
   */
  public double getMaxValue() {

    return m_maxValue;
  }

  /**
   * Return a copy of the bins 
   *
   * @return a copy of the bins
   */
  public Vector getBinsCopy () {

    Vector bins = new Vector();
    Vector oldBins = getBins();
    for (int i = 0; i < oldBins.size(); i++) {
      Bin bin = (Bin)oldBins.elementAt(i);
      bins.add(bin);
    }
    return bins;
  }  

  /**
   * Return average (over cv) number of illegalCuts.
   *
   * @return the avg number of illegal cuts
   */
  public double getAvgCVIllegalCuts() {
    
    return m_avgCVIllegalCuts;
  }

  /**
   * Return number of illegal cuts.
   *
   * @return the number of illegal cuts
   */
  public double getNumIllegalCuts() {
    
    return m_numIllegalCuts;
  }
  /**
   * Return difference between number of illegal cuts at CV time and final model building.
   *
   * @return the difference 
   */
  public double getDiffNumIllegalCuts() {
    
    return m_diffNumIllegalCuts;
  }

  /**
   * Sets the number of instances for illegal cut computation
   *
   * @param numInst  number of instances for illegal cut computation
   */
  public void setNumInstForIllCut(double numInst) {
    m_numInstForIllCut = numInst;
  }

  /**
   * Returns the number of instances for illegal cut computation
   *
   * @return max the maximum number of binss
   */
  public double getNumInstForIllCut() {
    return m_numInstForIllCut;
  }

  /**
   * Sets the value for the alpha uniform noise.
   * @param newValue the new alpha value
   */
  public void setAlpha(double newValue) {
    m_alpha = newValue;
  }

  /**
   * Gets the value for the alpha uniform noise.
   * @return the alpha value
   */
  public double getAlpha() {
    return m_alpha;
  }


  /**
   * Initialize the estimator with a new dataset.
   *
   * @param data the dataset used to build this estimator 
   * @param attrIndex attribute the estimator is for
   */
  public void addValues(Instances inst, int attrIndex, double min, double max,
      double factor) throws Exception{
    
    dbo.dpln(D_TRACE, "addValues -BinningEstimator");
    m_minValue = min;
    m_maxValue = max;
    
    // if number of instances is small then total number of instances, reduce alpha
    setAlpha(getAlpha() * factor);
    
    addValues(attrIndex, inst);
    
    // -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
    if (dbo.dl(D_HISTOGRAMS)) {
      try {
        writeHistogram(m_filePostfix, inst.relationName(), attrIndex);
      } catch (Exception ex) {
        ex.printStackTrace();	
        System.out.println(ex.getMessage());
      }
    }
    // -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
  }

  /**
   * Initialize the estimator with all values of one attribute of a dataset. 
   * Minimal and maximal value has been set already
   *
   * @param attrIndex attribute the estimator is for
   * @param data the dataset used to build this estimator 
   * @exception if building of estimator goes wrong
   */
  public abstract void addValues(int attrIndex, Instances data) throws Exception;
  
  /**
   * Actually just gets the width of the first or the second bin
   * Used to get the bin width of EW histograms,
   * Therefore it takes the second bin in case the first is smaller because of 
   * CV of origin
   *
   * @return the maximum density
   */
  public double getEWWidth() {

    //DBO.pln("bins in maxheight "+m_bins);
    
    double maxWidth = 0.0;

    // no data therefore no bins, max width is 0.0 
    if (m_bins == null) return maxWidth;

    // if there are more than one bin the first one could 
    int i = 0;
    if (m_bins.size() > 1) { 
      i = 1; 
    }
    Bin bin = (Bin) m_bins.elementAt(i);
    maxWidth = bin.getMaxValue() - bin.getMinValue();    
    return maxWidth;
  }

  /**
   * Get the maximum height of the histogram
   *
   * @return the maximum density
   */
  public double getMaxHeight() {

    //DBO.pln("bins in maxheight "+m_bins);
    
    double maxHeight = 0.0;

    // no data therefore no bins, max height is 0.0 
    if (m_bins == null) return maxHeight;

    // look through all bins
    for (int i = 0; i < m_bins.size(); i++) {
      Bin bin = (Bin) m_bins.elementAt(i);
      double dens = bin.getDensity();
      if (dens > maxHeight) {
	maxHeight = dens;
      }
    }
       
    return maxHeight;
  }

 
  /**
   * Tests whether the current estimation object is equal to another
   * estimation object
   *
   * @param obj the object to compare against
   * @return true if the two objects are equal
   */
  public boolean equals(Object obj) {
    
    if ((obj == null) || !(obj.getClass().equals(this.getClass()))) {
      return false;
    }
     if (!super.equals(obj)) return false;
    BinningEstimator cmp = (BinningEstimator) obj;
   
    if (m_filePostfix == null) {
      if (cmp.m_filePostfix != null) return false; 
    } else {
      if (!m_filePostfix.equals(cmp.m_filePostfix)) return false;
    }

    /** the bins that result from discretization*/
    if (!BinningUtils.equalBins(m_bins, cmp.m_bins)) return false;
    if (m_minValue != cmp.m_minValue) return false;
    if (m_maxValue != cmp.m_maxValue) return false;
    if (m_avgCVIllegalCuts != cmp.m_avgCVIllegalCuts) return false;
    if (m_numIllegalCuts != cmp.m_numIllegalCuts) return false;
    if (m_diffNumIllegalCuts != cmp.m_diffNumIllegalCuts) return false;
    if (m_fileName == null) {
      if (cmp.m_fileName != null) return false; 
    } else {
   if (!m_fileName.equals(cmp.m_fileName)) return false;
    }
    if (m_alpha != cmp.m_alpha) return false;
    if (m_numInstForIllCut != cmp.m_numInstForIllCut) return false;
    
    
    return true;
  }
  
  /**
   * Parses a given list of options. Valid options are:<p>
   *
   * -D  <br>
   * If set, estimator is run in debug mode and 
   * may output additional info to the console.<p>
   *
   * @param options the list of options as an array of strings
   * @exception Exception if an option is not supported
   */
  public void setOptions(String[] options) throws Exception {

    super.setOptions(options);

    // output data 
    String outputRange = Utils.getOption('V', options);
    setVerboseLevels(outputRange);

    // filename for verbose output
    String fileName = Utils.getOption('X', options);
    if (fileName.length() > 0)
      setFileName(fileName);

    // set uniform alpha noise
    String alphaString = Utils.getOption('A', options);
    if (alphaString.length() > 0)
      setAlpha(Double.parseDouble(alphaString));

    // min number of instances in a cut otherwise it is an illegal cut
    String numInstForIllCut = Utils.getOption('H', options);
    if (numInstForIllCut.length() != 0) {
      setNumInstForIllCut((double)Integer.parseInt(numInstForIllCut));
    } 
  }

  /**
   * Gets the current settings of the Estimator.
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

    // verbose levels
    String verboseLevels = getVerboseLevels();
    if (verboseLevels.length() > 0) {
      result.add("-V");
      result.add("" + verboseLevels);
    }
    // filename
    String filename = getFileName();
    if (filename != null) {
      if (filename.length() > 0) {
	result.add("-X");
	result.add("" + filename);
      }
    }

    // alpha
    if (getAlpha() != 1.0) {
      result.add("-A");
      result.add("" + getAlpha());
    }

    // min number of instances before illegal
    if (getNumInstForIllCut() > -1.0) {
      result.add("-H");
      result.add("" + getNumInstForIllCut());
    }
    
    return (String[])result.toArray(new String[result.size()]);
  }

  /**
   * Returns an enumeration describing the available options.
   *
   * @return an enumeration of all the available options.
   */
  public Enumeration listOptions() {
    
    Vector newVector = new Vector(4);
    newVector.addElement(
      new Option(
		 "\tSwitch on verbose mode and give list of output options.\n"
		 + "\teg: 1,2,11",
		 "V", 1, "-V <option list>"));
    newVector.addElement(
      new Option(
		 "\tFilename for verbose output\n"
		 + "\t(Default is up to 20 characters of relation name).",
		 "X", 1, "-X <file name>"));
    newVector.addElement(
      new Option(
		 "\tAlpha value, or number of instances uniformly spread over range\n"
		 + "\tDefault is 1.0.\"",
		 "A", 1, "-A <value>"));
    newVector.addElement(
      new Option(
		 "\tMinimal number of instances allowed in bin,\n"
		 + "\totherwise cut is avoided.\n",
		 "H", 1, "-H <num>"));

    Enumeration enu = super.listOptions();
    while (enu.hasMoreElements()) {
      newVector.addElement(enu.nextElement());
    }
    return newVector.elements();
  }

  /**
   * Writes data to file that can be used to plot a histogram.
   * Filename is aprameter f + ".hist".
   *
   *@param f string to build filename
   *@param bins vector of bins
   *@param identicalValue value, is NaN if not all values were identical
   */
    public void writeHistogram(String postfix, String relName, 
			       int attrIndex) throws Exception {
	
	StringBuffer filename = new StringBuffer(relName);
	if (m_fileName != null) {
	    filename = new StringBuffer(m_fileName);
	} else {
	    if (filename.length() > 20) {
		filename = new StringBuffer(filename.substring(0, 20));
	    }     
	}
	
	if (m_classValueIndex > -1) {
	    filename.append("-"+attrIndex+"-" + m_classValueIndex + postfix);
	} else {
	    filename.append("-"+attrIndex + postfix);
	}
	BinningUtils.writeHistogram(filename.toString(), 
				      m_bins, m_minValue, m_maxValue, 1.0);
    }
}
